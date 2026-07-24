"""Batched block-diffusion generator for LLaDA2.0 (inclusionAI/LLaDA2.0-mini) with ODD support.

This mirrors the model's native single-sample ``LLaDA2MoeModelLM.generate()``
(modeling_llada2_moe.py) but runs a whole batch of samples in parallel and lets an
ODD/DPP strategy (strategies.py) modify the generation-region logits before token
acceptance, exactly as DiverseGenerator (generator.py) does for LLaDA-8B.

Faithfulness to the native generate():
  * Same mask template: total_length = ceil((prompt+gen)/block) * block, filled with
    mask_id (156895), prompt copied in front.
  * Same block-diagonal causal attention mask (tril block mask, repeat_interleave,
    .log() -> additive bf16 mask), expanded to the batch dimension. The model forward
    requires the mask shape (B, 1, L, L) and fp32-casts logits internally.
  * Same position_ids = arange(total_length).
  * Same per-block iterative refinement with `steps` denoising steps and the
    _get_num_transfer_tokens schedule.
  * Same acceptance rule: sampled tokens with confidence > threshold are all
    committed; otherwise the top `num_transfer` highest-confidence tokens are
    committed (applied per batch row).
  * Same temperature / top-k / top-p sampler (verbatim copy of the native helpers).
  * Same eos handling: stop refining new blocks once eos appears (batched: once
    EVERY row has an eos), and trim each returned row at its first eos. Rows that
    finish (eos) early keep being carried through remaining blocks for the still
    unfinished rows; this is output-equivalent to the native per-row break because
    everything after the first eos is trimmed before decoding.

Documented deviations from the native code:
  * temperature == 0: the native sampler falls through to torch.multinomial on the
    un-tempered softmax (i.e. it is NOT greedy despite the docstring), which makes
    native temp-0 decoding stochastic. We use argmax (true greedy) at temperature 0
    for determinism. Pass ``exact_native_sampling=True`` to reproduce the native
    multinomial behaviour bit-for-bit (used by the parity check in llada2_smoke.py).
  * ODD guidance is computed in bf16 (``strategy_dtype``) on a copy of the
    generation-region logits, and the full fp32 logits tensor is freed while the
    strategy's autograd pass runs. This halves the several vocab-sized (157184)
    tensors the strategy materialises; on the ODD path the sampler therefore sees
    bf16-precision guided logits recast to fp32. The baseline path (alpha == 0)
    samples the untouched fp32 logits.
  * use_cache=False is passed to forward (the native code lets the model allocate a
    DynamicCache it never reads; logits are unaffected).

Alpha annealing choice: GLOBAL progress. DiverseGenerator anneals alpha with
gamma(t) = 1 - i/steps over its single denoising window. Block diffusion has
nested loops (blocks x denoising steps); we anneal over the global progress
  progress = (gen_block_index * steps + step) / (num_gen_blocks * steps)
so guidance is strongest at the start of the sequence and decays monotonically to 0
at the end, matching the single-window semantics of the paper. The alternative,
restarting the anneal inside every block, would re-inject full-strength guidance at
each block boundary and perturb late-sequence tokens where quality matters most; we
deliberately do not do that.

ODD scoping: at every denoising step the strategy sees logits/x/mask for the whole
generation region decoded so far, x[:, prompt_len:window_end] (committed tokens in
earlier blocks become one-hot features in FeatureExtractor, masked positions in the
active block contribute their softmax), mirroring how DiverseGenerator scopes
gen_logits to everything after the prompt.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

LLADA2_MASK_ID = 156895
LLADA2_EOS_ID = 156892  # also the pad token


def load_llada2(model_name: str = "inclusionAI/LLaDA2.0-mini", load_in_4bit: bool = True):
    """Load LLaDA2.0 with the same conventions as utils.load_model, but through
    AutoModelForCausalLM (the trunk-only AutoModel has no lm_head / logits)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


# ---- verbatim ports of the native sampling helpers (modeling_llada2_moe.py) ----

def _top_k_logits(logits, k):
    if k is None or k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(
        logits < min_values, torch.full_like(logits, float("-inf")), logits
    )


def _top_p_logits(logits, p):
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool),
        -1,
        sorted_indices,
        sorted_mask,
    )
    return logits.masked_fill(mask_indices, float("-inf"))


def _sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    logits = _top_k_logits(logits, top_k)
    logits = _top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    token_prob = torch.gather(probs, -1, token)
    return token.view(*orig_shape), token_prob.view(*orig_shape)


def _sample_greedy(logits):
    """Deterministic greedy sampling with the same (token, prob) contract."""
    probs = F.softmax(logits, dim=-1)
    token_prob, token = probs.max(dim=-1)
    return token, token_prob


def _get_num_transfer_tokens(block_length, steps):
    if steps == 0:
        return torch.tensor([], dtype=torch.int64)
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.full((steps,), base, dtype=torch.int64)
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


class LLaDA2DiverseGenerator:
    """DiverseGenerator-compatible batched generator for LLaDA2.0 block diffusion.

    Public interface mirrors generator.DiverseGenerator:
        generate(prompt, batch_size, steps, gen_length, temperature) -> ([], texts)
    """

    def __init__(self, model, tokenizer, strategy, mask_token_id: int = LLADA2_MASK_ID,
                 block_length: int = 32, threshold: float = 0.95,
                 eos_token_id: int = LLADA2_EOS_ID, early_stop: bool = True,
                 strategy_dtype: torch.dtype = torch.bfloat16):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.mask_token_id = mask_token_id
        self.block_length = block_length
        self.threshold = threshold
        self.eos_token_id = eos_token_id
        self.early_stop = early_stop
        self.strategy_dtype = strategy_dtype
        self.device = model.device

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
        self.protected_tokens = torch.tensor(
            sorted({eos_token_id, pad_id, mask_token_id}), device=self.device
        )
        self.last_sequences = None  # trimmed generated token ids per row (for debugging/parity)

    def encode_prompt(self, prompt: str, batch_size: int) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        encoded = self.tokenizer(
            [prompt_str] * batch_size, return_tensors="pt", padding=True,
            add_special_tokens=False,
        )
        return encoded.input_ids.to(self.device)

    def generate(self, prompt: str, batch_size: int, steps: int, gen_length: int,
                 temperature: float, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, minimal_topk: int = 1,
                 exact_native_sampling: bool = False) -> Tuple[List[Dict], List[str]]:
        prompt_ids = self.encode_prompt(prompt, batch_size)
        prompt_length = prompt_ids.shape[1]

        denoising_steps = min(steps, gen_length // minimal_topk)
        if denoising_steps <= 0:
            denoising_steps = 1

        num_blocks = (prompt_length + gen_length + self.block_length - 1) // self.block_length
        total_length = num_blocks * self.block_length

        # Native block-diagonal causal mask, broadcast to the batch: same for every
        # row, so a (B,1,L,L) expand view is enough (transformers' SDPA helper
        # passes dim-4 masks through unchanged).
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        attn_full = (
            block_mask.repeat_interleave(self.block_length, dim=0)
            .repeat_interleave(self.block_length, dim=1)
            .unsqueeze(0).unsqueeze(0)
        ).log().to(torch.bfloat16)
        attn_full = attn_full.expand(batch_size, -1, -1, -1)

        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0).expand(batch_size, -1)

        x = torch.full((batch_size, total_length), self.mask_token_id,
                       dtype=torch.long, device=self.device)
        x[:, :prompt_length] = prompt_ids

        prefill_blocks = prompt_length // self.block_length
        schedule = _get_num_transfer_tokens(self.block_length, denoising_steps)

        gen_blocks = max(num_blocks - prefill_blocks, 1)
        total_global_steps = gen_blocks * denoising_steps
        original_alpha = self.strategy.alpha

        try:
            for num_block in range(prefill_blocks, num_blocks):
                window_end = (num_block + 1) * self.block_length
                cur_x = x[:, :window_end]  # view: writes go through to x
                cur_attn = attn_full[:, :, :window_end, :window_end]
                cur_pos = position_ids[:, :window_end]

                for step in range(denoising_steps):
                    active_block_mask = cur_x[:, -self.block_length:] == self.mask_token_id
                    if active_block_mask.sum() == 0:
                        break

                    with torch.no_grad():
                        logits = self.model(
                            cur_x, attention_mask=cur_attn, position_ids=cur_pos,
                            use_cache=False,
                        ).logits  # (B, window_end, V) fp32

                    # --- ODD / diversity guidance on the generation region ---
                    global_step = (num_block - prefill_blocks) * denoising_steps + step
                    curr_alpha = original_alpha * (1.0 - global_step / total_global_steps)

                    if curr_alpha > 0.0:
                        gen_mask = cur_x[:, prompt_length:] == self.mask_token_id
                        gen_logits = logits[:, prompt_length:, :].to(self.strategy_dtype)
                        # The active (last) block can overlap the prompt tail in the
                        # first gen block, so the gen region may be shorter than
                        # block_length. Keep the fp32 active-block slice and later
                        # overwrite only the part covered by the gen region.
                        active_logits = logits[:, -self.block_length:, :].clone()
                        del logits  # free the full fp32 logits during the strategy's autograd pass
                        self.strategy.alpha = curr_alpha
                        try:
                            with torch.enable_grad():
                                guided, _ = self.strategy.apply(
                                    gen_logits,
                                    mask_index=gen_mask,
                                    x=cur_x[:, prompt_length:],
                                    history_vecs=[], history_qualities=[],
                                    protected_tokens=self.protected_tokens,
                                )
                        finally:
                            self.strategy.alpha = original_alpha
                        overlap = min(self.block_length, window_end - prompt_length)
                        active_logits[:, -overlap:, :] = guided[:, -overlap:, :].float()
                        del gen_logits, guided
                    else:
                        active_logits = logits[:, -self.block_length:, :]
                        del logits

                    # --- native sampling + threshold/top-confidence acceptance ---
                    with torch.no_grad():
                        if temperature == 0 and not exact_native_sampling:
                            x0, x0_p = _sample_greedy(active_logits)
                        else:
                            x0, x0_p = _sample_with_temperature_topk_topp(
                                active_logits, temperature=temperature,
                                top_k=top_k, top_p=top_p,
                            )
                        del active_logits

                        num_to_transfer = int(schedule[step].item())
                        confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool)

                        for b in range(batch_size):
                            n_mask_b = int(active_block_mask[b].sum().item())
                            if n_mask_b == 0:
                                continue  # this row's block is already fully decoded
                            high_conf_mask = confidence[b] > self.threshold
                            if int(high_conf_mask.sum().item()) >= num_to_transfer:
                                transfer_index[b] = high_conf_mask
                            else:
                                _, idx = torch.topk(
                                    confidence[b], k=min(num_to_transfer, n_mask_b)
                                )
                                transfer_index[b, idx] = True

                        if transfer_index.any():
                            cur_x[:, -self.block_length:][transfer_index] = x0[transfer_index]

                if self.early_stop and self.eos_token_id is not None:
                    has_eos = (x[:, prompt_length:window_end] == self.eos_token_id).any(dim=1)
                    if bool(has_eos.all()):
                        break
        finally:
            self.strategy.alpha = original_alpha

        # --- trim at first eos per row (native behaviour) and decode ---
        gen_tokens = x[:, prompt_length: prompt_length + gen_length]
        sequences, texts = [], []
        for b in range(batch_size):
            row = gen_tokens[b]
            eos_positions = (row == self.eos_token_id).nonzero(as_tuple=True)[0]
            end = int(eos_positions[0].item()) + 1 if len(eos_positions) > 0 else gen_length
            trimmed = row[:end]
            sequences.append(trimmed.detach().cpu())
            texts.append(self.tokenizer.decode(trimmed.tolist(), skip_special_tokens=True))

        self.last_sequences = sequences
        return [], texts
