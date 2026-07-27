"""Batched full-sequence masked-diffusion generator for RND1 (radicalnumerics/RND1-Base-0910)
with ODD support.

This ports RND1's native ``diffusion_sample()`` (github.com/RadicalNumerics/RND1,
rnd/sampling.py) to a batched, DiverseGenerator-compatible class and lets an ODD/DPP
strategy (strategies.py) modify the generation-region logits before tokens are
committed, exactly as generator.DiverseGenerator does for LLaDA-8B.

RND1 = Qwen3-30B-A3B (sparse MoE, 3B active) converted to a full-sequence masked
diffusion model with bidirectional attention (config is_causal=false). It is a BASE
model: prompts are raw strings, NO chat template (encode_prompt below is plain
tokenization; contrast DiverseGenerator.apply_chat_template).

Faithfulness to the native diffusion_sample():
  * Same canvas: x = [prompt_ids | mask_id * gen_length]; optionally EOS forced at
    the last canvas position (add_eos_at_end, demo default True). The prompt segment
    is FROZEN: `maskable` covers only canvas positions, every commit writes through
    x[:, prompt_len:], and denoising never touches prompt tokens (the model attends
    bidirectionally over prompt+canvas, but only canvas positions are ever masked
    or re-written).
  * Same AR-retained shift: RND1's logits at position i predict the token at
    position i+1 (sampling.py: "Shift predictions: pos i predicts token i+1").
    The distribution for canvas position j therefore comes from hidden state j-1;
    we slice hidden[:, prompt_len-1:-1] so the gen-region logits tensor is
    position-aligned with x[:, prompt_len:] (index k <-> canvas token k), which is
    the alignment strategies.FeatureExtractor expects.
  * Same unmasking rule (the default entropy path): with step counting DOWN from
    steps-1 to 1, rate = step/steps, keep the (initial_masked * rate) HIGHEST-entropy
    still-masked positions masked and commit the rest; any survivors are filled with
    the final predictions after the loop. Entropy/predictions come from the
    temperature-scaled (clamped at 1e-8), optionally top-k/top-p filtered
    distribution; greedy argmax iff temperature == 0 (demo convention), otherwise a
    categorical sample. The eb_gamma sampler variant is not ported (harness never
    uses it).
  * Same special-token handling: BOS/EOS/PAD positions are excluded from
    `maskable`; decoding trims each row at its first EOS.

Documented deviations from the native code:
  * Batched: the native loop is written for batch size 1 (with batch-ready
    internals); we run the whole 16-sample batch in one forward.
  * lm_head is applied only to the gen-region hidden states instead of the full
    sequence, so the (B, prompt+gen, 151936) logits tensor is never materialised
    (identical values, large memory saving with 8-shot prompts).
  * Entropy uses where(p > 0, p*logp, 0) instead of p*logp: identical when no
    top-k/top-p filter is active (the default), and avoids the 0 * -inf = NaN the
    native code produces when filters ARE active.
  * ODD guidance runs in bf16 on the gen-region logits (llada2_generator
    convention) and the sampler sees the guided logits recast to fp32; the baseline
    path (alpha == 0) samples the untouched logits.
  * Base-model few-shot continuation guard: after decoding, each sample is
    truncated at the first occurrence of any `stop_strings` entry (e.g.
    "Question:"), so the harness answer extractors (last-number semantics) never
    see a self-continued next exemplar. This only affects decoded text, never the
    canvas or the prompt.

ODD integration (mirrors DiverseGenerator / LLaDA2DiverseGenerator):
  * strategy.apply(logits, mask_index, x, history_vecs, history_qualities,
    protected_tokens) is called once per forward pass on the aligned gen-region
    logits, before predictions/entropies are derived, so guidance shifts both WHAT
    is predicted and WHICH positions look confident.
  * alpha is annealed over the single full-sequence window:
    alpha_i = alpha * (1 - i/steps) with i the 0-based forward-pass index.
  * protected_tokens = {eos, pad, mask}: the guidance gradient is zeroed on those
    vocab entries.
"""

import torch
from typing import Dict, List, Optional, Tuple

# From radicalnumerics/RND1-Base-0910 config.json.
RND1_MASK_ID = 151669
RND1_EOS_ID = 151645
RND1_PAD_ID = 151643


def load_rnd1(model_name: str = "radicalnumerics/RND1-Base-0910", load_in_4bit: bool = True):
    """Load RND1 with the same conventions as the other backends (4-bit nf4,
    bf16 compute) through AutoModelForMaskedLM (the HF auto_map routes it to the
    remote-code RND1LM class, which owns lm_head + the diffusion GenerationMixin).

    device_map={"": 0}: single-GPU placement, as with DiffusionGemma ("auto"
    over-estimates the quantized MoE footprint and preemptively dispatches
    modules to CPU, which the bnb quantizer then refuses).

    IMPORTANT: the post-load memory print below is a quantization tripwire.
    RND1 stores its 128 experts as fused 3D tensors (Qwen3MoeExperts), which are
    NOT nn.Linear modules, so bitsandbytes may silently leave ~29B of the 30.5B
    parameters in bf16 (the DiffusionGemma failure mode). nf4 should read
    ~16-20 GiB; ~57-60 GiB means the quantization was skipped for the experts.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # demo_rnd_generation.py setting
        device_map={"": 0},
    )
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc_gib = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"[rnd1] post-load torch.cuda.memory_allocated(0) = {alloc_gib:.1f} GiB")
        if load_in_4bit and alloc_gib > 30.0:
            print(
                "[rnd1] WARNING: 4-bit nf4 was requested but the allocation looks like "
                "bf16 (~57 GiB). bitsandbytes only rewrites nn.Linear modules; RND1's "
                "fused Qwen3MoeExperts 3D expert tensors (~29B params) are left "
                "unquantized. Expect OOM on <80 GB cards."
            )
    return model, tokenizer


class RND1DiverseGenerator:
    """DiverseGenerator-compatible batched generator for RND1 full-sequence
    masked diffusion.

    Public interface mirrors generator.DiverseGenerator:
        generate(prompt, batch_size, steps, gen_length, temperature) -> ([], texts)
    """

    def __init__(self, model, tokenizer, strategy,
                 mask_token_id: int = RND1_MASK_ID,
                 eos_token_id: int = RND1_EOS_ID,
                 pad_token_id: int = RND1_PAD_ID,
                 add_eos_at_end: bool = True,
                 stop_strings: Optional[List[str]] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 strategy_dtype: torch.dtype = torch.bfloat16):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        self.add_eos_at_end = add_eos_at_end
        self.stop_strings = list(stop_strings) if stop_strings else []
        self.top_k = top_k
        self.top_p = top_p
        self.strategy_dtype = strategy_dtype
        self.device = model.device

        self.protected_tokens = torch.tensor(
            sorted({eos_token_id, self.pad_token_id, mask_token_id}), device=self.device
        )
        self.last_sequences = None  # trimmed generated token ids per row (debug/parity)

    def encode_prompt(self, prompt: str, batch_size: int) -> torch.Tensor:
        """RND1-Base is a BASE model: plain tokenization of the raw prompt string,
        no chat template (matches demo_rnd_generation.py / the model card). All
        rows are identical, so no padding / attention mask is needed."""
        encoded = self.tokenizer([prompt] * batch_size, return_tensors="pt")
        return encoded.input_ids.to(self.device)

    # ------------------------------------------------------------------
    # One forward pass -> (predictions, entropies) for the gen region,
    # with ODD guidance injected before the native sampling math.
    # ------------------------------------------------------------------
    def _forward_scores(self, x: torch.Tensor, prompt_len: int, temperature: float,
                        greedy: bool, step_alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if hasattr(self.model, "model") and hasattr(self.model, "lm_head"):
                # Trunk forward, then lm_head only on the hidden states we need:
                # RND1 keeps the AR shift (logits at i predict token i+1), so the
                # distribution for canvas position j lives at hidden index j-1.
                hidden = self.model.model(input_ids=x).last_hidden_state
                gen_logits = self.model.lm_head(hidden[:, prompt_len - 1:-1, :])
            else:  # fallback: full logits, then the same aligned slice
                logits = self.model(input_ids=x).logits
                gen_logits = logits[:, prompt_len - 1:-1, :]

        # --- ODD / diversity guidance on the aligned gen-region logits ---
        if step_alpha > 0.0 and self.strategy is not None:
            gen_mask = x[:, prompt_len:] == self.mask_token_id
            guided_in = gen_logits.to(self.strategy_dtype)
            del gen_logits
            original_alpha = self.strategy.alpha
            self.strategy.alpha = step_alpha
            try:
                with torch.enable_grad():
                    guided, _ = self.strategy.apply(
                        guided_in,
                        mask_index=gen_mask,
                        x=x[:, prompt_len:],
                        history_vecs=[], history_qualities=[],
                        protected_tokens=self.protected_tokens,
                    )
            finally:
                self.strategy.alpha = original_alpha
            gen_logits = guided

        # --- native sampling math (sampling.py forward_scores), fp32 ---
        with torch.no_grad():
            safe_temperature = max(temperature, 1e-8)
            logits = gen_logits.float() / safe_temperature
            del gen_logits
            if self.top_k is not None and self.top_k > 0:
                logits = _apply_top_k_filtering(logits, self.top_k)
            if self.top_p is not None and 0 < self.top_p < 1.0:
                logits = _apply_top_p_filtering(logits, self.top_p)

            logp = torch.log_softmax(logits, dim=-1)
            if greedy:
                pred = logp.argmax(-1)
            else:
                pred = torch.distributions.Categorical(logits=logp).sample()

            p = logp.exp()
            # where() instead of the native p*logp: avoids 0 * -inf = NaN when
            # top-k/top-p filters are active; identical otherwise.
            ent = -torch.where(p > 0, p * logp, torch.zeros_like(p)).sum(-1)
        return pred, ent

    def generate(self, prompt: str, batch_size: int, steps: int, gen_length: int,
                 temperature: float) -> Tuple[List[Dict], List[str]]:
        prompt_ids = self.encode_prompt(prompt, batch_size)
        prompt_len = prompt_ids.shape[1]
        seq_len = prompt_len + gen_length
        steps = max(int(steps), 1)
        greedy = temperature == 0.0  # demo_rnd_generation.py convention
        base_alpha = self.strategy.alpha if self.strategy is not None else 0.0

        # --- native canvas: frozen prompt + fully masked generation region ---
        x = torch.full((batch_size, seq_len), self.mask_token_id,
                       dtype=torch.long, device=self.device)
        x[:, :prompt_len] = prompt_ids
        if self.add_eos_at_end and self.eos_token_id is not None:
            x[:, -1] = self.eos_token_id

        # maskable == positions the denoiser may write: canvas mask tokens only.
        # The prompt segment is never maskable and never rewritten.
        maskable = torch.zeros_like(x, dtype=torch.bool)
        maskable[:, prompt_len:] = x[:, prompt_len:] == self.mask_token_id
        total_masked0 = maskable.sum(dim=1, keepdim=True)  # (B, 1), per native

        finf = torch.finfo(torch.float32)
        forward_idx = 0
        pred, ent = self._forward_scores(
            x, prompt_len, temperature, greedy,
            step_alpha=base_alpha * (1.0 - forward_idx / steps),
        )

        # --- native entropy-decay unmasking loop (sampling.py, eb_gamma=None) ---
        for step in range(steps - 1, 0, -1):
            gen_maskable = maskable[:, prompt_len:]  # view
            rate = step / steps
            cutoff_len = (total_masked0 * rate).long().clamp(min=0)  # (B, 1)

            # Keep the cutoff_len HIGHEST-entropy masked positions masked;
            # everything else still masked gets committed this step.
            sel_scores = ent.masked_fill(~gen_maskable, -finf.max)
            keep_mask = torch.zeros_like(sel_scores, dtype=torch.bool)
            k_max = int(cutoff_len.max().item())
            if k_max > 0:
                _, idx = torch.topk(sel_scores, k_max, dim=-1, largest=True)
                for b in range(batch_size):
                    k_b = int(cutoff_len[b].item())
                    if k_b > 0:
                        keep_mask[b, idx[b, :k_b]] = True

            to_unmask = gen_maskable & ~keep_mask
            if to_unmask.any():
                x[:, prompt_len:][to_unmask] = pred[to_unmask]
                maskable[:, prompt_len:][to_unmask] = False

            if maskable.any():
                forward_idx += 1
                pred, ent = self._forward_scores(
                    x, prompt_len, temperature, greedy,
                    step_alpha=base_alpha * (1.0 - forward_idx / steps),
                )

        # --- native finalization: fill any survivors with the last predictions ---
        gen_maskable = maskable[:, prompt_len:]
        if gen_maskable.any():
            x[:, prompt_len:][gen_maskable] = pred[gen_maskable]

        # --- decode canvas ONLY (prompt never decoded), trim at first EOS,
        #     then cut at the first stop string (few-shot continuation guard) ---
        gen_tokens = x[:, prompt_len:]
        sequences, texts = [], []
        for b in range(batch_size):
            row = gen_tokens[b]
            eos_positions = (row == self.eos_token_id).nonzero(as_tuple=True)[0]
            end = int(eos_positions[0].item()) if len(eos_positions) > 0 else gen_length
            trimmed = row[:end]
            sequences.append(trimmed.detach().cpu())
            text = self.tokenizer.decode(trimmed.tolist(), skip_special_tokens=True)
            for stop in self.stop_strings:
                cut = text.find(stop)
                if cut != -1:
                    text = text[:cut]
            texts.append(text)

        self.last_sequences = sequences
        return [], texts


# ---- verbatim-equivalent ports of the native filtering helpers (sampling.py) ----

def _apply_top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    top_k_values, top_k_indices = torch.topk(logits, min(k, logits.size(-1)), dim=-1)
    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(-1, top_k_indices, top_k_values)
    return filtered_logits


def _apply_top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 0] = False  # keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))
