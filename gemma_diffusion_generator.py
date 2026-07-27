"""DiffusionGemma (google/diffusiongemma-26B-A4B-it) adapter for ODD.

Injects the ODD strategy into DiffusionGemma's denoising loop through the
public ``logits_processor`` argument of ``DiffusionGemmaGenerationMixin
.generate()`` — custom processors are applied first, on the diffusion logits
of shape (batch, canvas_length, vocab). This mirrors the Dream integration
(bench_common.DreamGenerator's generation_logits_hook_func) rather than the
hand-rolled LLaDA2 sampler port: no sampler surgery is required.

Semantics mapping vs the masked-diffusion backends:
  - DiffusionGemma canvases start as RANDOM tokens (EntropyBoundSampler), not
    mask tokens, and the processor cannot see per-token acceptance state, so
    the whole canvas plays the role of the "still masked" region:
    ``mask_index`` is all-True over the canvas. FeatureExtractor's confidence
    weighting therefore averages max-softmax confidence over the full canvas —
    the natural analog of "confidence over undecided positions".
  - alpha anneal alpha*(1 - step/steps) counts denoising calls per generate();
    with max_new_tokens <= canvas_length (256) generation is single-canvas, so
    the counter is exact. Multi-canvas runs reuse the schedule modulo
    max_denoising_steps (early stopping makes this approximate; fine for
    probes, revisit before quoting multi-canvas numbers).
  - temperature: our sweeps' single theta maps to a constant schedule
    (t_min == t_max == theta). Pass temperature=None to keep the checkpoint's
    native linear schedule.

Environment: needs transformers >= 5.14 (v5 API with DiffusionGemma classes)
in a FRESH env — the dream env pins 4.46 and must not be upgraded. The bf16
checkpoint download is ~52GB; 4-bit nf4 fits a 24GB card (26B MoE, A4B).
"""

import torch
from typing import List, Optional, Tuple

MODEL_NAME = "google/diffusiongemma-26B-A4B-it"


def load_diffusion_gemma(model_name: str = MODEL_NAME, load_in_4bit: bool = True):
    """Load DiffusionGemma with the same quantization conventions as the other
    backends (4-bit nf4, bf16 compute). Returns (model, processor); use
    processor.tokenizer for decoding."""
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        DiffusionGemmaForBlockDiffusion,
    )

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(model_name)
    # device_map={"": 0}: put everything on the (single visible) GPU. "auto"
    # over-estimates the quantized MoE footprint and preemptively dispatches
    # modules to CPU, which the bnb quantizer then refuses.
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    return model, processor


class ODDCanvasLogitsProcessor:
    """LogitsProcessor applying ODD across the batch on each denoising step.

    Duck-typed rather than subclassing transformers.LogitsProcessor so this
    module imports without transformers (mirrors bench_common's lazy-import
    convention); LogitsProcessorList only requires __call__.
    """

    def __init__(self, strategy, base_alpha: float, max_denoising_steps: int):
        self.strategy = strategy
        self.base_alpha = base_alpha
        self.max_denoising_steps = max(int(max_denoising_steps), 1)
        self._step = 0
        self._warned_shape = False

    def reset(self):
        self._step = 0

    def __call__(self, input_ids, scores):
        # Diffusion logits arrive as (B, canvas, V); anything else means the
        # processor was invoked outside the denoising loop -- pass through.
        if scores.dim() != 3:
            if not self._warned_shape:
                print(f"[ODDCanvasLogitsProcessor] unexpected scores shape "
                      f"{tuple(scores.shape)}; passing through")
                self._warned_shape = True
            return scores

        step_in_canvas = self._step % self.max_denoising_steps
        self._step += 1
        step_alpha = self.base_alpha * (1.0 - step_in_canvas / self.max_denoising_steps)
        if step_alpha <= 0.0:
            return scores

        with torch.enable_grad():
            canvas_logits = scores.float().clone()
            # Current canvas ids if the loop provides them; otherwise fall back
            # to the denoiser's argmax (only used for masking bookkeeping).
            if input_ids is not None and input_ids.dim() == 2 \
                    and input_ids.shape[1] == scores.shape[1]:
                x = input_ids
            else:
                x = scores.argmax(dim=-1)
            mask_index = torch.ones_like(x, dtype=torch.bool)

            self.strategy.alpha = step_alpha
            guided, _ = self.strategy.apply(
                logits=canvas_logits, mask_index=mask_index, x=x,
                history_vecs=[], history_qualities=[],
                protected_tokens=None,
            )
        return guided.detach().to(scores.dtype)


class GemmaDiffusionDiverseGenerator:
    """DiverseGenerator-compatible generator:
        generate(prompt, batch_size, steps, gen_length, temperature) -> ([], texts)
    strategy=None means baseline (no logits processor passed at all).
    """

    def __init__(self, model, processor, strategy):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.strategy = strategy
        self.base_alpha = strategy.alpha if strategy is not None else 0.0

    def generate(self, prompt: str, batch_size: int, steps: int, gen_length: int,
                 temperature: Optional[float]) -> Tuple[List, List[str]]:
        from transformers import LogitsProcessorList

        chat = [{"role": "user", "content": prompt}]
        input_ids = self.processor.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True,
        ).to(self.model.device)
        input_ids = input_ids.expand(batch_size, -1).contiguous()
        prompt_len = input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": gen_length,
            "max_denoising_steps": steps,
        }
        # Constant-theta schedule for sweep comparability; None keeps the
        # checkpoint's native linear t_max -> t_min schedule.
        if temperature is not None:
            t = max(float(temperature), 1e-4)
            gen_kwargs["t_min"] = t
            gen_kwargs["t_max"] = t

        if self.strategy is not None:
            odd_proc = ODDCanvasLogitsProcessor(self.strategy, self.base_alpha, steps)
            gen_kwargs["logits_processor"] = LogitsProcessorList([odd_proc])

        with torch.no_grad():
            output = self.model.generate(input_ids, **gen_kwargs)

        sequences = output.sequences if hasattr(output, "sequences") else output
        samples = [
            self.tokenizer.decode(seq[prompt_len:].tolist(), skip_special_tokens=True)
            for seq in sequences
        ]
        return [], samples
