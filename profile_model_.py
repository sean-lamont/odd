import gc
import numpy as np
import os
import sys
import time
import torch
from types import SimpleNamespace
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Ensure dpp_core can be imported
try:
    from dpp_core import FeatureExtractor, get_strategy, DPPGenerator
except ImportError:
    print("Error: Could not import 'dpp_core'. Make sure it is in the same directory.")
    sys.exit(1)


def load_model(cfg):
    print(f"Loading {cfg.model.name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        cfg.model.name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.eval()
    tokenizer.padding_side = 'left'

    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        embedding_matrix = model.model.transformer.wte.weight
    else:
        embedding_matrix = None

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = cfg.model.mask_token_id

    return model, tokenizer, embedding_matrix, mask_token_id


# -----------------------------------------------------------------------------
# 2. PROFILING CONFIGURATION
# -----------------------------------------------------------------------------

BASE_CONFIG = SimpleNamespace(
    model=SimpleNamespace(
        name="GSAI-ML/LLaDA-8B-Instruct",
        load_in_4bit=True,
        mask_token_id=126336
    ),
    strategy=SimpleNamespace(
        target="logits",
        pool="max",
        top_k=0
    )
)

batch_sizes = [1, 2, 4, 8, 16, 32, 64]#, 128, 256]
steps = [16, 32, 64, 128]
length = [16, 32, 64, 128]#, 256, 512]

batch_sizes.reverse()

# Generate scenarios
SCENARIOS = [(f'B:{b} S:{s} L:{l}', b, s, l) for b in batch_sizes for s in steps for l in length if s <= l]

PROMPT = "The quick brown fox jumps over the lazy dog"


# -----------------------------------------------------------------------------
# 3. PROFILING ENGINE
# -----------------------------------------------------------------------------

def run_single_pass(generator, batch, steps, length, prompt):
    """Runs a generation pass and returns wall-clock time (s) and peak VRAM (MB)."""
    gc.collect()

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated()
    else:
        initial_mem = 0

    start_t = time.perf_counter()

    # Generate
    _ = generator.generate(
        prompt=prompt,
        batch_size=batch,
        steps=steps,
        gen_length=length,
        temperature=1.0
    )

    if is_cuda:
        torch.cuda.synchronize()
        max_mem = torch.cuda.max_memory_allocated()
        mem_used_mb = (max_mem - initial_mem) / (1024 ** 2)
    else:
        mem_used_mb = 0.0

    end_t = time.perf_counter()

    return end_t - start_t, mem_used_mb


def main():
    print(f"--- Starting Profiler ---")
    print(f"Model: {BASE_CONFIG.model.name}")
    print(f"4-Bit Quantization: {BASE_CONFIG.model.load_in_4bit}")

    # 1. Load Model
    model, tokenizer, embedding_matrix, mask_token_id = load_model(BASE_CONFIG)

    # 2. Setup Feature Extractor
    feature_extractor = FeatureExtractor(
        embedding_matrix=embedding_matrix,
        kernel_target=BASE_CONFIG.strategy.target,
        pooling_method=BASE_CONFIG.strategy.pool,
        top_k=BASE_CONFIG.strategy.top_k
    )

    print("\n" + "=" * 115)
    print(
        f"{'Scenario':<18} | {'B':<3} | {'Stp':<3} | {'Len':<4} | {'Base(s)':<7} | {'Strat(s)':<8} | {'T_OH':<7} | {'Base(MB)':<8} | {'Strat(MB)':<9} | {'Mem_OH':<8}")
    print("=" * 115)

    for label, batch, steps, length in SCENARIOS:
        try:
            # --- A. RUN BASELINE ---
            strat_base = get_strategy("baseline", 0.0, 0.0, feature_extractor)
            gen_base = DPPGenerator(model, tokenizer, strat_base, mask_token_id)

            # Warmup run (for CUDA graph compilation / allocations)
            if SCENARIOS.index((label, batch, steps, length)) == 0:
                run_single_pass(gen_base, 1, 2, 5, "warmup")

            time_base, mem_base = run_single_pass(gen_base, batch, steps, length, PROMPT)

            # --- B. RUN STRATEGY ---
            strat_ortho = get_strategy("batched_orth", 64, 1.0, feature_extractor)
            gen_ortho = DPPGenerator(model, tokenizer, strat_ortho, mask_token_id)

            time_strat, mem_strat = run_single_pass(gen_ortho, batch, steps, length, PROMPT)

            # --- C. CALCULATE OVERHEAD ---
            time_overhead_s = time_strat - time_base
            time_overhead_pct = (time_overhead_s / time_base) * 100.0 if time_base > 0 else 0.0

            mem_overhead_mb = mem_strat - mem_base

            # Color coding for Time Overhead
            if time_overhead_pct < 5.0:
                color = "\033[92m"  # Green
            elif time_overhead_pct < 15.0:
                color = "\033[93m"  # Yellow
            else:
                color = "\033[91m"  # Red
            reset = "\033[0m"

            print(
                f"{label:<18} | {batch:<3} | {steps:<3} | {length:<4} | "
                f"{time_base:<7.2f} | {time_strat:<8.2f} | {color}+{time_overhead_pct:.1f}%{reset} | "
                f"{mem_base:<8.0f} | {mem_strat:<9.0f} | {mem_overhead_mb:<8.0f}"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{label:<18} | {batch:<3} | OOM ERROR (Reduce Batch/Length)")
                torch.cuda.empty_cache()
            else:
                print(f"{label:<18} | ERROR: {e}")


if __name__ == "__main__":
    main()