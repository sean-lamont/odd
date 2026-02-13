import gc
import hydra
import numpy as np
import os
import sys
import time
import torch
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from types import SimpleNamespace

from dpp_core import FeatureExtractor, get_strategy, DPPGenerator


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

# Define the base Model Config (Mocking Hydra's DictConfig)
# CHANGE 'name' to the actual model you want to test (e.g., 'mistralai/Mistral-7B-v0.1')
BASE_CONFIG = SimpleNamespace(
    model=SimpleNamespace(
        name="GSAI-ML/LLaDA-8B-Instruct",  # Replace with your 4-bit compatible model
        load_in_4bit=True,
        mask_token_id=126336  # Default for GPT2, change for others
    ),
    strategy=SimpleNamespace(
        target="logits",
        pool="max",
        top_k=0
    )
)

# Define the scenarios to profile
# (Label, Batch Size, Steps, Gen Length)
SCENARIOS = [
    ("Baseline (Batch 1)", 1, 10, 32),
    ("Small Batch (4)", 4, 10, 32),
    ("Medium Batch (8)", 8, 10, 32),
    (" Batch (16)", 16, 10, 32),
    (" Batch (32)", 32, 10, 32),
    (" Batch (64)", 64, 10, 32),
    ("Long Gen (64)", 32, 10, 64),
    ("Long Gen (128)", 32, 10, 128),
    ("High Steps (50)", 4, 50, 32),
]

PROMPT = "The quick brown fox jumps over the lazy dog"


# -----------------------------------------------------------------------------
# 3. PROFILING ENGINE
# -----------------------------------------------------------------------------

def run_single_pass(generator, batch, steps, length, prompt):
    """Runs a generation pass and returns wall-clock time in seconds."""
    # Garbage collection to ensure fair memory usage
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_t = time.perf_counter()

    # We ignore the output samples/history, we just want execution time
    _ = generator.generate(
        prompt=prompt,
        batch_size=batch,
        steps=steps,
        gen_length=length,
        temperature=1.0
    )

    torch.cuda.synchronize()
    return time.perf_counter() - start_t


def main():
    print(f"--- Starting Profiler ---")
    print(f"Model: {BASE_CONFIG.model.name}")
    print(f"4-Bit Quantization: {BASE_CONFIG.model.load_in_4bit}")

    # 1. Load Model (Once)
    model, tokenizer, embedding_matrix, mask_token_id = load_model(BASE_CONFIG)

    # 2. Setup Feature Extractor
    feature_extractor = FeatureExtractor(
        embedding_matrix=embedding_matrix,
        kernel_target=BASE_CONFIG.strategy.target,
        pooling_method=BASE_CONFIG.strategy.pool,
        top_k=BASE_CONFIG.strategy.top_k
    )

    print("\n" + "=" * 85)
    print(
        f"{'Scenario':<20} | {'B':<3} | {'Stp':<3} | {'Len':<4} | {'Base(s)':<8} | {'Strat(s)':<8} | {'Overhead':<10}")
    print("=" * 85)

    for label, batch, steps, length in SCENARIOS:
        try:
            # --- A. RUN BASELINE (No Overhead) ---
            # Strategy: Baseline (alpha=0)
            strat_base = get_strategy("baseline", 0.0, 0.0, feature_extractor)
            gen_base = DPPGenerator(model, tokenizer, strat_base, mask_token_id)

            # Warmup run (short) for the first iteration only
            if batch == 1 and steps == 10:
                run_single_pass(gen_base, 1, 2, 5, "warmup")

            time_base = run_single_pass(gen_base, batch, steps, length, PROMPT)

            # --- B. RUN STRATEGY (Orthogonal Projection) ---
            # Strategy: Orthogonal Projection (alpha=0.5)
            strat_ortho = get_strategy("orthogonal_projection", 0.5, 1.0, feature_extractor)
            gen_ortho = DPPGenerator(model, tokenizer, strat_ortho, mask_token_id)

            time_strat = run_single_pass(gen_ortho, batch, steps, length, PROMPT)

            # --- C. CALCULATE OVERHEAD ---
            overhead_s = time_strat - time_base
            overhead_pct = (overhead_s / time_base) * 100.0

            # Color coding
            if overhead_pct < 5.0:
                color = "\033[92m"  # Green
            elif overhead_pct < 15.0:
                color = "\033[93m"  # Yellow
            else:
                color = "\033[91m"  # Red
            reset = "\033[0m"

            print(
                f"{label:<20} | {batch:<3} | {steps:<3} | {length:<4} | {time_base:<8.4f} | {time_strat:<8.4f} | {color}+{overhead_pct:.1f}%{reset}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{label:<20} | {batch:<3} | OOM ERROR (Reduce Batch Size)")
            else:
                print(f"{label:<20} | ERROR: {e}")


if __name__ == "__main__":
    main()