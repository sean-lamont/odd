import torch
import torch.nn.functional as F
import time
import sys
import os

# Ensure dpp_core can be imported
sys.path.append(os.getcwd())

try:
    from dpp_core import FeatureExtractor, OrthogonalProjectionStrategy, BatchedOrthogonalProjectionStrategy
except ImportError:
    print("Error: Could not import 'dpp_core.py'. Make sure it is in the same directory.")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Test Harness
# -----------------------------------------------------------------------------

def run_benchmark(batch_sizes, seq_lengths, vocab_size=126464, generation_steps=[10, 50, 100]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Benchmark on: {device.upper()}")
    print("-" * 90)
    print(
        f"{'Batch':<6} | {'SeqLen':<6} | {'Step Time (ms)':<15} | {'VRAM (MB)':<10} | {'Est. Overhead (50 steps)':<25}")
    print("-" * 90)

    results = []

    # Initialize Strategy ONCE (stateless, so reusable)
    feature_extractor = FeatureExtractor(pooling_method='max', use_confidence_weighting=True)
    strategy = BatchedOrthogonalProjectionStrategy(alpha=64, quality_scale=1.0, feature_extractor=feature_extractor)

    for B in batch_sizes:
        for S in seq_lengths:

            # 1. Setup Data
            try:
                # Logits can be huge (B * S * V * 4 bytes). Check if fits.
                logits = torch.randn(B, S, vocab_size, device=device, requires_grad=True)
                mask = torch.ones(B, S, dtype=torch.bool, device=device)
                x = torch.randint(0, vocab_size, (B, S), device=device)
                # Protected tokens (EOS, PAD)
                protected = torch.tensor([0, 1, 2], device=device)

                history_vecs = []
                history_quals = []

                # 2. Warmup (Run once to compile kernels/allocate)
                strategy.apply(logits, mask, x, history_vecs, history_quals, protected)
                torch.cuda.synchronize()

                # 3. Timing Loop
                # We measure the AVERAGE time over N runs to smooth out noise
                N_runs = 20
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # Reset memory max tracker to capture only THIS run's overhead
                torch.cuda.reset_peak_memory_stats()
                # We want to measure the DELTA (peak during op - baseline)
                initial_memory = torch.cuda.memory_allocated()

                start_event.record()
                for _ in range(N_runs):
                    # We must recreate detached logits each time so graphs don't accumulate
                    l_run = logits.detach().requires_grad_(True)
                    strategy.apply(l_run, mask, x, history_vecs, history_quals, protected)
                end_event.record()

                torch.cuda.synchronize()

                # 4. Metrics
                total_time_ms = start_event.elapsed_time(end_event)
                avg_step_time_ms = total_time_ms / N_runs

                max_memory = torch.cuda.max_memory_allocated()
                # Overhead is roughly Peak - Initial
                memory_overhead_mb = (max_memory - initial_memory) / 1024 / 1024

                # Extrapolations (Total added time for a full generation)
                overhead_50 = (avg_step_time_ms * 50) / 1000.0  # Seconds

                print(
                    f"{B:<6} | {S:<6} | {avg_step_time_ms:<15.4f} | {memory_overhead_mb:<10.2f} | {overhead_50:<25.4f}s")

                results.append({
                    "batch": B, "seq": S,
                    "step_ms": avg_step_time_ms,
                    "mem_mb": memory_overhead_mb
                })

                # Cleanup to prevent OOM
                del logits, mask, x, history_vecs, l_run
                torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"{B:<6} | {S:<6} | {'OOM / ERROR':<15} | {'-':<10} | {str(e)[:30]}...")
                print (e)
                torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    # Configure your sweep here
    # Warning: Large Batch * SeqLen * VocabSize (32k) consumes massive VRAM.
    # Adjust mostly Batch Size.
    BATCH_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]
    SEQ_LENGTHS = [32, 64, 128, 512, 1024]

    # Run
    stats = run_benchmark(BATCH_SIZES, SEQ_LENGTHS)

    # Extrapolation Summary Table
    print("\n" + "=" * 80)
    print(" EXTRAPOLATION: Total Added Latency (Seconds) for Full Generation")
    print("=" * 80)
    print(f"{'Config (B, S)':<20} | {'10 Steps':<10} | {'50 Steps':<10} | {'100 Steps':<10}")
    print("-" * 80)

    for r in stats:
        name = f"B={r['batch']}, S={r['seq']}"
        t_10 = (r['step_ms'] * 10) / 1000
        t_50 = (r['step_ms'] * 50) / 1000
        t_100 = (r['step_ms'] * 100) / 1000
        print(f"{name:<20} | {t_10:<10.4f}s | {t_50:<10.4f}s | {t_100:<10.4f}s")