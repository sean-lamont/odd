"""Overgeneration + filtering baseline for the ODD rebuttal (§6).

Tests the claim that a *literal* generate-N-then-filter-to-16 pipeline built on
STANDARD (baseline) sampling cannot match ODD's Pass@16, because the deficit is
generation-side, not selection-side. For each HumanEval problem we

  1. draw N baseline samples (N in --pool-sizes, e.g. {17, 32}),
  2. filter down to 16 with embedding-space greedy farthest-point (k-centre)
     selection seeded with the first sample,

and report three per-problem Pass@16 style metrics, averaged over problems:

  (a) selected@16  -- any-correct among the 16 k-centre-SELECTED samples,
  (b) first16      -- any-correct among the FIRST 16 samples (this is exactly a
                      standard batch-16 baseline run: the control),
  (c) oracle@N     -- any-correct anywhere in the pool of N (upper bound).

N=17 is ~wall-clock-matched to ODD (one extra attempt, 3.4-6.7% overhead);
N=32 is 2x compute. Baseline strategy only -- no ODD guidance here.

VRAM PARITY: baseline samples are independent, so drawing N in chunks of <=16
(16+1 for N=17; 16+16 for N=32) is mathematically identical to a single draw
while keeping the peak batch -- hence VRAM -- identical to every other run.
Texts and correctness flags are concatenated across chunks in order, so the
first 16 samples are always the first chunk.

Eval semantics (prompt formatting, clean_code_for_harness, check_correctness,
timeout 3.0s) are copied verbatim from sweep_humaneval_plain.py. Paper config
defaults: gen_length 64, steps 32.

Examples:
    # CPU-only smoke test (stub generator+embeddings, offline wandb):
    python sweep_overgen.py --dry-run --n-problems 3 --pool-sizes 17

    # Deadline sanity read on GPU (1 temp, 3 problems):
    python sweep_overgen.py --n-problems 3 --pool-sizes 17 --temperatures 1.0

    # Full run, N=17 across all three temps:
    python sweep_overgen.py --pool-sizes 17 --temperatures 0.5 1.0 1.5
"""

import time

import numpy as np
import wandb
from bench_common import (
    build_arg_parser,
    build_generator,
    compose_cfg,
    init_wandb,
    load_shared_resources,
    RunWriter,
)
from human_eval.data import read_problems
from human_eval.execution import check_correctness


# Copied verbatim from sweep_humaneval_plain.py (kept local so we don't import
# from a script module that loads the model at import time).
def clean_code_for_harness(prompt, completion):
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]
    return completion


# --- stubs for --dry-run (no torch / no model) -----------------------------

STUB_SOLUTIONS = {
    "HumanEval/0": (
        "    for i in range(len(numbers)):\n"
        "        for j in range(i + 1, len(numbers)):\n"
        "            if abs(numbers[i] - numbers[j]) < threshold:\n"
        "                return True\n"
        "    return False\n"
    ),
    "HumanEval/2": "    return number % 1.0\n",
}


def stub_overgen_samples(problem, pool_size):
    """Canned pool of `pool_size` generations for --dry-run: a known-good body
    (where we have one) at indices 1 and 18, broken bodies elsewhere. Index 18
    sits outside the first 16, so a task with a canned solution should give
    first16 = 1 (correct is at idx 1) and oracle@N = 1; selected@16 depends on
    the stub embedding geometry."""
    correct = STUB_SOLUTIONS.get(problem["task_id"])
    samples = []
    for i in range(pool_size):
        if correct is not None and i in (1, 18):
            samples.append(correct + f"# sample {i}\n")
        elif i == 2:
            samples.append("    return undefined_name_xyz\n")
        else:
            samples.append(f"    return {i}  # wrong on purpose\n")
    return samples


def stub_embed(texts):
    """Deterministic pseudo-random unit vectors per text, so the k-centre path
    is exercised in --dry-run without torch / sentence-transformers."""
    vecs = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2 ** 32))
        v = rng.standard_normal(16)
        v /= (np.linalg.norm(v) + 1e-9)
        vecs.append(v)
    return np.stack(vecs).astype(np.float32)


# --- selection + embedding-space diversity ---------------------------------

def embed_texts(eval_model, texts):
    """L2-normalised embeddings as a float32 numpy array (rows = texts). With
    unit vectors, cosine similarity is a plain dot product -- matching
    utils.calculate_diversity_score's cos_sim semantics."""
    return eval_model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)


def kcentre_select(embeddings, k, seed_idx=0):
    """Greedy farthest-point (k-centre) selection of k indices, seeded with
    seed_idx. At each step add the point whose minimum cosine distance to the
    already-selected set is largest. Returns selected indices (seed first).
    If the pool has <= k points, all are returned."""
    n = embeddings.shape[0]
    if n <= k:
        return list(range(n))
    selected = [seed_idx]
    # cosine distance = 1 - cos_sim (unit vectors => 1 - dot)
    min_dist = 1.0 - embeddings @ embeddings[seed_idx]
    min_dist[seed_idx] = -np.inf
    while len(selected) < k:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        new_dist = 1.0 - embeddings @ embeddings[nxt]
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[nxt] = -np.inf
    return selected


def subset_diversity(embeddings, indices):
    """1 - mean off-diagonal cosine similarity over the given rows (same
    definition as utils.calculate_diversity_score, computed from the cached
    unit-vector embeddings)."""
    if len(indices) < 2:
        return 0.0
    sub = embeddings[indices]
    sims = sub @ sub.T
    n = len(indices)
    off = (float(sims.sum()) - float(np.trace(sims))) / (n * (n - 1))
    return 1.0 - off


def generate_pool(generator, prompt, pool_size, chunk_cap, cfg):
    """Draw `pool_size` independent baseline samples in chunks of <= chunk_cap
    (keeps peak batch, hence VRAM, identical to a batch-`chunk_cap` run).
    Returns (samples, gen_time). Samples are concatenated in draw order, so the
    first chunk_cap entries are the standard-baseline first-16 control."""
    samples = []
    remaining = pool_size
    start_t = time.time()
    while remaining > 0:
        chunk = min(chunk_cap, remaining)
        _, chunk_samples = generator.generate(
            prompt=prompt,
            batch_size=chunk,
            steps=cfg.steps,
            gen_length=cfg.gen_length,
            temperature=cfg.temperature,
        )
        samples.extend(chunk_samples)
        remaining -= chunk
    return samples, time.time() - start_t


def main():
    parser = build_arg_parser(
        description="Overgeneration + k-centre filtering baseline (HumanEval) for "
                    "the ODD rebuttal; same eval semantics as sweep_humaneval_plain.py",
        default_project="rebuttal-overgen",
        default_gen_length=64,  # paper config
    )
    parser.add_argument("--pool-sizes", nargs="+", type=int, default=[17, 32],
                        help="Overgeneration pool sizes N to sweep "
                             "(N=17 ~ wall-clock matched to ODD; N=32 = 2x compute)")
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="Sandbox timeout per sample in seconds (sweep_human_eval.py uses 3.0)")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size (chunk cap) must be >= 1")

    print(">>> Loading HumanEval (local human_eval/data/HumanEval.jsonl.gz)...")
    problems = list(read_problems().values())
    if args.n_problems > 0:
        problems = problems[: args.n_problems]
    print(f">>> Evaluating {len(problems)} problems | pool sizes {args.pool_sizes} | "
          f"temps {args.temperatures} | chunk cap {args.batch_size}")

    shared = None
    if not args.dry_run:
        base_cfg = compose_cfg(args, "baseline", 0.0, args.temperatures[0])
        shared = load_shared_resources(base_cfg)

    # Baseline strategy only -- no ODD. Outer loop over N so N=17 (all temps)
    # finishes before N=32, matching the launch chain / log split.
    for run_idx in range(args.n_runs):
        for pool_size in args.pool_sizes:
            for temperature in args.temperatures:
                cfg = compose_cfg(args, "baseline", 0.0, temperature)
                base_name = f"overgen_N{pool_size}_temp{temperature:g}"
                run_name = base_name if args.n_runs == 1 else f"{base_name}_run{run_idx}"
                oracle_key = f"oracle_pass_at_{pool_size}"

                run = init_wandb(args, cfg, run_name)
                results_table = wandb.Table(
                    columns=["task_id", "prompt", "completion", "cleaned_code",
                             "result", "passed", "selected", "in_first16"]
                )
                writer = RunWriter(
                    args.results_dir, "overgen", run_name,
                    csv_fieldnames=["task_id", "n_correct", "n_samples",
                                    "selected_pass_at_16", "first16_pass_at_16",
                                    oracle_key, "div_selected", "div_first16",
                                    "gen_time"],
                )
                generator = None if args.dry_run else build_generator(cfg, shared)

                selected_flags = []
                first16_flags = []
                oracle_flags = []
                div_selected_scores = []
                gen_times = []

                try:
                    print(f"\n>>> STARTING RUN {run_name}: strategy=baseline, N={pool_size}, "
                          f"temp={temperature}, chunk_cap={args.batch_size}, "
                          f"gen_length={cfg.gen_length}, steps={cfg.steps}")

                    for i, problem in enumerate(problems):
                        task_id = problem["task_id"]
                        prompt = problem["prompt"]

                        if args.dry_run:
                            samples = stub_overgen_samples(problem, pool_size)
                            gen_time = 0.0
                        else:
                            samples, gen_time = generate_pool(
                                generator, prompt, pool_size, args.batch_size, cfg)
                        gen_times.append(gen_time)

                        # execution-based correctness for every sample in the pool
                        cleaned_all = []
                        results_all = []
                        correct_flags = []
                        for s in samples:
                            cleaned = clean_code_for_harness(prompt, s)
                            res = check_correctness(problem, cleaned, timeout=args.timeout)
                            cleaned_all.append(cleaned)
                            results_all.append(res)
                            correct_flags.append(res["passed"])

                        # embed pool -> k-centre filter to 16 (seed = first sample)
                        embeddings = stub_embed(samples) if args.dry_run else embed_texts(
                            shared["eval_model"], samples)
                        selected_idx = kcentre_select(embeddings, 16, seed_idx=0)
                        selected_set = set(selected_idx)
                        n_first16 = min(16, len(samples))
                        first16_idx = list(range(n_first16))

                        selected_pass = 1.0 if any(correct_flags[j] for j in selected_idx) else 0.0
                        first16_pass = 1.0 if any(correct_flags[:n_first16]) else 0.0
                        oracle_pass = 1.0 if any(correct_flags) else 0.0
                        selected_flags.append(selected_pass)
                        first16_flags.append(first16_pass)
                        oracle_flags.append(oracle_pass)

                        div_selected = subset_diversity(embeddings, selected_idx)
                        div_first16 = subset_diversity(embeddings, first16_idx)
                        div_selected_scores.append(div_selected)

                        for sample_idx, (s, cleaned, res) in enumerate(
                                zip(samples, cleaned_all, results_all)):
                            is_sel = sample_idx in selected_set
                            in_f16 = sample_idx < n_first16
                            results_table.add_data(
                                task_id, prompt, s, cleaned, res["result"],
                                res["passed"], is_sel, in_f16)
                            writer.add_sample({
                                "task_id": task_id,
                                "sample_idx": sample_idx,
                                "prompt": prompt,
                                "completion": s,
                                "cleaned_code": cleaned,
                                "result": res["result"],
                                "passed": res["passed"],
                                "selected": is_sel,
                                "in_first16": in_f16,
                                "pool_size": pool_size,
                                "gen_time": gen_time,
                            })
                        writer.add_problem({
                            "task_id": task_id,
                            "n_correct": sum(correct_flags),
                            "n_samples": len(samples),
                            "selected_pass_at_16": selected_pass,
                            "first16_pass_at_16": first16_pass,
                            oracle_key: oracle_pass,
                            "div_selected": div_selected,
                            "div_first16": div_first16,
                            "gen_time": gen_time,
                        })

                        print(f"[{i + 1}/{len(problems)}] {task_id}: "
                              f"pool-correct {sum(correct_flags)}/{len(samples)} | "
                              f"sel@16 {selected_pass:.0f} first16 {first16_pass:.0f} "
                              f"oracle {oracle_pass:.0f} | time {gen_time:.2f}s")

                    metrics = {
                        "selected_pass_at_16": float(np.mean(selected_flags)) if selected_flags else 0.0,
                        "first16_pass_at_16": float(np.mean(first16_flags)) if first16_flags else 0.0,
                        oracle_key: float(np.mean(oracle_flags)) if oracle_flags else 0.0,
                        "avg_div_selected": float(np.mean(div_selected_scores)) if div_selected_scores else 0.0,
                        "avg_time": float(np.mean(gen_times)) if gen_times else 0.0,
                        "std_time": float(np.std(gen_times)) if gen_times else 0.0,
                        "n_problems_evaluated": len(gen_times),
                        "pool_size": pool_size,
                        "temperature": temperature,
                    }
                    print(f"RESULTS {run_name}: "
                          f"selected@16 {metrics['selected_pass_at_16']:.4f} | "
                          f"first16 {metrics['first16_pass_at_16']:.4f} | "
                          f"oracle@{pool_size} {metrics[oracle_key]:.4f}")

                    wandb.log({**metrics, "results_table": results_table})
                    writer.finish(metrics)
                    print(f">>> Saved local results: {writer.jsonl_path} | {writer.csv_path}")

                finally:
                    wandb.finish()


if __name__ == "__main__":
    main()
