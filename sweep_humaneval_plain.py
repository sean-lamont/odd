"""HumanEval benchmark harness for the ODD rebuttal experiments.

Replicates EXACTLY the evaluation semantics of sweep_human_eval.py:
  - prompt: problem['prompt'] passed verbatim to DiverseGenerator (which wraps
    it in the chat template),
  - completion cleanup via clean_code_for_harness (copied verbatim),
  - correctness via human_eval.execution.check_correctness (timeout 3.0s),
  - empirical prefix-slice pass@k,
  - gen_length default 256,
but WITHOUT the Optuna/Postgres orchestration: a plain argparse grid loop
(see bench_common.py), local JSONL/CSV outputs, wandb project rebuttal-humaneval.

Examples:
    # CPU-only smoke test (stub generator, offline wandb):
    python sweep_humaneval_plain.py --dry-run --n-problems 3 --batch-size 8

    # 50-problem LLaDA-1.5 quick read on GPU:
    python sweep_humaneval_plain.py --model-config llada15 --n-problems 50

    # Full run (all 164 problems):
    python sweep_humaneval_plain.py --model-config llada15
"""

import time

import wandb
from bench_common import (
    aggregate_metrics,
    build_arg_parser,
    build_generator,
    compose_cfg,
    init_wandb,
    iter_grid,
    load_shared_resources,
    make_diversity_fn,
    RunWriter,
    update_pass_at_k,
)
from human_eval.data import read_problems
from human_eval.execution import check_correctness


# Copied verbatim from sweep_human_eval.py (kept local so we don't import from
# a script module that loads the model at import time).
def clean_code_for_harness(prompt, completion):
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]
    return completion


# Known-good completions (function bodies) for --dry-run sandbox verification.
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


def stub_humaneval_samples(problem, batch_size):
    """Canned generations for --dry-run: a known-good body (where we have one)
    at sample indices 1 and 5, broken bodies elsewhere. For tasks with a canned
    solution the expected empirical scores are pass@1 = 0.0 and pass@k = 1.0
    for k >= 2; tasks without one score 0 at every k."""
    correct = STUB_SOLUTIONS.get(problem["task_id"])
    samples = []
    for i in range(batch_size):
        if correct is not None and i in (1, 5):
            samples.append(correct + f"# sample {i}\n")
        elif i == 2:
            samples.append("    return undefined_name_xyz\n")
        else:
            samples.append(f"    return {i}  # wrong on purpose\n")
    return samples


def main():
    parser = build_arg_parser(
        description="HumanEval grid sweep (no Optuna/Postgres) for the ODD rebuttal; "
                    "same eval semantics as sweep_human_eval.py",
        default_project="rebuttal-humaneval",
        default_gen_length=256,
    )
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="Sandbox timeout per sample in seconds (sweep_human_eval.py uses 3.0)")
    args = parser.parse_args()

    print(">>> Loading HumanEval (local human_eval/data/HumanEval.jsonl.gz)...")
    problems = list(read_problems().values())
    if args.n_problems > 0:
        problems = problems[: args.n_problems]
    print(f">>> Evaluating {len(problems)} problems")

    shared = None
    if not args.dry_run:
        base_cfg = compose_cfg(args, args.strategies[0], 0.0, 0.0)
        shared = load_shared_resources(base_cfg)
    diversity_fn = make_diversity_fn(args.dry_run, shared)

    for run_idx, strategy, alpha, temperature in iter_grid(args):
        cfg = compose_cfg(args, strategy, alpha, temperature)
        run_name = f"humaneval_{strategy}_alpha{alpha:g}_temp{temperature:g}_run{run_idx}"

        run = init_wandb(args, cfg, run_name)
        results_table = wandb.Table(
            columns=["task_id", "prompt", "completion", "cleaned_code", "result", "passed", "diversity"]
        )
        writer = RunWriter(
            args.results_dir, "humaneval", run_name,
            csv_fieldnames=["task_id", "n_correct", "n_samples", "pass_at_1",
                            f"pass_at_{args.batch_size}", "diversity", "gen_time"],
        )
        generator = None if args.dry_run else build_generator(cfg, shared)

        pass_at_k_totals = {k: [] for k in range(1, args.batch_size + 1)}
        diversity_scores = []
        gen_times = []

        try:
            print(f"\n>>> STARTING RUN {run_name}: strategy={strategy}, alpha={alpha}, "
                  f"temp={temperature}, batch={args.batch_size}")

            for i, problem in enumerate(problems):
                task_id = problem["task_id"]
                prompt = problem["prompt"]

                start_t = time.time()
                if args.dry_run:
                    samples = stub_humaneval_samples(problem, args.batch_size)
                else:
                    _, samples = generator.generate(
                        prompt=prompt,
                        batch_size=cfg.batch_size,
                        steps=cfg.steps,
                        gen_length=cfg.gen_length,
                        temperature=cfg.temperature,
                    )
                gen_times.append(time.time() - start_t)

                correct_flags = []
                batch_results = []
                for s in samples:
                    cleaned = clean_code_for_harness(prompt, s)
                    res = check_correctness(problem, cleaned, timeout=args.timeout)
                    batch_results.append((s, cleaned, res))
                    correct_flags.append(res["passed"])

                cumulative_correct = update_pass_at_k(correct_flags, args.batch_size, pass_at_k_totals)
                div = diversity_fn(samples)
                diversity_scores.append(div)

                for sample_idx, (s, cleaned, res) in enumerate(batch_results):
                    results_table.add_data(task_id, prompt, s, cleaned, res["result"], res["passed"], div)
                    writer.add_sample({
                        "task_id": task_id,
                        "sample_idx": sample_idx,
                        "prompt": prompt,
                        "completion": s,
                        "cleaned_code": cleaned,
                        "result": res["result"],
                        "passed": res["passed"],
                        "diversity": div,
                        "gen_time": gen_times[-1],
                    })
                writer.add_problem({
                    "task_id": task_id,
                    "n_correct": sum(correct_flags),
                    "n_samples": len(samples),
                    "pass_at_1": pass_at_k_totals[1][-1],
                    f"pass_at_{args.batch_size}": pass_at_k_totals[args.batch_size][-1],
                    "diversity": div,
                    "gen_time": gen_times[-1],
                })

                print(f"[{i + 1}/{len(problems)}] {task_id}: "
                      f"correct {sum(correct_flags)}/{len(samples)} | "
                      f"cumulative {cumulative_correct} | time {gen_times[-1]:.2f}s")

            metrics = aggregate_metrics(pass_at_k_totals, diversity_scores, gen_times)
            print(f"RESULTS {run_name}: Pass@1: {metrics['pass_at_1']:.4f} | "
                  f"Pass@{args.batch_size}: {metrics[f'pass_at_{args.batch_size}']:.4f} | "
                  f"Div: {metrics['avg_diversity']:.4f}")

            wandb.log({**metrics, "results_table": results_table})
            writer.finish(metrics)
            print(f">>> Saved local results: {writer.jsonl_path} | {writer.csv_path}")

        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
