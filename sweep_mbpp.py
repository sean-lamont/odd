"""MBPP benchmark harness for the ODD rebuttal experiments.

Evaluates strategies (baseline / odd / ...) on the full MBPP test split
(task_ids 11-510, 500 problems) with the same per-problem batch generation,
empirical prefix-slice pass@k and diversity logging as sweep_human_eval.py,
but WITHOUT the Optuna/Postgres orchestration: a plain argparse grid loop.

Correctness is checked by executing the generated code plus all three MBPP
asserts inside the human_eval execution sandbox (multiprocess + timeout +
reliability_guard).

Examples:
    # CPU-only smoke test (stub generator, offline wandb):
    python sweep_mbpp.py --dry-run --n-problems 3 --batch-size 8

    # 50-problem quick read on GPU:
    python sweep_mbpp.py --n-problems 50

    # Full sweep:
    python sweep_mbpp.py
"""

import time

from datasets import load_dataset

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
from human_eval.execution import check_correctness


# Copied from sweep_human_eval.py (kept local so we don't import from a script
# module that loads the model at import time).
def clean_code_for_harness(prompt, completion):
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]
    return completion


def build_mbpp_prompt(row):
    """Zero-shot instruct prompt; the first test case doubles as the function
    signature hint (standard MBPP practice)."""
    return (
        "You are an expert Python programmer. Write a Python function to solve "
        f"the following problem:\n{row['text']}\n"
        f"Your code should pass this test:\n{row['test_list'][0]}\n"
        "Write only the function."
    )


def build_mbpp_problem(row):
    """Adapt an MBPP row into the HumanEval-style problem dict expected by
    human_eval.execution.check_correctness.

    check_correctness builds:
        prompt + completion + "\n" + test + "\n" + "check(<entry_point>)"
    MBPP asserts reference the target function by name, so the completion just
    has to define it in the same namespace; check(candidate) ignores its
    argument and we pass entry_point="None".
    """
    setup = row.get("test_setup_code") or ""
    test_body = "\n".join("    " + t for t in row["test_list"])
    test = setup + "\n\ndef check(candidate):\n" + test_body + "\n"
    return {"task_id": row["task_id"], "prompt": "", "test": test, "entry_point": "None"}


def stub_mbpp_samples(row, batch_size):
    """Canned generations for --dry-run: the dataset's reference solution
    (correct) at sample indices 1 and 5, assorted broken completions elsewhere.
    Expected empirical scores: pass@1 = 0.0, pass@k = 1.0 for k >= 2."""
    correct = f"```python\n{row['code']}\n```"
    samples = []
    for i in range(batch_size):
        if i in (1, 5):
            samples.append(correct + f"\n# sample {i}")
        elif i == 2:
            samples.append("def broken_syntax(:\n    return")
        elif i == 3:
            samples.append("```python\ndef unrelated_helper(x):\n    return x\n```")
        else:
            samples.append(f"def wrong_answer_{i}(*args, **kwargs):\n    return {i}")
    return samples


def main():
    parser = build_arg_parser(
        description="MBPP grid sweep (no Optuna/Postgres) for the ODD rebuttal",
        default_project="rebuttal-mbpp",
        default_gen_length=256,
    )
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Sandbox timeout per sample in seconds")
    args = parser.parse_args()

    print(">>> Loading MBPP (full config, test split: task_ids 11-510)...")
    # Canonical HF location of the full "mbpp" config (bare "mbpp" is a legacy
    # alias that newer datasets/huggingface_hub versions reject).
    dataset = load_dataset("google-research-datasets/mbpp", split="test")
    problems = list(dataset)
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
        run_name = f"mbpp_{strategy}_alpha{alpha:g}_temp{temperature:g}_run{run_idx}"

        run = init_wandb(args, cfg, run_name)
        results_table = wandb.Table(
            columns=["task_id", "prompt", "completion", "cleaned_code", "result", "passed", "diversity"]
        )
        writer = RunWriter(
            args.results_dir, "mbpp", run_name,
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

            for i, row in enumerate(problems):
                task_id = row["task_id"]
                prompt = build_mbpp_prompt(row)

                start_t = time.time()
                if args.dry_run:
                    samples = stub_mbpp_samples(row, args.batch_size)
                else:
                    _, samples = generator.generate(
                        prompt=prompt,
                        batch_size=cfg.batch_size,
                        steps=cfg.steps,
                        gen_length=cfg.gen_length,
                        temperature=cfg.temperature,
                    )
                gen_times.append(time.time() - start_t)

                # Evaluate batch in the human_eval sandbox
                he_problem = build_mbpp_problem(row)
                correct_flags = []
                batch_results = []
                for s in samples:
                    cleaned = clean_code_for_harness(prompt, s)
                    res = check_correctness(he_problem, cleaned, timeout=args.timeout)
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

                print(f"[{i + 1}/{len(problems)}] task {task_id}: "
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
