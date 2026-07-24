"""MATH-500 benchmark harness for the ODD rebuttal experiments.

Evaluates strategies (baseline / odd / ...) on HuggingFaceH4/MATH-500 (test
split, 500 problems) with the same per-problem batch generation, empirical
prefix-slice pass@k and diversity logging as sweep_gsm8k.py, but WITHOUT the
Optuna/Postgres orchestration: a plain argparse grid loop.

Correctness: the last \\boxed{...} in each generation (balanced-brace parse)
is compared to the dataset's `answer` field. If the `math_verify` package is
installed (pinned in requirements-rebuttal.txt: math-verify==0.9.0) it is used
for robust symbolic equivalence (fractions, sqrt forms, etc.); otherwise we
fall back to normalized string matching plus numeric equality.

Examples:
    # CPU-only smoke test (stub generator, offline wandb):
    python sweep_math500.py --dry-run --n-problems 3 --batch-size 8

    # 50-problem quick read on GPU:
    python sweep_math500.py --n-problems 50

    # Full sweep:
    python sweep_math500.py
"""

import re
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

# math_verify is optional: prefer it when present, fall back gracefully.
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    HAVE_MATH_VERIFY = True
except Exception:
    HAVE_MATH_VERIFY = False


def build_math_prompt(row):
    """Mirror the GSM8K prompt style, but ask for the answer in \\boxed{}."""
    return (
        f"Question: {row['problem']}\n"
        "Let's think step by step. Put your final answer inside \\boxed{}.\n"
        "Answer:"
    )


def extract_last_boxed(text):
    """Return the contents of the last \\boxed{...} in text, using a
    balanced-brace parse (handles nested braces like \\boxed{\\frac{1}{2}}).
    Returns None if no boxed answer is found."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] in " \t":
        i += 1
    if i >= len(text):
        return None
    if text[i] != "{":
        # rare "\boxed 5" form: take the next non-whitespace token
        m = re.match(r"([^$\s]+)", text[i:])
        return m.group(1) if m else None
    depth = 0
    start = i + 1
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:j]
    return None  # unbalanced braces


def _normalize_math(s):
    s = s.strip()
    for tok in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "$"]:
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\s+", "", s)
    m = re.fullmatch(r"\\text\{(.*)\}", s)
    if m:
        s = m.group(1)
    return s


def _to_float(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _fallback_equal(pred, gold):
    p, g = _normalize_math(pred), _normalize_math(gold)
    if p == g:
        return True
    fp, fg = _to_float(p), _to_float(g)
    if fp is not None and fg is not None:
        # numeric equality also covers trailing zeros (0.50 == 0.5)
        return abs(fp - fg) < 1e-6
    return False


def is_answer_correct(pred, gold):
    """pred: extracted boxed string (or None); gold: dataset `answer` field."""
    if pred is None:
        return False
    if HAVE_MATH_VERIFY:
        try:
            if _mv_verify(_mv_parse(f"${gold}$"), _mv_parse(f"${pred}$")):
                return True
        except Exception:
            pass
    return _fallback_equal(pred, gold)


def stub_math_samples(row, batch_size):
    """Canned generations for --dry-run: the gold answer boxed at sample
    indices 1 and 5, wrong/absent boxes elsewhere. Expected empirical scores:
    pass@1 = 0.0, pass@k = 1.0 for k >= 2."""
    correct = ("We compute this step by step. Therefore the final answer is "
               f"$\\boxed{{{row['answer']}}}$.")
    samples = []
    for i in range(batch_size):
        if i in (1, 5):
            samples.append(correct + f" (sample {i})")
        elif i == 2:
            samples.append("I cannot determine the answer to this question.")
        elif i == 3:
            samples.append("After simplification we get $\\boxed{\\frac{1}{999999}}$.")
        else:
            samples.append(f"A quick estimate gives $\\boxed{{-42424{i}}}$.")
    return samples


def main():
    parser = build_arg_parser(
        description="MATH-500 grid sweep (no Optuna/Postgres) for the ODD rebuttal",
        default_project="rebuttal-math500",
        default_gen_length=256,
    )
    args = parser.parse_args()

    print(f">>> math_verify available: {HAVE_MATH_VERIFY}")
    print(">>> Loading HuggingFaceH4/MATH-500 (test split)...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
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
        run_name = f"math500_{strategy}_alpha{alpha:g}_temp{temperature:g}_run{run_idx}"

        run = init_wandb(args, cfg, run_name)
        results_table = wandb.Table(
            columns=["problem", "gold", "generated", "extracted", "is_correct", "diversity"]
        )
        writer = RunWriter(
            args.results_dir, "math500", run_name,
            csv_fieldnames=["problem_idx", "gold", "n_correct", "n_samples", "pass_at_1",
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
                gold = row["answer"]
                prompt = build_math_prompt(row)

                start_t = time.time()
                if args.dry_run:
                    samples = stub_math_samples(row, args.batch_size)
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
                extracted = []
                for s in samples:
                    pred = extract_last_boxed(s)
                    extracted.append(pred)
                    correct_flags.append(is_answer_correct(pred, gold))

                cumulative_correct = update_pass_at_k(correct_flags, args.batch_size, pass_at_k_totals)
                div = diversity_fn(samples)
                diversity_scores.append(div)

                for sample_idx, (s, pred, is_correct) in enumerate(zip(samples, extracted, correct_flags)):
                    results_table.add_data(row["problem"], gold, s, pred, is_correct, div)
                    writer.add_sample({
                        "problem_idx": i,
                        "sample_idx": sample_idx,
                        "problem": row["problem"],
                        "gold": gold,
                        "generated": s,
                        "extracted": pred,
                        "is_correct": is_correct,
                        "diversity": div,
                        "gen_time": gen_times[-1],
                    })
                writer.add_problem({
                    "problem_idx": i,
                    "gold": gold,
                    "n_correct": sum(correct_flags),
                    "n_samples": len(samples),
                    "pass_at_1": pass_at_k_totals[1][-1],
                    f"pass_at_{args.batch_size}": pass_at_k_totals[args.batch_size][-1],
                    "diversity": div,
                    "gen_time": gen_times[-1],
                })

                print(f"[{i + 1}/{len(problems)}]: "
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
