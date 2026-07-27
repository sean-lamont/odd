"""GSM8K benchmark harness for the ODD rebuttal experiments.

Replicates EXACTLY the evaluation semantics of sweep_gsm8k.py:
  - prompt: "Question: {q}\\nLet's think step by step.\\nAnswer:",
  - answer extraction: last number in the generation (extract_answer_num) vs
    the "#### N" gold (extract_gold_num), correct iff |pred - gold| < 1e-4,
  - problems whose gold cannot be parsed are skipped,
  - empirical prefix-slice pass@k,
  - gen_length default 128,
but WITHOUT the Optuna/Postgres orchestration: a plain argparse grid loop
(see bench_common.py), local JSONL/CSV outputs, wandb project rebuttal-gsm8k.

--n-problems defaults to 200 (the subset used in the paper's sweep); pass -1
for the full 1319-problem test split.

Examples:
    # CPU-only smoke test (stub generator, offline wandb):
    python sweep_gsm8k_plain.py --dry-run --n-problems 3 --batch-size 8

    # 50-problem LLaDA-1.5 quick read on GPU:
    python sweep_gsm8k_plain.py --model-config llada15 --n-problems 50

    # Full test split:
    python sweep_gsm8k_plain.py --model-config llada15 --n-problems -1
"""

import re
import time

from datasets import load_dataset

import wandb
from bench_common import (
    aggregate_metrics,
    apply_prompt_style,
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


# Copied verbatim from sweep_gsm8k.py.
def extract_answer_num(text):
    try:
        text = text.replace(',', '')
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums: return float(nums[-1])
    except Exception as e:
        print(e)
    return None


# Copied verbatim from sweep_gsm8k.py.
def extract_gold_num(answer_str):
    if "####" in answer_str:
        try:
            val = answer_str.split("####")[1].strip()
            return float(val.replace(',', ''))
        except:
            pass
    return None


def build_gsm8k_prompt(question):
    # Identical to sweep_gsm8k.py.
    return f"Question: {question}\nLet's think step by step.\nAnswer:"


def stub_gsm8k_samples(gold, batch_size):
    """Canned generations for --dry-run: the gold number (as the LAST number in
    the text, which is what extract_answer_num picks up) at sample indices 1
    and 5, wrong/absent numbers elsewhere. Expected empirical scores:
    pass@1 = 0.0, pass@k = 1.0 for k >= 2."""
    samples = []
    for i in range(batch_size):
        if i == 1:
            samples.append(f"Let's add everything up carefully. The answer is {gold:g}")
        elif i == 5:
            samples.append(f"An alternative derivation also gives the answer as {gold:g}")
        elif i == 2:
            samples.append("I cannot work this one out.")  # no number -> None
        else:
            samples.append(f"A rough estimate puts the total at {gold + 3 + i:g}")
    return samples


def main():
    parser = build_arg_parser(
        description="GSM8K grid sweep (no Optuna/Postgres) for the ODD rebuttal; "
                    "same eval semantics as sweep_gsm8k.py. --n-problems defaults "
                    "to 200 (paper subset); -1 = full 1319-problem test split",
        default_project="rebuttal-gsm8k",
        default_gen_length=128,
    )
    parser.set_defaults(n_problems=200)  # match the paper's sweep subset
    args = parser.parse_args()

    print(">>> Loading GSM8K (main config, test split)...")
    # Canonical HF location (bare "gsm8k" is a legacy alias that newer
    # datasets/huggingface_hub versions reject).
    dataset = load_dataset("openai/gsm8k", "main", split="test")
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
        run_name = f"gsm8k_{strategy}_alpha{alpha:g}_temp{temperature:g}_run{run_idx}"

        run = init_wandb(args, cfg, run_name)
        results_table = wandb.Table(
            columns=["question", "gold", "generated", "is_correct", "diversity"]
        )
        writer = RunWriter(
            args.results_dir, "gsm8k", run_name,
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
                q = row["question"]
                gold = extract_gold_num(row["answer"])
                if gold is None:
                    continue  # identical to sweep_gsm8k.py

                # Default: zero-shot CoT prompt (identical to sweep_gsm8k.py).
                # Model configs with prompt_style: fewshot_gsm8k (base models,
                # e.g. rnd1) get the standard k-shot prefix instead; the prefix
                # lives entirely in the prompt segment, never in the canvas.
                prompt = apply_prompt_style(cfg, q, build_gsm8k_prompt(q))

                start_t = time.time()
                if args.dry_run:
                    samples = stub_gsm8k_samples(gold, args.batch_size)
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
                for s in samples:
                    val = extract_answer_num(s)
                    is_correct = (val is not None and abs(val - gold) < 1e-4)
                    correct_flags.append(is_correct)

                cumulative_correct = update_pass_at_k(correct_flags, args.batch_size, pass_at_k_totals)
                div = diversity_fn(samples)
                diversity_scores.append(div)

                for sample_idx, (s, is_correct) in enumerate(zip(samples, correct_flags)):
                    results_table.add_data(q, gold, s, is_correct, div)
                    writer.add_sample({
                        "problem_idx": i,
                        "sample_idx": sample_idx,
                        "question": q,
                        "gold": gold,
                        "generated": s,
                        "extracted": extract_answer_num(s),
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
