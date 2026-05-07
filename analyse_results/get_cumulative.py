import json
from collections import defaultdict


def generate_latex_table(datasets):
    """
    Parses datasets and directly outputs a formatted LaTeX table.
    """
    # ALLOWED_STRATEGIES = ["baseline", "odd"]
    ALLOWED_STRATEGIES = ["baseline", "batched_orth", 'odd'] # legacy name for ODD strategy, provided csv uses this

    all_stats = {}
    all_totals = {}

    for dataset_name, json_path in datasets.items():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"% Error: File {json_path} not found.")
            continue

        stats_summary = defaultdict(lambda: {'solved_ids': set(), 'total_trials': 0})
        total_problems = set()

        for problem_id, runs_list in data.items():
            total_problems.add(problem_id)

            for run in runs_list:
                run_strategy = run.get('strategy', 'odd')
                if run_strategy not in ALLOWED_STRATEGIES:
                    continue

                try:
                    alpha = float(run.get('alpha'))
                except (ValueError, TypeError):
                    alpha = run.get('alpha')

                try:
                    count = int(run.get('pass_count', 0))
                except (ValueError, TypeError):
                    count = 0

                stats_summary[alpha]['total_trials'] += 1
                if count > 0:
                    stats_summary[alpha]['solved_ids'].add(problem_id)

        all_stats[dataset_name] = stats_summary
        all_totals[dataset_name] = len(total_problems)

    print("\\begin{table}[t]")
    print("\\centering")
    print(
        "\\caption{Cumulative problem coverage across all temperature settings. Orthogonal repulsion ($\\alpha > 0$) increases coverage on open-ended generation tasks (HumanEval) while maintaining robust performance on deterministic reasoning tasks (GSM8K).}")
    print("\\label{tab:cumulative_coverage}")
    print("\\begin{tabular}{@{}llcc@{}}")
    print("\\toprule")
    print(
        "\\textbf{Dataset} & \\textbf{$\\alpha$ (Repulsion)} & \\textbf{Solved (Union)} & \\textbf{Coverage} \\\\ \\midrule")

    for idx, (dataset_name, stats_summary) in enumerate(all_stats.items()):
        sorted_alphas = sorted(stats_summary.keys())
        num_rows = len(sorted_alphas)
        total_probs = all_totals[dataset_name]

        max_solved = max([len(d['solved_ids']) for d in stats_summary.values()]) if stats_summary else 0

        for i, alpha in enumerate(sorted_alphas):
            d = stats_summary[alpha]
            solved_count = len(d['solved_ids'])
            trials = d['total_trials']

            coverage = (solved_count / total_probs) * 100 if total_probs else 0.0

            # Apply bolding to the maximum values
            if solved_count == max_solved:
                solved_str = f"\\textbf{{{solved_count}}}"
                cov_str = f"\\textbf{{{coverage:.2f}\\%}}"
            else:
                solved_str = f"{solved_count}"
                cov_str = f"{coverage:.2f}\\%"

            alpha_str = "0.0 (Baseline)" if alpha == 0.0 else str(alpha)

            # Multirow label only on the first iteration for the dataset
            dataset_str = f"\\multirow{{{num_rows}}}{{*}}{{\\textbf{{{dataset_name}}}}}" if i == 0 else ""

            # Print the formatted row with the trials as a LaTeX comment
            row = f"{dataset_str:<35} & {alpha_str:<15} & {solved_str:<22} & {cov_str:<15} \\\\ % Total Trials: {trials}"
            print(row)

        if idx < len(all_stats) - 1:
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    target_datasets = {
        'GSM8K': 'gsm8k_table.json',
        'HumanEval': 'humaneval_table.json'
    }
    generate_latex_table(target_datasets)