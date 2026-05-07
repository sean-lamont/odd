import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def extract_fair_stats(datasets):
    """
    Extracts solved problem sets strictly grouped by (Dataset, Temp, Alpha).
    This ensures a fair 1-to-1 comparison of trials.
    """
    ALLOWED_STRATEGIES = ["baseline", "odd"]

    # Structure: stats[dataset][temp][alpha] = set(solved_ids)
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    totals = {}

    for dataset_name, json_path in datasets.items():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"% Error: File {json_path} not found.")
            continue

        total_problems = set()

        for problem_id, runs_list in data.items():
            total_problems.add(problem_id)

            for run in runs_list:
                strategy = run.get('strategy', 'odd')
                if strategy not in ALLOWED_STRATEGIES:
                    continue

                temp = run.get('temperature', 0.0)

                try:
                    alpha = float(run.get('alpha', 0.0))
                except (ValueError, TypeError):
                    alpha = 0.0

                # Force baseline strategy to have alpha 0.0 for grouping
                if strategy == 'baseline':
                    alpha = 0.0

                try:
                    count = int(run.get('pass_count', 0))
                except (ValueError, TypeError):
                    count = 0

                if count > 0:
                    stats[dataset_name][temp][alpha].add(problem_id)

        totals[dataset_name] = len(total_problems)

    return stats, totals


def generate_fair_latex_table(stats, totals):
    """
    Generates a pivot-style LaTeX table: Rows = Temp, Columns = Alpha.
    Includes a final row for the cumulative union across all temperatures.
    """
    print("% " + "=" * 60)
    print("% FAIR CUMULATIVE TABLE (TEMP x ALPHA)")
    print("% " + "=" * 60)
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\setlength{\\tabcolsep}{8pt}")
    print(
        "\\caption{Cumulative problem coverage (union of solved instances) evaluated fairly across configurations. Each individual temperature cell represents 4 trials. The 'All Temps' row represents the union across all 5 temperatures (20 trials). Bold values indicate the highest coverage per row.}")
    print("\\label{tab:fair_cumulative_coverage}")
    print("\\begin{tabular}{l ccccc}")
    print("\\toprule")
    print(
        "\\multirow{2}{*}{\\textbf{Dataset / Temp ($\\theta$)}} & \\multicolumn{5}{c}{\\textbf{Coverage (\\%) by Approach}} \\\\")
    print("\\cmidrule(lr){2-6}")
    print(
        " & \\textbf{Baseline} & \\textbf{$\\alpha=8$} & \\textbf{$\\alpha=16$} & \\textbf{$\\alpha=64$} & \\textbf{$\\alpha=128$} \\\\ \\midrule")

    alpha_columns = [0.0, 8.0, 16.0, 64.0, 128.0]

    for idx, (dataset_name, temp_dict) in enumerate(stats.items()):
        print(f"\\multicolumn{{6}}{{l}}{{\\textsc{{{dataset_name}}}}} \\\\ \\midrule")

        sorted_temps = sorted(temp_dict.keys())
        total_probs = totals[dataset_name]

        # 1. Print individual temperature rows
        for temp in sorted_temps:
            row_strs = [f"{temp:.1f}"]

            coverages = []
            for a in alpha_columns:
                solved = len(temp_dict[temp][a]) if a in temp_dict[temp] else 0
                coverages.append((solved / total_probs) * 100 if total_probs else 0.0)

            max_cov = max(coverages)

            for cov in coverages:
                if cov == max_cov and cov > 0:
                    row_strs.append(f"\\textbf{{{cov:.1f}}}")
                else:
                    row_strs.append(f"{cov:.1f}")

            print(" & ".join(row_strs) + " \\\\")

        # 2. Print the cumulative union row across all temperatures for this dataset
        print("\\cmidrule(lr){2-6}")
        union_row_strs = ["\\textbf{All Temps (Union)}"]
        union_coverages = []

        for a in alpha_columns:
            union_solved = set()
            for temp in sorted_temps:
                if a in temp_dict[temp]:
                    union_solved.update(temp_dict[temp][a])

            union_cov = (len(union_solved) / total_probs) * 100 if total_probs else 0.0
            union_coverages.append(union_cov)

        max_union_cov = max(union_coverages)

        for cov in union_coverages:
            if cov == max_union_cov and cov > 0:
                union_row_strs.append(f"\\textbf{{{cov:.1f}}}")
            else:
                union_row_strs.append(f"{cov:.1f}")

        print(" & ".join(union_row_strs) + " \\\\")

        if idx < len(stats) - 1:
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")


def generate_coverage_plots(stats, totals):
    """
    Generates publication-ready line plots of Coverage vs Temperature.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    datasets = list(stats.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5), sharey=False)

    if len(datasets) == 1:
        axes = [axes]

    alpha_markers = {
        0.0: ('o', 'black', 'Baseline', 'solid'),
        8.0: ('s', '#1f77b4', '$\\alpha=8$', 'dashed'),
        16.0: ('^', '#ff7f0e', '$\\alpha=16$', 'dashed'),
        64.0: ('D', '#2ca02c', '$\\alpha=64$', 'dashed'),
        128.0: ('v', '#d62728', '$\\alpha=128$', 'dashed')
    }

    for ax, dataset_name in zip(axes, datasets):
        temp_dict = stats[dataset_name]
        sorted_temps = sorted(temp_dict.keys())
        total_probs = totals[dataset_name]

        for alpha, (marker, color, label, ls) in alpha_markers.items():
            y_values = []
            for temp in sorted_temps:
                solved = len(temp_dict[temp][alpha]) if alpha in temp_dict[temp] else 0
                cov = (solved / total_probs) * 100 if total_probs else 0.0
                y_values.append(cov)

            ax.plot(sorted_temps, y_values, marker=marker, color=color,
                    label=label, linestyle=ls, linewidth=2, markersize=8)

        ax.set_title(f"{dataset_name} - Cumulative Coverage", fontweight='bold')
        ax.set_xlabel("Temperature ($\\theta$)")
        ax.set_ylabel("Coverage (%)")
        ax.set_xticks(sorted_temps)

        # Grid and aesthetics
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if dataset_name == datasets[0]:
            ax.legend(title="Approach", loc='best')

    plt.tight_layout()
    plt.savefig("coverage_vs_temperature.pdf", dpi=300, bbox_inches='tight')
    print(">>> Saved plot to 'coverage_vs_temperature.pdf'")


if __name__ == "__main__":
    target_datasets = {
        'GSM8K': 'dream_gsm8k_table.json',
        'HumanEval': 'dream_humaneval_table.json'
    }

    stats, totals = extract_fair_stats(target_datasets)
    generate_fair_latex_table(stats, totals)
    generate_coverage_plots(stats, totals)