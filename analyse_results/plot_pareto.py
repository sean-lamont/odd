import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_pass1_vs_pass16_corrected(json_path, filename):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_path} not found.")
        return

    rows = []
    # ALLOWED_STRATEGIES = ["baseline", "odd"]
    ALLOWED_STRATEGIES = ["baseline", "batched_orth", "odd"]

    for problem_id, runs_list in data.items():
        for run in runs_list:
            run_strategy = run.get('strategy', 'odd')
            if run_strategy not in ALLOWED_STRATEGIES:
                continue

            alpha = run.get('alpha')
            temp = run.get('temperature')

            try:
                count = int(run.get('pass_count', 0))
            except (ValueError, TypeError):
                count = 0

            p1_run = count / 16.0
            p16_run = 1.0 if count > 0 else 0.0

            rows.append({
                'problem_id': problem_id,
                'alpha': float(alpha),
                'temperature': float(temp),
                'p1_run': p1_run,
                'p16_run': p16_run,
            })

    if not rows:
        print("No data found matching the allowed strategies.")
        return

    df = pd.DataFrame(rows)

    problem_level = df.groupby(['problem_id', 'alpha', 'temperature'])[['p1_run', 'p16_run']].mean().reset_index()
    problem_level = problem_level.rename(columns={'p1_run': 'p1_problem', 'p16_run': 'p16_problem'})
    final_df = problem_level.groupby(['alpha', 'temperature'])[['p1_problem', 'p16_problem']].mean().reset_index()

    plt.figure(figsize=(9, 6.5))

    sns.set_theme(style="whitegrid", font_scale=1.1)

    unique_alphas = sorted(final_df['alpha'].unique())
    palette = sns.color_palette("mako", n_colors=len(unique_alphas))

    for i, alpha in enumerate(unique_alphas):
        subset = final_df[final_df['alpha'] == alpha].sort_values(by='temperature')

        # Plot lines and markers together
        plt.plot(
            subset['p1_problem'],
            subset['p16_problem'],
            marker='o',
            markersize=9,  # Larger dots
            markeredgecolor='white',  # White border makes dots pop
            markeredgewidth=1.5,  # Thickness of the white border
            linewidth=2.5,  # Thicker, bolder lines
            color=palette[i],
            label=f'$\\alpha={alpha}$',
            zorder=3  # Ensure lines are drawn over the grid
        )

    x_min, x_max = final_df['p1_problem'].min(), final_df['p1_problem'].max()
    y_min, y_max = final_df['p16_problem'].min(), final_df['p16_problem'].max()

    eps_x = max((x_max - x_min) * 0.12, 0.02)
    eps_y = max((y_max - y_min) * 0.12, 0.02)

    plt.xlim(x_min - eps_x, x_max + eps_x)
    plt.ylim(y_min - eps_y, y_max + eps_y)

    plt.xlabel('Pass@1 (Average Single-Sample Accuracy)', fontsize=12, fontweight='500')
    plt.ylabel('Pass@16', fontsize=12, fontweight='500')

    plt.legend(title='Repulsion ($\\alpha$)', title_fontsize='11', fontsize='10',
               loc='upper left', frameon=True, shadow=False, borderpad=1)

    plt.grid(True, linestyle='-', alpha=0.4)
    sns.despine(left=True, bottom=True)  # Removes the hard black box around the plot

    plt.annotate(
        'Decreasing Temperature $\\rightarrow$',
        xy=(0.90, 0.05), xycoords='axes fraction',
        ha='right', va='bottom', fontsize=10, color='gray', style='italic'
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_pass1_vs_pass16_corrected('humaneval_table.json', 'pareto_humaneval.pdf')
    plot_pass1_vs_pass16_corrected('gsm8k_table.json', 'pareto_gsm8k.pdf')
