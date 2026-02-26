import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# pareto averaged over each alpha
def plot_pass1_vs_pass16_avg_alpha(json_path, filename):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_path} not found.")
        return

    rows = []
    ALLOWED_STRATEGIES = ["baseline", "batched_orth"]

    for problem_id, runs_list in data.items():
        for run in runs_list:
            run_strategy = run.get('strategy', 'batched_orth')
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

    alpha_temp_level = problem_level.groupby(['alpha', 'temperature'])[['p1_run', 'p16_run']].mean().reset_index()

    final_df = alpha_temp_level.groupby('alpha')[['p1_run', 'p16_run']].mean().reset_index()

    final_df = final_df.sort_values(by='alpha')

    plt.figure(figsize=(9, 6.5))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    trajectory_color = sns.color_palette("mako")[1]

    plt.plot(
        final_df['p1_run'],
        final_df['p16_run'],
        marker='o',
        markersize=10,
        markeredgecolor='white',
        markeredgewidth=1.5,
        linewidth=2.5,
        color=trajectory_color,
        zorder=3
    )

    for i, row in final_df.iterrows():
        alpha_val = row['alpha']

        if alpha_val == 0:
            label = "0 (Baseline)"
            font_w = 'bold'
        else:
            # Format alpha cleanly (e.g., 2.0 -> 2)
            label = f"$\\alpha={alpha_val:g}$"
            font_w = '500'

        # Dynamically place text slightly above and to the right of the points
        plt.annotate(
            label,
            (row['p1_run'], row['p16_run']),
            textcoords="offset points",
            xytext=(8, 8),
            ha='left',
            fontsize=11,
            fontweight=font_w,
            color='#333333'
        )

    x_min, x_max = final_df['p1_run'].min(), final_df['p1_run'].max()
    y_min, y_max = final_df['p16_run'].min(), final_df['p16_run'].max()

    eps_x = max((x_max - x_min) * 0.15, 0.02)
    eps_y = max((y_max - y_min) * 0.15, 0.02)

    plt.xlim(x_min - eps_x, x_max + eps_x)
    plt.ylim(y_min - eps_y, y_max + eps_y)

    plt.xlabel('Pass@1 (Average Single-Sample Accuracy)', fontsize=13, fontweight='bold')
    plt.ylabel('Pass@16 (Cumulative Coverage)', fontsize=13, fontweight='bold')

    plt.grid(True, linestyle='-', alpha=0.4)
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_pass1_vs_pass16_avg_alpha('gsm8k_table.json', 'gsm8k_pareto_alpha_avg.pdf')
    plot_pass1_vs_pass16_avg_alpha('humaneval_table.json', 'humaneval_pareto_alpha_avg.pdf')
