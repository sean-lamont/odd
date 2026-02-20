import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_pass1_vs_pass16_compare_baseline(json_path):
    # 1. Load Data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_path} not found.")
        return

    # 2. Flatten JSON into a list of runs
    rows = []
    ALLOWED_STRATEGIES = ["baseline", "batched_orth"]

    for problem_id, runs_list in data.items():
        for run in runs_list:
            run_strategy = run.get('strategy', 'batched_orth')
            if run_strategy not in ALLOWED_STRATEGIES:
                continue

            alpha = float(run.get('alpha', 0))
            temp = float(run.get('temperature', 0))

            try:
                count = int(run.get('pass_count', 0))
            except (ValueError, TypeError):
                count = 0

            p1_run = count / 16.0
            p16_run = 1.0 if count > 0 else 0.0

            rows.append({
                'problem_id': problem_id,
                'alpha': alpha,
                'temperature': temp,
                'p1_run': p1_run,
                'p16_run': p16_run,
            })

    if not rows:
        print("No data found matching the constraints.")
        return

    df = pd.DataFrame(rows)

    # 3. STAGE 1 AGGREGATION: Average over runs per Problem, Alpha, and Temp
    problem_level = df.groupby(['problem_id', 'alpha', 'temperature'])[['p1_run', 'p16_run']].mean().reset_index()

    # 4. STAGE 2 AGGREGATION: Average over Problems
    alpha_temp_level = problem_level.groupby(['alpha', 'temperature'])[['p1_run', 'p16_run']].mean().reset_index()

    # 5. STAGE 3: Split Baseline from Strategy and Average the Strategy Alphas
    baseline_df = alpha_temp_level[alpha_temp_level['alpha'] == 0.0].copy()
    baseline_df = baseline_df.sort_values(by='temperature').reset_index(drop=True)

    strategy_raw = alpha_temp_level[alpha_temp_level['alpha'] > 0.0].copy()
    strategy_df = strategy_raw.groupby('temperature')[['p1_run', 'p16_run']].mean().reset_index()
    strategy_df = strategy_df.sort_values(by='temperature').reset_index(drop=True)

    # Merge to easily pair the points for drawing arrows
    merged_df = pd.merge(baseline_df, strategy_df, on='temperature', suffixes=('_base', '_strat'))

    if merged_df.empty:
        print("Error: Could not align baseline and strategy temperatures.")
        return

    # 6. Plotting Aesthetics Setup
    plt.figure(figsize=(9, 6.5))
    sns.set_theme(style="whitegrid", font_scale=1.1)

    color_base = '#7f8c8d'  # Neutral gray for baseline
    color_strat = sns.color_palette("mako")[2]  # Strong color for strategy

    # 7. Plot the Trajectories
    plt.plot(
        merged_df['p1_run_base'], merged_df['p16_run_base'],
        marker='s', markersize=8, markeredgecolor='white', markeredgewidth=1.2,
        color=color_base, label='Baseline ($\\alpha=0$)', linewidth=2.5, zorder=3
    )

    plt.plot(
        merged_df['p1_run_strat'], merged_df['p16_run_strat'],
        marker='o', markersize=9, markeredgecolor='white', markeredgewidth=1.2,
        color=color_strat, label='ODD (Avg $\\alpha > 0$)', linewidth=2.5, zorder=3
    )

    # 8. Draw Deltas (Arrows) and Annotate Temperatures
    for i, row in merged_df.iterrows():
        # Arrow from baseline to strategy
        plt.annotate(
            "",
            xy=(row['p1_run_strat'], row['p16_run_strat']),  # Arrow head
            xytext=(row['p1_run_base'], row['p16_run_base']),  # Arrow tail
            arrowprops=dict(arrowstyle="->", color="#95a5a6", alpha=0.8, linestyle="--", linewidth=1.5),
            zorder=2
        )

        # Annotate the temperature value near the strategy point
        temp_val = row['temperature']
        label = f"$\\theta={temp_val:g}$"

        # Shift text dynamically so it doesn't overlap the lines
        shift_x = 8 if row['p1_run_strat'] > row['p1_run_base'] else -8
        ha_align = 'left' if row['p1_run_strat'] > row['p1_run_base'] else 'right'

        plt.annotate(
            label,
            (row['p1_run_strat'], row['p16_run_strat']),
            textcoords="offset points",
            xytext=(shift_x, 8),
            ha=ha_align,
            fontsize=12,
            fontweight='bold',
            color=color_strat
        )

    # 9. Dynamic Bounds with Padding
    all_x = pd.concat([merged_df['p1_run_base'], merged_df['p1_run_strat']])
    all_y = pd.concat([merged_df['p16_run_base'], merged_df['p16_run_strat']])

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    eps_x = max((x_max - x_min) * 0.15, 0.02)
    eps_y = max((y_max - y_min) * 0.15, 0.02)

    plt.xlim(x_min - eps_x, x_max + eps_x)
    plt.ylim(y_min - eps_y, y_max + eps_y)

    # 10. Titles and Labels
    plt.xlabel('Pass@1 (Average Single-Sample Accuracy)', fontsize=13, fontweight='bold')
    plt.ylabel('Pass@16 (Cumulative Coverage)', fontsize=13, fontweight='bold')

    # Legend & Grid
    plt.legend(fontsize='11', loc='lower right', frameon=True, shadow=False, borderpad=1)
    plt.grid(True, linestyle='-', alpha=0.4)
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig('pareto_temp_he.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# --- Usage ---
if __name__ == "__main__":
    plot_pass1_vs_pass16_compare_baseline('he_table_batched.json')