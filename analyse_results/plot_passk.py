import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np


def plot_pass_at_k_grid_bespoke(
        csv_path,
        strategy_col,
        temp_col,
        filename,
        metric_prefix='pass_at_',
        max_k=16,
        alpha_key='alpha',
        strategy_name_key='name',
        allowed_strategies=["baseline", "odd"],
):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    df = df.dropna(subset=[strategy_col])
    try:
        df[strategy_col] = df[strategy_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing dictionary: {e}")
        return

    strategy_df = pd.json_normalize(df[strategy_col])
    strategy_df.index = df.index

    overlap = set(df.columns).intersection(set(strategy_df.columns))
    overlap.discard(strategy_col)
    if overlap:
        df = df.drop(columns=list(overlap))

    df = pd.concat([df.drop(columns=[strategy_col]), strategy_df], axis=1)

    if strategy_name_key in df.columns and allowed_strategies:
        df = df[df[strategy_name_key].isin(allowed_strategies)]

    expanded_alpha_col = alpha_key
    df[expanded_alpha_col] = pd.to_numeric(df[expanded_alpha_col], errors='coerce')
    df = df.dropna(subset=[expanded_alpha_col, temp_col])

    metric_cols = [f"{metric_prefix}{i}" for i in range(1, max_k + 1)]
    melted_df = pd.melt(
        df,
        id_vars=[temp_col, expanded_alpha_col],
        value_vars=metric_cols,
        var_name='k_label',
        value_name='pass_at_k'
    )
    melted_df = melted_df.dropna(subset=['pass_at_k'])
    melted_df['k'] = melted_df['k_label'].str.extract(r'(\d+)').astype(int)

    final_df = melted_df.groupby([temp_col, expanded_alpha_col, 'k'])['pass_at_k'].mean().reset_index()

    unique_temps = sorted(final_df[temp_col].unique())[:5]  # Enforce max 5 temps
    unique_alphas = sorted(final_df[expanded_alpha_col].unique())

    if not unique_temps:
        print("No valid data found to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True, sharey=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    non_zero_alphas = [a for a in unique_alphas if a != 0]
    palette = sns.color_palette("mako", n_colors=len(non_zero_alphas))
    alpha_to_color = {a: palette[i] for i, a in enumerate(non_zero_alphas)}

    target_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]]
    legend_ax = axes[0, 2]

    legend_ax.axis('off')

    for idx, (temp, ax) in enumerate(zip(unique_temps, target_axes)):
        subset_temp = final_df[final_df[temp_col] == temp]

        for alpha in unique_alphas:
            subset_alpha = subset_temp[subset_temp[expanded_alpha_col] == alpha].sort_values(by='k')

            is_baseline = (alpha == 0)
            lw = 3.0 if is_baseline else 2.0
            ls = '--' if is_baseline else '-'
            color = '#333333' if is_baseline else alpha_to_color[alpha]

            # Updated the baseline label here
            label = '$\\alpha=0$ (Baseline)' if is_baseline else f'$\\alpha={alpha}$'

            ax.plot(
                subset_alpha['k'],
                subset_alpha['pass_at_k'],
                label=label if idx == 0 else None,  # Only need labels once for the legend
                linewidth=lw,
                linestyle=ls,
                color=color,
                marker='o',
                markersize=4,
                zorder=5 if is_baseline else 3
            )

        # Subplot Aesthetics
        ax.set_title(f"$\\theta = {temp}$", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(1, max_k)

        if max_k == 16:
            ax.set_xticks([1, 4, 8, 12, 16])
        else:
            ax.set_xticks(range(1, max_k + 1, max(1, max_k // 4)))

        ax.grid(True, linestyle='-', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Use supxlabel and supylabel for a single centered label across the whole figure
    fig.supxlabel("$k$ (Number of Samples)", fontsize=14, fontweight=500)
    fig.supylabel("Pass@$k$ (Empirical)", fontsize=14, fontweight=500)

    y_min = final_df['pass_at_k'].min()
    y_max = final_df['pass_at_k'].max()
    eps_y = max((y_max - y_min) * 0.05, 0.02)

    plt.ylim(y_min - eps_y, y_max + eps_y)

    handles, labels = target_axes[0].get_legend_handles_labels()
    legend_ax.legend(
        handles, labels,
        title="Repulsion ($\\alpha$)",
        title_fontsize='13',
        loc='center',
        frameon=True,  # Turned on the border
        edgecolor='black',  # Set border color
        framealpha=1.0,  # Make the background solid
        fontsize='12'
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_pass_at_k_grid_bespoke(
        csv_path='gsm8k.csv',
        strategy_col='strategy',
        temp_col='temperature',
        filename='gsm8k_passk_grid.pdf',
        metric_prefix='pass_at_',
        max_k=16,
        alpha_key='alpha',
        strategy_name_key='name',
        allowed_strategies=['baseline', 'odd']
    )
    plot_pass_at_k_grid_bespoke(
        csv_path='humaneval.csv',
        strategy_col='strategy',
        temp_col='temperature',
        filename='humaneval_passk_grid.pdf',
        metric_prefix='pass_at_',
        max_k=16,
        alpha_key='alpha',
        strategy_name_key='name',
        allowed_strategies=['baseline', 'odd']
    )

