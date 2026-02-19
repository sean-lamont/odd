# import pandas as pd
# import matplotlib.pyplot as plt
# import ast
# import numpy as np
#
#
# def plot_alpha_equally_spaced(csv_path, metric_col, strategy_col, temp_col, alpha_key='alpha'):
#     """
#     Plots alpha vs metric where the X-axis is CATEGORICAL (equally spaced).
#     Even if alphas are [2, 4, 128], they will be plotted at x=[0, 1, 2].
#     """
#
#     # 1. Load Data
#     try:
#         df = pd.read_csv(csv_path)
#     except FileNotFoundError:
#         print(f"Error: File {csv_path} not found.")
#         return
#
#     # 2. Parse Strategy
#     print(f"Parsing '{strategy_col}' column...")
#     df = df.dropna(subset=[strategy_col])
#     try:
#         df[strategy_col] = df[strategy_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
#     except (ValueError, SyntaxError) as e:
#         print(f"Error parsing dictionary: {e}")
#         return
#
#     # Expand dictionary
#     strategy_df = pd.json_normalize(df[strategy_col])
#     df = pd.concat([df.drop(columns=[strategy_col]), strategy_df], axis=1)
#
#     # 3. Validation & Cleaning
#     expanded_alpha_col = alpha_key
#     if expanded_alpha_col not in df.columns:
#         print(f"Error: Key '{alpha_key}' not found.")
#         return
#
#     # Drop NaNs
#     df = df.dropna(subset=[metric_col, expanded_alpha_col, temp_col])
#
#     # Force sortable types (float)
#     df[expanded_alpha_col] = pd.to_numeric(df[expanded_alpha_col], errors='coerce')
#     df = df.dropna(subset=[expanded_alpha_col])
#
#     # 4. Global X-Axis Mapping (The Magic Step)
#     # Get ALL unique alpha values sorted numerically
#     all_unique_alphas = sorted(df[expanded_alpha_col].unique())
#
#     # Create a map: {0.1: 0, 0.5: 1, 128: 2, ...}
#     alpha_to_index = {val: i for i, val in enumerate(all_unique_alphas)}
#
#     # 5. Group & Aggregate
#     grouped = df.groupby([temp_col, expanded_alpha_col])[metric_col].agg(['mean', 'sem', 'count']).reset_index()
#     unique_temps = sorted(grouped[temp_col].unique())
#
#     if not unique_temps:
#         print("No valid data found to plot.")
#         return
#
#     plt.figure(figsize=(10, 6))
#
#     # 6. Plotting Loop
#     for temp in unique_temps:
#         subset = grouped[grouped[temp_col] == temp]
#         subset = subset.sort_values(by=expanded_alpha_col)
#
#         # Transform Alpha Values to Indices (0, 1, 2...)
#         x_indices = [alpha_to_index[a] for a in subset[expanded_alpha_col]]
#
#         # Plot using INDICES as x, but Labels as label
#         container = plt.errorbar(
#             x=x_indices,
#             y=subset['mean'],
#             yerr=subset['sem'],
#             fmt='o-',
#             capsize=5,
#             linewidth=2,
#             label=f'Temp: {temp}'
#         )
#
#         # Get line color
#         line_color = container[0].get_color()
#
#         # Add Baseline (Horizontal Line)
#         # We look for alpha == 0
#         baseline_row = subset[subset[expanded_alpha_col] == 0]
#
#         if not baseline_row.empty:
#             baseline_val = baseline_row['mean'].values[0]
#
#             plt.axhline(
#                 y=baseline_val,
#                 color=line_color,
#                 linestyle='--',
#                 alpha=1,
#                 linewidth=1.5
#             )
#
#     # 7. Formatting
#     # plt.title(f'Effect of {alpha_key} (Equally Spaced)', fontsize=14)
#     plt.ylabel(f'{metric_col} (Mean ± SE)', fontsize=12)
#     plt.xlabel(alpha_key, fontsize=12)
#
#     # --- CRITICAL: Set Custom X-Ticks ---
#     # Ticks at 0, 1, 2...
#     # Labels are the actual Alpha values
#     plt.xticks(
#         ticks=range(len(all_unique_alphas)),
#         labels=all_unique_alphas
#     )
#     # ------------------------------------
#
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(title="Temperature")
#     plt.tight_layout()
#
#     # Save & Show
#     plt.savefig('gsm_div.png')
#     plt.show()
#
#
# # --- Usage Example ---
# if __name__ == "__main__":
#     plot_alpha_equally_spaced(
#         csv_path='gsm8k.csv',
#         # metric_col='pass_at_16',
#         metric_col='avg_diversity',
#         strategy_col='strategy',
#         temp_col='temperature',
#         alpha_key='alpha'
#     )

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np


def plot_alpha_equally_spaced(
        csv_path,
        metric_col,
        strategy_col,
        temp_col,
        alpha_key='alpha',
        strategy_name_key='name',
        allowed_strategies=["baseline", "batched_orth"]
):
    """
    Plots alpha vs metric where the X-axis is CATEGORICAL (equally spaced).
    Incorporates strict strategy filtering and premium aesthetic formatting.
    """

    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # 2. Parse Strategy
    print(f"Parsing '{strategy_col}' column...")
    df = df.dropna(subset=[strategy_col])
    try:
        df[strategy_col] = df[strategy_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing dictionary: {e}")
        return

    # Expand dictionary into columns
    strategy_df = pd.json_normalize(df[strategy_col])

    # FIX 1: Force the new dataframe to use the exact same row indices as the main dataframe
    strategy_df.index = df.index

    # FIX 2: Identify and remove any overlapping columns from the main dataframe
    # (so the dictionary's values take precedence and we avoid duplicate labels)
    overlap = set(df.columns).intersection(set(strategy_df.columns))
    overlap.discard(strategy_col)  # Keep strategy_col for the drop below
    if overlap:
        df = df.drop(columns=list(overlap))

    # Concatenate safely
    df = pd.concat([df.drop(columns=[strategy_col]), strategy_df], axis=1)
    # 3. Validation & Filtering
    if strategy_name_key in df.columns and allowed_strategies:
        initial_count = len(df)
        df = df[df[strategy_name_key].isin(allowed_strategies)]
        print(f"Filtered runs from {initial_count} to {len(df)} using allowed strategies: {allowed_strategies}")

    expanded_alpha_col = alpha_key
    if expanded_alpha_col not in df.columns:
        print(f"Error: Key '{alpha_key}' not found in the expanded dictionary columns.")
        return

    # Drop NaNs for essential plotting columns
    df = df.dropna(subset=[metric_col, expanded_alpha_col, temp_col])

    # Force sortable types (float)
    df[expanded_alpha_col] = pd.to_numeric(df[expanded_alpha_col], errors='coerce')
    df = df.dropna(subset=[expanded_alpha_col])

    # 4. Global X-Axis Mapping (Equally Spaced)
    all_unique_alphas = sorted(df[expanded_alpha_col].unique())
    alpha_to_index = {val: i for i, val in enumerate(all_unique_alphas)}

    # 5. Group & Aggregate
    grouped = df.groupby([temp_col, expanded_alpha_col])[metric_col].agg(['mean', 'sem', 'count']).reset_index()
    unique_temps = sorted(grouped[temp_col].unique())

    if not unique_temps:
        print("No valid data found to plot after filtering.")
        return

    # 6. Plotting Setup (Aesthetics)
    plt.figure(figsize=(9, 6.5))
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Use the premium mako palette to match the Pareto plot style
    palette = sns.color_palette("mako", n_colors=len(unique_temps))

    # 7. Plotting Loop
    for i, temp in enumerate(unique_temps):
        subset = grouped[grouped[temp_col] == temp].sort_values(by=expanded_alpha_col)

        # Transform Alpha Values to Indices (0, 1, 2...)
        x_indices = [alpha_to_index[a] for a in subset[expanded_alpha_col]]

        # Plot error bars and lines with premium formatting
        container = plt.errorbar(
            x=x_indices,
            y=subset['mean'],
            # yerr=subset['sem'],
            fmt='o-',
            capsize=0,  # Remove caps for a cleaner look
            markersize=9,  # Large dots
            markeredgecolor='white',  # White border to make dots pop
            markeredgewidth=1.5,
            linewidth=2.5,  # Thicker, bolder lines
            color=palette[i],
            label=f'$\\theta$ = {temp}',
            zorder=3  # Draw above the grid
        )

        line_color = container[0].get_color()

        # Add Baseline (Horizontal Line for alpha == 0)
        baseline_row = subset[subset[expanded_alpha_col] == 0]
        if not baseline_row.empty:
            baseline_val = baseline_row['mean'].values[0]
            plt.axhline(
                y=baseline_val,
                color=line_color,
                linestyle='--',
                alpha=0.6,  # Slightly faded baseline
                linewidth=1.5,
                zorder=1
            )

    # 8. Dynamic Epsilon Y-Axis Zoom
    y_min = (grouped['mean'] - grouped['sem']).min()
    y_max = (grouped['mean'] + grouped['sem']).max()
    eps_y = max((y_max - y_min) * 0.08, 0.01)

    plt.ylim(y_min - eps_y, y_max + eps_y)

    # 9. Formatting & Clean Up
    plt.ylabel(f'Average Batch Diversity', fontsize=12, fontweight='500')
    plt.xlabel('Repulsion ($\\alpha$)', fontsize=12, fontweight='500')

    # Custom X-Ticks to map indices back to Alpha labels
    plt.xticks(
        ticks=range(len(all_unique_alphas)),
        labels=[f"{a:g}" for a in all_unique_alphas]  # :g removes trailing zeros
    )

    # Soften the grid and remove harsh bounding box
    plt.grid(True, linestyle='-', alpha=0.4)
    sns.despine(left=True, bottom=True)

    plt.legend(
        title="Temperature ($\\theta$)",
        title_fontsize='11',
        fontsize='10',
        loc='best',
        frameon=True,
        shadow=False,
        borderpad=1
    )

    plt.tight_layout()
    plt.savefig('he_div_aesthetic.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# --- Usage Example ---
if __name__ == "__main__":
    plot_alpha_equally_spaced(
        csv_path='human_eval.csv',
        metric_col='avg_diversity',  # Or 'pass_at_16'
        strategy_col='strategy',
        temp_col='temperature',
        alpha_key='alpha',
        strategy_name_key='name',  # The key in the dict representing the strategy name
        allowed_strategies=['baseline', 'batched_orth']
    )