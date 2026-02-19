import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt


def plot_batched_orth_equally_spaced(csv_path, metric_col, strategy_col, temp_col, target_strategy='batched_orth',
                                     alpha_key='alpha'):
    """
    Plots alpha vs metric where the X-axis is CATEGORICAL (equally spaced).
    Separates the 'baseline' strategy and plots it as a horizontal line per temperature.
    """

    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # 2. Parse Strategy
    print(f"Parsing '{strategy_col}' column...")
    df = df.dropna(subset=[strategy_col]).reset_index(drop=True)
    try:
        df[strategy_col] = df[strategy_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing dictionary: {e}")
        return

    # Expand dictionary
    strategy_df = pd.json_normalize(df[strategy_col].tolist())

    # Extract strategy name
    if 'name' in strategy_df.columns:
        df['strategy_name'] = strategy_df['name'].astype(str).str.lower()
    elif 'type' in strategy_df.columns:
        df['strategy_name'] = strategy_df['type'].astype(str).str.lower()
    else:
        df['strategy_name'] = 'unknown'

    # Extract alpha
    expanded_alpha_col = alpha_key
    if expanded_alpha_col in strategy_df.columns:
        df[expanded_alpha_col] = pd.to_numeric(strategy_df[expanded_alpha_col], errors='coerce')
    elif expanded_alpha_col in df.columns:
        df[expanded_alpha_col] = pd.to_numeric(df[expanded_alpha_col], errors='coerce')

    # 3. Validation & Cleaning
    df = df.dropna(subset=[metric_col, temp_col])
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

    # Separate target strategy and baseline
    df_target = df[df['strategy_name'] == target_strategy].dropna(subset=[expanded_alpha_col])
    df_baseline = df[df['strategy_name'] == 'baseline']

    # 4. Global X-Axis Mapping (The Magic Step)
    # Get ALL unique alpha values sorted numerically for the target strategy
    all_unique_alphas = sorted(df_target[expanded_alpha_col].unique())

    if not all_unique_alphas:
        print(f"Error: No valid alpha values found for strategy '{target_strategy}'.")
        return

    # Create a map: {0.1: 0, 0.5: 1, 128: 2, ...}
    alpha_to_index = {val: i for i, val in enumerate(all_unique_alphas)}

    # 5. Group & Aggregate
    # Target is grouped by temp AND alpha
    grouped_target = df_target.groupby([temp_col, expanded_alpha_col])[metric_col].agg(
        ['mean', 'sem', 'count']).reset_index()

    # Baseline is grouped ONLY by temp
    grouped_baseline = df_baseline.groupby([temp_col])[metric_col].agg(['mean']).reset_index()

    unique_temps = sorted(grouped_target[temp_col].unique())

    if not unique_temps:
        print("No valid data found to plot.")
        return

    plt.figure(figsize=(10, 6))

    # 6. Plotting Loop
    for temp in unique_temps:
        subset = grouped_target[grouped_target[temp_col] == temp]
        subset = subset.sort_values(by=expanded_alpha_col)

        # Transform Alpha Values to Indices (0, 1, 2...)
        x_indices = [alpha_to_index[a] for a in subset[expanded_alpha_col]]

        # Plot using INDICES as x, but Labels as label
        container = plt.errorbar(
            x=x_indices,
            y=subset['mean'],
            yerr=subset['sem'],
            fmt='o-',
            capsize=5,
            linewidth=2,
            label=f'Temp: {temp}'
        )

        # Get line color
        line_color = container[0].get_color()

        # Add Baseline (Horizontal Line)
        # Instead of alpha == 0, we look up the baseline mean for this specific temperature
        baseline_row = grouped_baseline[grouped_baseline[temp_col] == temp]

        if not baseline_row.empty:
            baseline_val = baseline_row['mean'].values[0]

            plt.axhline(
                y=baseline_val,
                color=line_color,
                linestyle='--',
                alpha=1,
                linewidth=1.5
            )

    # 7. Formatting
    plt.title(f'Effect of {alpha_key} on {target_strategy}', fontsize=14)
    plt.ylabel(f'{metric_col} (Mean ± SE)', fontsize=12)
    plt.xlabel(alpha_key, fontsize=12)

    # --- CRITICAL: Set Custom X-Ticks ---
    # Ticks at 0, 1, 2...
    # Labels are the actual Alpha values
    plt.xticks(
        ticks=range(len(all_unique_alphas)),
        labels=all_unique_alphas
    )
    # ------------------------------------

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Temperature")
    plt.tight_layout()

    # Save & Show
    plt.savefig('he_div_fixed.png')
    plt.show()


# --- Usage Example ---
if __name__ == "__main__":
    plot_batched_orth_equally_spaced(
        csv_path='human_eval.csv',
        metric_col='avg_diversity',
        strategy_col='strategy',
        temp_col='temperature',
        target_strategy='batched_orth',
        alpha_key='alpha'
    )