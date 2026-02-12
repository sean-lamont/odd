import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np


def plot_alpha_equally_spaced(csv_path, metric_col, strategy_col, temp_col, alpha_key='alpha'):
    """
    Plots alpha vs metric where the X-axis is CATEGORICAL (equally spaced).
    Even if alphas are [2, 4, 128], they will be plotted at x=[0, 1, 2].
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

    # Expand dictionary
    strategy_df = pd.json_normalize(df[strategy_col])
    df = pd.concat([df.drop(columns=[strategy_col]), strategy_df], axis=1)

    # 3. Validation & Cleaning
    expanded_alpha_col = alpha_key
    if expanded_alpha_col not in df.columns:
        print(f"Error: Key '{alpha_key}' not found.")
        return

    # Drop NaNs
    df = df.dropna(subset=[metric_col, expanded_alpha_col, temp_col])

    # Force sortable types (float)
    df[expanded_alpha_col] = pd.to_numeric(df[expanded_alpha_col], errors='coerce')
    df = df.dropna(subset=[expanded_alpha_col])

    # 4. Global X-Axis Mapping (The Magic Step)
    # Get ALL unique alpha values sorted numerically
    all_unique_alphas = sorted(df[expanded_alpha_col].unique())

    # Create a map: {0.1: 0, 0.5: 1, 128: 2, ...}
    alpha_to_index = {val: i for i, val in enumerate(all_unique_alphas)}

    # 5. Group & Aggregate
    grouped = df.groupby([temp_col, expanded_alpha_col])[metric_col].agg(['mean', 'sem', 'count']).reset_index()
    unique_temps = sorted(grouped[temp_col].unique())

    if not unique_temps:
        print("No valid data found to plot.")
        return

    plt.figure(figsize=(10, 6))

    # 6. Plotting Loop
    for temp in unique_temps:
        subset = grouped[grouped[temp_col] == temp]
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
        # We look for alpha == 0
        baseline_row = subset[subset[expanded_alpha_col] == 0]

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
    # plt.title(f'Effect of {alpha_key} (Equally Spaced)', fontsize=14)
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
    plt.savefig('gsm_div.png')
    plt.show()


# --- Usage Example ---
if __name__ == "__main__":
    plot_alpha_equally_spaced(
        csv_path='gsm8k.csv',
        # metric_col='pass_at_16',
        metric_col='avg_diversity',
        strategy_col='strategy',
        temp_col='temperature',
        alpha_key='alpha'
    )