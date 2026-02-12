import pandas as pd
import matplotlib.pyplot as plt
import ast


def plot_alpha_by_temp_parsed(csv_path, metric_col, strategy_col, temp_col, alpha_key='alpha'):
    """
    Loads W&B export data, parses a dictionary column (strategy),
    and plots the results.

    Args:
        csv_path (str): Path to CSV.
        metric_col (str): The y-axis metric (e.g., 'val_accuracy').
        strategy_col (str): The column containing the dict string (e.g., 'config.strategy').
        temp_col (str): The column for temperature (e.g., 'config.temperature').
        alpha_key (str): The specific key INSIDE the strategy dict to use as x-axis.
    """

    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # 2. Parse the Dictionary Column
    print(f"Parsing '{strategy_col}' column...")

    # Ensure we drop rows where the strategy column itself is empty/NaN before parsing
    df = df.dropna(subset=[strategy_col])

    # Convert string "{'alpha': 0.1}" -> dict {'alpha': 0.1}
    try:
        # We use apply to safely evaluate the string literal
        df[strategy_col] = df[strategy_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing dictionary: {e}")
        return

    # Expand the dictionary into separate columns
    # This creates new columns like 'strategy.alpha', 'strategy.beta', etc.
    strategy_df = pd.json_normalize(df[strategy_col])

    # Reset index to ensure alignment during concatenation
    df = df.reset_index(drop=True)
    strategy_df = strategy_df.reset_index(drop=True)

    # Merge the new columns back into the main dataframe
    df = pd.concat([df, strategy_df], axis=1)

    # 3. Define the new column name for Alpha
    # json_normalize usually keeps keys as is, so if key was 'alpha', col is 'alpha'
    # BUT if you nested it, verify the output. Usually it's just the key name.
    expanded_alpha_col = alpha_key

    # Check if the expansion worked
    if expanded_alpha_col not in df.columns:
        print(f"Error: Could not find key '{alpha_key}' inside the strategy dictionary.")
        print(f"Available keys found: {strategy_df.columns.tolist()}")
        return

    # 4. Filter NaNs (Strategy keys, Temperature, or Metric)
    # Now we use the NEW expanded column name
    df = df.dropna(subset=[metric_col, expanded_alpha_col, temp_col])

    # 5. Group & Aggregate
    grouped = df.groupby([temp_col, expanded_alpha_col])[metric_col].agg(['mean', 'sem', 'count']).reset_index()

    # --- NEW SECTION: Print Run Counts ---
    print("\n" + "=" * 40)
    print(f"Run Counts per Configuration")
    print("=" * 40)

    # Pivot for a cleaner table view (Rows=Alpha, Cols=Temp)
    count_table = grouped.pivot(index=expanded_alpha_col, columns=temp_col, values='count')
    print(count_table.fillna(0).astype(int))
    print("=" * 40 + "\n")
    # -------------------------------------

    unique_temps = sorted(grouped[temp_col].unique())

    if not unique_temps:
        print("No valid data found to plot.")
        return

    # 6. Plotting Loop
    for temp in unique_temps:
        subset = grouped[grouped[temp_col] == temp]
        subset = subset.sort_values(by=expanded_alpha_col)

        plt.figure(figsize=(8, 5))

        plt.errorbar(
            x=subset[expanded_alpha_col],
            y=subset['mean'],
            yerr=subset['sem'],
            fmt='o-',
            capsize=5,
            linewidth=2,
            label=f'Temp: {temp}'
        )

        plt.title(f'Effect of {alpha_key} (Temp = {temp})', fontsize=14)
        plt.xlabel(alpha_key, fontsize=12)
        plt.ylabel(f'{metric_col} (Mean ± SE)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'gsm8k_alpha_{temp}.png')

# --- Usage Example ---
plot_alpha_by_temp_parsed(
    csv_path='gsm8k.csv',
    metric_col='pass_at_16',
    strategy_col='strategy',  # The column with "{...}"
    temp_col='temperature',
    alpha_key='alpha'                # The key inside the dict you want on X-axis
)