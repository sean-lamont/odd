import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np


def plot_pass_k_distinct(csv_path, strategy_col, temp_col, alpha_key='alpha'):
    """
    Plots Pass@k scaling with DISTINCT colors and markers for each Alpha configuration.
    """

    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # 2. Parse Strategy
    print(f"Parsing '{strategy_col}'...")
    df = df.dropna(subset=[strategy_col])
    try:
        df[strategy_col] = df[strategy_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    except Exception as e:
        print(f"Error parsing dictionary: {e}")
        return

    strategy_df = pd.json_normalize(df[strategy_col])
    df = pd.concat([df.drop(columns=[strategy_col]), strategy_df], axis=1)

    # 3. Clean & Filter
    # Ensure columns exist
    expanded_alpha_col = alpha_key
    pass_at_cols = [c for c in df.columns if c.startswith("pass_at_")]

    if not pass_at_cols:
        print("Error: No 'pass_at_' columns found.")
        return

    df = df.dropna(subset=[temp_col, expanded_alpha_col] + pass_at_cols)
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    df[expanded_alpha_col] = pd.to_numeric(df[expanded_alpha_col], errors='coerce')

    # 4. Melt Data (Wide -> Long)
    melted_df = df.melt(
        id_vars=[temp_col, expanded_alpha_col],
        value_vars=pass_at_cols,
        var_name='k_label',
        value_name='Pass Rate'
    )

    # Extract integer k
    melted_df['k'] = melted_df['k_label'].str.extract(r'(\d+)').astype(int)

    # Sort
    melted_df = melted_df.sort_values(by=[temp_col, expanded_alpha_col, 'k'])

    # 5. Plotting
    sns.set_style("whitegrid")

    # Define Plot
    g = sns.relplot(
        data=melted_df,
        x="k",
        y="Pass Rate",
        kind="line",
        errorbar=None,
        # --- DISTINCT VISUALS ---
        col=temp_col,  # Subplot per Temperature
        hue=expanded_alpha_col,  # Different Color per Alpha
        style=expanded_alpha_col,  # Different Marker per Alpha
        markers=True,  # Show the markers
        dashes=False,  # Keep all lines solid (easier to read with markers)
        palette="tab10",  # High contrast categorical palette
        # ------------------------

        height=5,
        aspect=1.2,
        linewidth=2,
        markersize=8  # Make markers slightly larger to be visible
    )

    # 6. Formatting
    g.set_axis_labels("Attempts ($k$)", "Pass Rate")
    g.legend.set_title(f"Alpha")

    # Set titles
    g.set_titles("Temperature = {col_name}")

    # Fix X-axis ticks to integers
    max_k = melted_df['k'].max()
    for ax in g.axes.flatten():
        ax.set_xticks([1, 5, 10, max_k])
        ax.set_xlim(0.5, max_k + 0.5)

    plt.subplots_adjust(top=0.85)
    g.fig.suptitle('Pass@$k$ Scaling Performance', fontsize=16)

    # Save & Show
    plt.savefig('gsm_passk.png')
    plt.show()


# --- Run ---
if __name__ == "__main__":
    plot_pass_k_distinct(
        csv_path='gsm8k.csv',
        strategy_col='strategy',
        temp_col='temperature',
        alpha_key='alpha'
    )