import pandas as pd
import ast
import numpy as np


def load_and_process_csv(csv_path, dataset_name, strategy_col, alpha_key):
    """Helper function to load a CSV, parse the strategy JSON, and format columns."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return pd.DataFrame()

    df = df.dropna(subset=[strategy_col]).reset_index(drop=True)

    # Parse the strategy column
    try:
        df[strategy_col] = df[strategy_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    except (ValueError, SyntaxError):
        print(f"Error parsing dictionary column in {csv_path}.")
        return pd.DataFrame()

    strategy_df = pd.json_normalize(df[strategy_col].tolist())

    # Extract Strategy Name
    if 'name' in strategy_df.columns:
        df['strategy_name'] = strategy_df['name'].astype(str).str.lower()
    elif 'type' in strategy_df.columns:
        df['strategy_name'] = strategy_df['type'].astype(str).str.lower()
    else:
        df['strategy_name'] = 'unknown'

    # Extract Alpha
    if alpha_key in strategy_df.columns:
        df['alpha_val'] = pd.to_numeric(strategy_df[alpha_key], errors='coerce')
    elif alpha_key in df.columns:
        df['alpha_val'] = pd.to_numeric(df[alpha_key], errors='coerce')
    else:
        df['alpha_val'] = np.nan

    # For baseline, safely force alpha to 0 for table layout/sorting purposes
    df.loc[df['strategy_name'] == 'baseline', 'alpha_val'] = 0.0

    df['dataset'] = dataset_name
    return df


def generate_combined_latex_table(gsm_csv, he_csv, strategy_col, temp_col,
                                  target_strategy='orthogonal_projection',
                                  dpp_strategy='dpp',
                                  alpha_key='alpha'):
    """
    Generates a combined LaTeX table for GSM8K and HumanEval.
    Includes Baseline, dpp Strategy, and the Target Strategy for pass@16.
    """
    df_gsm = load_and_process_csv(gsm_csv, 'GSM8K', strategy_col, alpha_key)
    df_he = load_and_process_csv(he_csv, 'HumanEval', strategy_col, alpha_key)

    df_all = pd.concat([df_gsm, df_he], ignore_index=True)

    if df_all.empty:
        print("Error: No valid data loaded from either CSV.")
        return

    for col in ['pass_at_16', temp_col]:
        if col not in df_all.columns:
            print(f"Error: '{col}' column not found in data.")
            return
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    df_all = df_all.dropna(subset=['pass_at_16', temp_col, 'alpha_val'])

    df_all['pass_at_16'] = df_all['pass_at_16'] * 100.0

    target_strategy = target_strategy.lower()
    dpp_strategy = dpp_strategy.lower()

    valid_strategies = ['baseline', dpp_strategy, target_strategy]
    df_all = df_all[df_all['strategy_name'].isin(valid_strategies)]

    unique_temps = sorted(df_all[temp_col].unique())
    num_cols = len(unique_temps)

    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\setlength{\tabcolsep}{8pt} % Adjust column padding")

    col_def = "l" + ("c" * num_cols)
    print(r"\begin{tabular}{" + col_def + "}")
    print(r"\toprule")

    print(r"\multirow{2}{*}{\textbf{Step Size} ($\alpha$)} & \multicolumn{" + str(
        num_cols) + r"}{c}{\textbf{Temperature} ($\theta$)} \\")
    print(r"\cmidrule(lr){2-" + str(num_cols + 1) + "}")

    header_temps = " & ".join([f"{t}" for t in unique_temps])
    print(f"  & {header_temps} \\\\")

    # 4. Process Each Dataset
    datasets = ['GSM8K', 'HumanEval']

    for i, dataset in enumerate(datasets):
        dataset_df = df_all[df_all['dataset'] == dataset]
        if dataset_df.empty:
            continue

        grouped = dataset_df.groupby(['strategy_name', 'alpha_val', temp_col])['pass_at_16'].agg(
            ['mean', 'std', 'count']).reset_index()
        grouped['sem'] = (grouped['std'] / np.sqrt(grouped['count'])).fillna(0.0)

        mean_pivot = grouped.pivot(index=['strategy_name', 'alpha_val'], columns=temp_col, values='mean')
        sem_pivot = grouped.pivot(index=['strategy_name', 'alpha_val'], columns=temp_col, values='sem')

        best_means = mean_pivot.max()

        if i > 0:
            print(r"\midrule")
        print(r"\multicolumn{6}{c}{\textsc{" + dataset + r"}} \\")
        print(r"\midrule")

        # Helper to print sections cleanly
        def print_section(strat_name, section_title):
            if strat_name not in mean_pivot.index.get_level_values(0):
                return False

            print(r"\multicolumn{6}{l}{\textit{" + section_title + r"}} \\")
            strat_alphas = sorted(mean_pivot.loc[strat_name].index)

            for alpha in strat_alphas:
                # Format 0 for baseline, and drop trailing decimals for others (e.g., 2.0 -> 2)
                row_label = "0 " if strat_name == 'baseline' else f"{alpha:g}  "
                row_str = f"{row_label:<3}"

                for t in unique_temps:
                    if t not in mean_pivot.columns or pd.isna(mean_pivot.loc[(strat_name, alpha), t]):
                        row_str += " & -"
                    else:
                        m = mean_pivot.loc[(strat_name, alpha), t]
                        s = sem_pivot.loc[(strat_name, alpha), t]

                        inner_str = f"{m:.1f} \\pm {s:.1f}"

                        # Bold if it matches the best mean (using a tiny epsilon for float comparison safety)
                        if m >= best_means[t] - 1e-5:
                            inner_str = f"\\mathbf{{{inner_str}}}"

                        row_str += f" & ${inner_str}$"

                print(row_str + r" \\")
            return True

        # Print sections in requested order
        has_baseline = print_section('baseline', 'Baseline (Standard LLaDA)')

        if print_section(dpp_strategy, 'dpp Strategy'):
            if has_baseline:
                print(r"\addlinespace")  # Optional styling gap

        if print_section(target_strategy, r'\sysname{} (Our Approach)'):
            print(r"\addlinespace")

    # 5. Print Table Footer
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(
        r"\caption{Pass@16 results (mean $\pm$ SE) for GSM8K and HumanEval over various temperature ($\theta$) and repulsion step sizes ($\alpha$). \textbf{Bold} values indicate the best result for each $\theta$ within that dataset.}")
    print(r"\label{tab:combined_results_final}")
    print(r"\end{table*}")


# --- Run ---
if __name__ == "__main__":
    generate_combined_latex_table(
        gsm_csv='gsm8k.csv',
        he_csv='humaneval.csv',
        strategy_col='strategy',
        temp_col='temperature',
        target_strategy='odd',
        dpp_strategy='dpp',
        alpha_key='alpha'
    )