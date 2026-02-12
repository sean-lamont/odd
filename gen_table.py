import pandas as pd
import ast
import numpy as np

def generate_latex_table_final(csv_path, metric_col, strategy_col, temp_col, alpha_key='alpha'):
    """
    Generates a LaTeX table with:
    - Theta (temperature) in headers.
    - Underlining for results > baseline.
    - Bolding for best result in column.
    - Extra line after baseline row.
    """
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # 2. Parse Strategy Column
    df = df.dropna(subset=[strategy_col])
    try:
        df[strategy_col] = df[strategy_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    except (ValueError, SyntaxError):
        print("Error parsing dictionary column.")
        return

    # Expand strategy column
    strategy_df = pd.json_normalize(df[strategy_col])
    df = pd.concat([df.drop(columns=[strategy_col]), strategy_df], axis=1)

    # 3. Clean & Convert Data
    expanded_alpha_col = alpha_key
    
    # Convert Cols to Numeric
    df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
    df[expanded_alpha_col] = pd.to_numeric(df[expanded_alpha_col], errors='coerce')
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')

    # Drop NaNs
    df = df.dropna(subset=[metric_col, expanded_alpha_col, temp_col])

    # --- Convert Metric to Percentage ---
    df[metric_col] = df[metric_col] * 100
    # ------------------------------------

    # 4. Aggregate Stats
    grouped = df.groupby([expanded_alpha_col, temp_col])[metric_col].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate Standard Error (SEM)
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['sem'] = grouped['sem'].fillna(0.0)

    # 5. Pivot for Table
    mean_pivot = grouped.pivot(index=expanded_alpha_col, columns=temp_col, values='mean')
    sem_pivot = grouped.pivot(index=expanded_alpha_col, columns=temp_col, values='sem')

    # 6. Generate LaTeX
    unique_temps = sorted(df[temp_col].unique())
    unique_alphas = sorted(df[expanded_alpha_col].unique())
    baseline_alpha = 0.0

    print(r"\begin{table*}[h]")
    print(r"\centering")
    
    col_def = "l" + ("c" * len(unique_temps))
    print(r"\begin{tabular}{" + col_def + "}")
    print(r"\hline")
    
    # Header: Use \theta instead of T
    header = r"$\alpha$"
    for t in unique_temps:
        header += f" & $\\theta={t}$"
    print(header + r" \\")
    print(r"\hline")

    # Rows
    for alpha in unique_alphas:
        # Label Baseline specially
        if alpha == baseline_alpha:
            row_label = f"{alpha} (Baseline)"
        else:
            row_label = f"{alpha}"
        
        row_str = row_label
        
        for temp in unique_temps:
            val_mean = mean_pivot.loc[alpha, temp]
            val_sem = sem_pivot.loc[alpha, temp]
            
            if pd.isna(val_mean):
                row_str += " & -"
                continue

            # 1. Formatting Base: Mean +/- SE
            # We construct the inner string first
            inner_str = f"{val_mean:.1f} \\pm {val_sem:.1f}"

            # 2. Logic: Best in Column (Bold)
            best_alpha_for_col = mean_pivot[temp].idxmax()
            if alpha == best_alpha_for_col:
                inner_str = f"\\textbf{{{inner_str}}}"

            # 3. Logic: Better than Baseline (Underline)
            # Only apply if not the baseline itself
            # if alpha != baseline_alpha:
            #     try:
            #         base_mean = mean_pivot.loc[baseline_alpha, temp]
            #         if val_mean > base_mean:
            #             inner_str = f"\\underline{{{inner_str}}}"
            #     except KeyError:
            #         pass # No baseline for this temp

            # 4. Wrap in Math Mode $$
            row_str += f" & ${inner_str}$"

            # --- COMMENTED OUT GAIN CALCULATION ---
            # if alpha != baseline_alpha:
            #     base_mean = mean_pivot.loc[baseline_alpha, temp]
            #     gain = ((val_mean - base_mean) / base_mean) * 100
            #     # ... add to string ...
            # ---------------------------------------

        print(row_str + r" \\")
        
        # Add extra line after baseline
        if alpha == baseline_alpha:
            print(r"\hline")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Accuracy (\%) with different $\alpha$ values. $\theta$ represents temperature. \textbf{Bold} indicates best in column. \underline{Underline} indicates improvement over baseline.}")
    print(r"\label{tab:results_final}")
    print(r"\end{table*}")

# --- Run ---
if __name__ == "__main__":
    generate_latex_table_final(
        csv_path='gsm8k.csv',
        metric_col='pass_at_16',
        strategy_col='strategy',
        temp_col='temperature',
        alpha_key='alpha'
    )