import ast
import re

import numpy as np
import pandas as pd


def extract_alg(row):
    if 'alg' in row and not pd.isna(row['alg']):
        val = str(row['alg']).lower()
        if 'maskgit_plus' in val or 'origin' in val:
            return val
    name = str(row.get('name', '')).lower()
    if 'maskgit_plus' in name:
        return 'maskgit_plus'
    if 'origin' in name:
        return 'origin'
    return 'origin'


def extract_alpha(row):
    try:
        if isinstance(row['strategy'], str) and row['strategy'].strip().startswith('{'):
            s_dict = ast.literal_eval(row['strategy'])
            if 'alpha' in s_dict:
                return float(s_dict['alpha'])
    except:
        pass
    if 'alpha' in row and not pd.isna(row['alpha']):
        return float(row['alpha'])
    return np.nan


def extract_strategy_safe(strategy_val):
    try:
        if isinstance(strategy_val, str) and strategy_val.strip().startswith('{'):
            s_dict = ast.literal_eval(strategy_val)
            if 'name' in s_dict:
                return s_dict['name'].lower()
    except:
        pass
    return str(strategy_val).lower()


def extract_len(name_str):
    match = re.search(r'len(\d+)', str(name_str))
    return int(match.group(1)) if match else np.nan


print("--- DATA PROCESSING STARTED ---")


# --- 1. DREAM MODEL TABLES ---
def generate_dream_tables():
    print("\n% ================= DREAM MODEL TABLES =================")
    try:
        df_gsm = pd.read_csv('dream_gsm8k_eval.csv')
        df_he = pd.read_csv('dream_humaneval_eval.csv')
    except Exception as e:
        print(f"% Missing files: {e}")
        return

    for df, name in [(df_gsm, 'GSM8K'), (df_he, 'HumanEval')]:
        df['dataset'] = name
        df['alg'] = df.apply(extract_alg, axis=1)
        df['strategy_name'] = df.apply(lambda row: extract_strategy_safe(row['strategy']), axis=1)
        df['strategy_name'] = df['strategy_name'].apply(
            lambda x: 'odd' if 'odd' in x else 'baseline' if 'baseline' in x else x)
        df['alpha_val'] = df.apply(extract_alpha, axis=1)
        df.loc[df['strategy_name'] == 'baseline', 'alpha_val'] = 0.0
        df['pass_at_16'] = df['pass_at_16'] * 100.0

    df_all = pd.concat([df_gsm, df_he], ignore_index=True)
    df_all = df_all.dropna(subset=['pass_at_16', 'temperature', 'alpha_val'])

    # for alg_type in ['origin', 'maskgit_plus']:
    for alg_type in ['maskgit_plus']:
        df_alg = df_all[df_all['alg'] == alg_type]
        if df_alg.empty: continue

        unique_temps = sorted(df_alg['temperature'].unique())
        num_cols = len(unique_temps)
        alg_label = "Random Unmasking (Origin)" if alg_type == 'origin' else "Highest Confidence Unmasking (MaskGIT+)"

        print(f"\n% --- DREAM: {alg_label.upper()} ---")
        print(r"\begin{table*}[t]")
        print(r"\centering")
        print(r"\setlength{\tabcolsep}{8pt}")
        col_def = "l" + ("c" * num_cols)
        print(r"\begin{tabular}{" + col_def + "}")
        print(r"\toprule")
        print(r"\multirow{2}{*}{\textbf{Step Size} ($\alpha$)} & \multicolumn{" + str(
            num_cols) + r"}{c}{\textbf{Temperature} ($\theta$)} \\")
        print(r"\cmidrule(lr){2-" + str(num_cols + 1) + "}")
        print("  & " + " & ".join([f"{t}" for t in unique_temps]) + r" \\")

        for dataset in ['GSM8K', 'HumanEval']:
            dataset_df = df_alg[df_alg['dataset'] == dataset]
            if dataset_df.empty: continue

            grouped = dataset_df.groupby(['strategy_name', 'alpha_val', 'temperature'])['pass_at_16'].agg(
                ['mean', 'std', 'count']).reset_index()
            # Calculate SEM, allowing NaNs for count <= 1
            grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])

            mean_pivot = grouped.pivot(index=['strategy_name', 'alpha_val'], columns='temperature', values='mean')
            sem_pivot = grouped.pivot(index=['strategy_name', 'alpha_val'], columns='temperature', values='sem')
            best_means = mean_pivot.max()

            if dataset == 'HumanEval': print(r"\midrule")
            print(r"\multicolumn{6}{c}{\textsc{" + dataset + r"}} \\")
            print(r"\midrule")

            def print_section(strat_name, section_title):
                if strat_name not in mean_pivot.index.get_level_values(0): return
                print(r"\multicolumn{6}{l}{\textit{" + section_title + r"}} \\")
                strat_alphas = sorted(mean_pivot.loc[strat_name].index)
                for alpha in strat_alphas:
                    row_label = "0 " if strat_name == 'baseline' else f"{alpha:g}  "
                    row_str = f"{row_label:<3}"
                    for t in unique_temps:
                        if t not in mean_pivot.columns or pd.isna(mean_pivot.loc[(strat_name, alpha), t]):
                            row_str += " & -"
                        else:
                            m = mean_pivot.loc[(strat_name, alpha), t]
                            s = sem_pivot.loc[(strat_name, alpha), t]

                            # Format based on whether SEM is valid
                            if pd.isna(s) or np.isnan(s):
                                inner_str = f"{m:.1f}"
                            else:
                                inner_str = f"{m:.1f} \\pm {s:.1f}"

                            if m >= best_means[t] - 1e-5: inner_str = f"\\mathbf{{{inner_str}}}"
                            row_str += f" & ${inner_str}$"
                    print(row_str + r" \\")

            print_section('baseline', 'Baseline (DREAM)')
            print(r"\addlinespace")
            print_section('odd', r'\sysname{} (Our Approach)')

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(f"\\caption{{Pass@16 results for DREAM using \\textbf{{{alg_label}}}.}}")
        print(f"\\label{{tab:dream_{alg_type}}}")
        print(r"\end{table*}")


# --- 2. FEATURE ABLATION TABLES ---
def generate_feature_tables():
    print("\n% ================= FEATURE ABLATION TABLES =================")
    try:
        df_gsm = pd.read_csv('gsm8k_feature_ablation.csv')
        df_he = pd.read_csv('humaneval_feature_ablation.csv')
    except Exception as e:
        print(f"% Missing files: {e}")
        return

    for df, name in [(df_gsm, 'GSM8K'), (df_he, 'HumanEval')]:
        df['dataset'] = name
        df['pass_at_16'] = df['pass_at_16'] * 100.0

    df_all = pd.concat([df_gsm, df_he], ignore_index=True)

    print(f"\n% --- FEATURE ABLATION ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lllc}")
    print(r"\toprule")
    print(r"\textbf{Target} & \textbf{Pool} & \textbf{Top-$k$} & \textbf{Pass@16 (\%)} \\")
    print(r"\midrule")

    for ds in ['GSM8K', 'HumanEval']:
        dataset_df = df[df['dataset'] == ds]
        if dataset_df.empty: continue

        agg = dataset_df.groupby(['target', 'pool', 'top_k'])['pass_at_16'].agg(
            ['mean', 'std', 'count']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'])
        agg = agg.sort_values(by='mean', ascending=False)

        print(r"\multicolumn{4}{c}{\textsc{" + ds + r"}} \\")
        print(r"\midrule")

        best_mean = agg['mean'].max()
        for _, row in agg.iterrows():
            target = "Logits" if row['target'] == 'logits' else "Embeddings"
            pool = "Max" if row['pool'] == 'max' else "Positional"
            top_k = "All" if row['top_k'] == 0 else str(int(row['top_k']))

            if pd.isna(row['sem']) or np.isnan(row['sem']):
                val_str = f"{row['mean']:.1f}"
            else:
                val_str = f"{row['mean']:.1f} \\pm {row['sem']:.1f}"

            if row['mean'] >= best_mean - 1e-5: val_str = f"\\mathbf{{{val_str}}}"
            print(f"{target} & {pool} & {top_k} & ${val_str}$ \\\\")

        if ds == 'GSM8K': print(r"\midrule")

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(f"\\caption{{Feature Ablation}}")
        print(f"\\label{{tab:feature_ablation}}")
        print(r"\end{table}")


def parse_strategy_dict(val):
    """Fallback parser if the name is buried inside a strategy dict."""
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
    if isinstance(val, dict):
        return val.get('name', str(val))
    return str(val)


def generate_generalisation_table():
    csv_paths = {
        'HumanEval': 'humaneval_generalisation.csv',
        'GSM8K': 'gsm8k_generalisation.csv'
    }

    all_data = []

    for dataset_name, csv_path in csv_paths.items():
        try:
            df = pd.read_csv(csv_path)
            df['dataset'] = dataset_name

            # 1. Prioritize the explicit 'name' column if it exists
            if 'name' in df.columns:
                df['raw_run_name'] = df['name'].astype(str)
            elif 'strategy' in df.columns:
                df['raw_run_name'] = df['strategy'].apply(parse_strategy_dict)
            else:
                print(f"% Error: Neither 'name' nor 'strategy' column found in {csv_path}")
                continue

            # 2. Extract gen_length directly from the string (e.g., 'len128' -> 128)
            df['extracted_len'] = df['raw_run_name'].apply(
                lambda x: int(re.search(r'len(\d+)', str(x).lower()).group(1)) if re.search(r'len(\d+)',
                                                                                            str(x).lower()) else None
            )

            # 3. Apply the extracted length
            df['gen_length'] = df['extracted_len']

            # 4. Assign approach based on the run name string
            df['strat_name'] = df['raw_run_name'].apply(
                lambda x: 'odd' if 'odd' in str(x).lower() else 'baseline'
            )

            all_data.append(df)

        except FileNotFoundError:
            print(f"% Error: File {csv_path} not found.")
            continue

    if not all_data:
        print("% Error: No valid data found in provided files.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Drop rows missing critical data
    initial_len = len(combined_df)
    combined_df = combined_df.dropna(subset=['steps', 'gen_length', 'pass_at_16'])

    if combined_df.empty:
        print(f"% Error: All {initial_len} rows were dropped. Check if regex failed to find 'lenXXX' in the names.")
        return

    # Convert pass_at_16 to percentage
    combined_df['pass_at_16_pct'] = combined_df['pass_at_16'] * 100

    # Calculate Mean and Standard Error (SEM)
    agg_df = combined_df.groupby(['dataset', 'steps', 'gen_length', 'strat_name'])['pass_at_16_pct'].agg(
        ['mean', 'sem']).reset_index()
    agg_df['sem'] = agg_df['sem'].fillna(0.0)

    # Sort columns and rows
    gen_lengths = sorted(agg_df['gen_length'].unique())
    steps_list = sorted(agg_df['steps'].unique())


    print("\n% ================= GENERALISATION TABLE =================")
    # --- LaTeX Table Generation ---
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\setlength{\\tabcolsep}{6pt}")
    print(
        "\\caption{Generalisation of Pass@16 (\\%) performance across varying generation lengths and diffusion steps. We report the mean and standard error across multiple runs. Bold values indicate the best performance for a given configuration.}")
    print("\\label{tab:generalisation_robustness}")

    cols_format = "ll l " + "c" * len(gen_lengths)
    print(f"\\begin{{tabular}}{{{cols_format}}}")
    print("\\toprule")

    length_headers = " & ".join([f"\\textbf{{{int(gl)} Tokens}}" for gl in gen_lengths])
    print(
        f"\\multirow{{2}}{{*}}{{\\textbf{{Dataset}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Steps}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Approach}}}} & \\multicolumn{{{len(gen_lengths)}}}{{c}}{{\\textbf{{Generation Length}}}} \\\\")
    print(f"\\cmidrule(lr){{4-{3 + len(gen_lengths)}}}")
    print(f" & & & {length_headers} \\\\ \\midrule")

    datasets = combined_df['dataset'].unique()

    for i, dataset in enumerate(datasets):
        ds_data = agg_df[agg_df['dataset'] == dataset]
        valid_steps = sorted(ds_data['steps'].unique())

        for j, step in enumerate(valid_steps):
            step_data = ds_data[ds_data['steps'] == step]

            baseline_row_vals = []
            odd_row_vals = []

            for gl in gen_lengths:
                b_cell = step_data[(step_data['strat_name'] == 'baseline') & (step_data['gen_length'] == gl)]
                o_cell = step_data[(step_data['strat_name'] == 'odd') & (step_data['gen_length'] == gl)]

                b_val = b_cell['mean'].values[0] if not b_cell.empty else None
                b_sem = b_cell['sem'].values[0] if not b_cell.empty else None

                o_val = o_cell['mean'].values[0] if not o_cell.empty else None
                o_sem = o_cell['sem'].values[0] if not o_cell.empty else None

                b_str = f"${b_val:.1f} \\pm {b_sem:.1f}$" if b_val is not None else "-"
                o_str = f"${o_val:.1f} \\pm {o_sem:.1f}$" if o_val is not None else "-"

                # Bold the higher mean value
                if b_val is not None and o_val is not None:
                    if b_val > o_val:
                        b_str = f"\\textbf{{{b_str}}}"
                    elif o_val > b_val:
                        o_str = f"\\textbf{{{o_str}}}"
                    else:
                        b_str = f"\\textbf{{{b_str}}}"
                        o_str = f"\\textbf{{{o_str}}}"
                elif b_val is not None and o_val is None:
                    b_str = f"\\textbf{{{b_str}}}"
                elif o_val is not None and b_val is None:
                    o_str = f"\\textbf{{{o_str}}}"

                baseline_row_vals.append(b_str)
                odd_row_vals.append(o_str)

            ds_str = f"\\multirow{{{len(valid_steps) * 2}}}{{*}}{{\\textsc{{{dataset}}}}}" if j == 0 else ""
            step_str = f"\\multirow{{2}}{{*}}{{{int(step)}}}"

            print(f"{ds_str:<25} & {step_str:<10} & \\textit{{Baseline}} & {' & '.join(baseline_row_vals)} \\\\")
            print(f"{'':<25} & {'':<10} & \\textbf{{\\sysname{{}}}} & {' & '.join(odd_row_vals)} \\\\")

            if j < len(valid_steps) - 1:
                print(f"\\cmidrule(lr){{2-{3 + len(gen_lengths)}}}")

        if i < len(datasets) - 1:
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    generate_dream_tables()
    generate_feature_tables()
    generate_generalisation_table()
