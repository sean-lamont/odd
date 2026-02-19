# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
#
# def plot_pass1_vs_pass16_corrected(json_path):
#     # 1. Load Data
#     try:
#         with open(json_path, 'r') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: File {json_path} not found.")
#         return
#
#     # 2. Flatten JSON into a list of runs
#     rows = []
#
#     # Filter constraints
#     ALLOWED_STRATEGIES = ["baseline", "batched_orth"]
#
#     for problem_id, runs_list in data.items():
#         for run in runs_list:
#             # --- FILTER LOGIC ---
#             # We assume the JSON run object has a 'strategy' key.
#             # If missing, we default to 'batched_orth' to be safe with older data,
#             # unless you have mixed data without labels, in which case stricter handling is needed.
#             run_strategy = run.get('strategy', 'batched_orth')
#
#             if run_strategy not in ALLOWED_STRATEGIES:
#                 continue
#             # --------------------
#
#             alpha = run.get('alpha')
#             temp = run.get('temperature')
#
#             # Force integer conversion to handle "0" strings correctly
#             try:
#                 count = int(run.get('pass_count', 0))
#             except (ValueError, TypeError):
#                 count = 0
#
#             # Calculate metrics for THIS specific run
#             # Pass@1: Fraction of samples correct in this batch
#             p1_run = count / 16.0
#
#             # Pass@16: Binary indicator (1 if solved in this batch, 0 if not)
#             p16_run = 1.0 if count > 0 else 0.0
#
#             rows.append({
#                 'problem_id': problem_id,  # Crucial for Stage 1 grouping
#                 'alpha': float(alpha),
#                 'temperature': float(temp),
#                 'p1_run': p1_run,
#                 'p16_run': p16_run,
#                 'raw_count': count
#             })
#
#     if not rows:
#         print("No data found matching the allowed strategies.")
#         return
#
#     df = pd.DataFrame(rows)
#
#     # --- DEBUG SECTION ---
#     total_runs = len(df)
#     failed_runs = len(df[df['raw_count'] == 0])
#     print("=" * 40)
#     print(f"DEBUG DATA CHECK")
#     print(f"Strategies Included: {ALLOWED_STRATEGIES}")
#     print(f"Total Runs Processed: {total_runs}")
#     print(f"Runs with pass_count == 0: {failed_runs}")
#     print(f"Runs with pass_count > 0:  {total_runs - failed_runs}")
#     if failed_runs == 0:
#         print("\nWARNING: No failed runs found! This explains why Pass@16 is 1.0.")
#         print("Check if your JSON file only contains successful attempts.")
#     print("=" * 40 + "\n")
#     # ---------------------
#
#     # 3. STAGE 1 AGGREGATION: Average over runs per Problem
#     problem_level = df.groupby(['problem_id', 'alpha', 'temperature'])[['p1_run', 'p16_run']].mean().reset_index()
#
#     # Rename for clarity
#     problem_level = problem_level.rename(columns={'p1_run': 'p1_problem', 'p16_run': 'p16_problem'})
#
#     # 4. STAGE 2 AGGREGATION: Average over Problems
#     final_df = problem_level.groupby(['alpha', 'temperature'])[['p1_problem', 'p16_problem']].mean().reset_index()
#
#     # 5. Plotting
#     plt.figure(figsize=(10, 7))
#     sns.set_style("whitegrid")
#
#     unique_alphas = sorted(final_df['alpha'].unique())
#     colors = plt.cm.viridis(np.linspace(0, 1, len(unique_alphas)))
#
#     for i, alpha in enumerate(unique_alphas):
#         subset = final_df[final_df['alpha'] == alpha]
#         subset = subset.sort_values(by='temperature')
#
#         plt.plot(
#             subset['p1_problem'],
#             subset['p16_problem'],
#             marker='o',
#             label=f'Alpha={alpha}',
#             linewidth=2,
#             color=colors[i]
#         )
#
#         # Annotate min/max temp
#         for _, row in subset.iterrows():
#             if row['temperature'] == subset['temperature'].min() or row['temperature'] == subset['temperature'].max():
#                 plt.text(
#                     row['p1_problem'],
#                     row['p16_problem'] + 0.01,
#                     f"T={row['temperature']}",
#                     fontsize=8,
#                     color=colors[i],
#                     fontweight='bold'
#                 )
#
#     plt.title('Pass@1 vs Pass@16 Trade-off (Filtered)', fontsize=14)
#     plt.xlabel('Pass@1 (Average Efficiency)', fontsize=12)
#     plt.ylabel('Pass@16 (Probability of Solution)', fontsize=12)
#     plt.legend(title='Alpha')
#     plt.grid(True, linestyle='--', alpha=0.6)
#
#     # Fix axes to 0-1 range unless data is weird
#     plt.xlim(left=0, right=max(final_df['p1_problem'].max() * 1.1, 0.1))
#     plt.ylim(bottom=0, top=1.05)
#
#     plt.tight_layout()
#     plt.savefig('pareto_batched.pdf')
#     plt.show()
#
#
# # --- Usage ---
# if __name__ == "__main__":
#     plot_pass1_vs_pass16_corrected('he_table_batched.json')


