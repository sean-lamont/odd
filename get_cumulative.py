import json
from collections import defaultdict


def calculate_solved_combined(json_path):
    # 1. Load Data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_path} not found.")
        return

    # 2. Track Data
    # Detailed: Key = (alpha, temp)
    stats_detailed = defaultdict(lambda: {'solved_ids': set(), 'total_trials': 0})

    # Summary: Key = alpha (Aggregating across temperatures)
    stats_summary = defaultdict(lambda: {'solved_ids': set(), 'total_trials': 0})

    total_problems = set()

    # Filter constraints
    ALLOWED_STRATEGIES = ["baseline", "orthogonal_projection"]

    for problem_id, runs_list in data.items():
        total_problems.add(problem_id)

        for run in runs_list:
            # --- FILTER LOGIC ---
            run_strategy = run.get('strategy', 'orthogonal_projection')
            if run_strategy not in ALLOWED_STRATEGIES:
                continue
            # --------------------

            # Normalize keys
            try:
                alpha = float(run.get('alpha'))
            except (ValueError, TypeError):
                alpha = run.get('alpha')

            try:
                temp = float(run.get('temperature'))
            except (ValueError, TypeError):
                temp = run.get('temperature')

            try:
                count = int(run.get('pass_count', 0))
            except (ValueError, TypeError):
                count = 0

            # --- UPDATE DETAILED STATS (Alpha, Temp) ---
            key_det = (alpha, temp)
            stats_detailed[key_det]['total_trials'] += 1
            if count > 0:
                stats_detailed[key_det]['solved_ids'].add(problem_id)

            # --- UPDATE SUMMARY STATS (Alpha only) ---
            # This aggregates across all temperatures for this alpha
            stats_summary[alpha]['total_trials'] += 1
            if count > 0:
                stats_summary[alpha]['solved_ids'].add(problem_id)

    # 3. Print Results

    # --- TABLE 1: DETAILED BREAKDOWN ---
    print(f"Total Unique Problems: {len(total_problems)}")
    print("\n" + "=" * 90)
    print("DETAILED BREAKDOWN (By Alpha & Temperature)")
    print("-" * 90)
    print(
        f"{'Alpha':<8} | {'Temp':<6} | {'Solved':<8} | {'Coverage %':<12} | {'Total Trials':<15} | {'Success/Trial':<15}")
    print("-" * 90)

    sorted_keys_det = sorted(stats_detailed.keys(), key=lambda x: (x[0], x[1]))

    if not sorted_keys_det:
        print("No data found for the allowed strategies.")
    else:
        for alpha, temp in sorted_keys_det:
            d = stats_detailed[(alpha, temp)]
            solved_count = len(d['solved_ids'])
            trials = d['total_trials']

            coverage = (solved_count / len(total_problems)) * 100 if total_problems else 0.0
            efficiency = (solved_count / trials) * 100 if trials > 0 else 0.0

            print(
                f"{alpha:<8} | {temp:<6} | {solved_count:<8} | {coverage:<11.2f}% | {trials:<15} | {efficiency:<14.2f}%")

    # --- TABLE 2: AGGREGATED SUMMARY ---
    print("\n" + "=" * 90)
    print("AGGREGATED SUMMARY (Cumulative by Alpha across all Temperatures)")
    print("-" * 90)
    print(
        f"{'Alpha':<8} | {'Solved (Union)':<15} | {'Coverage %':<12} | {'Total Trials':<15} | {'Global Efficiency':<18}")
    print("-" * 90)

    sorted_keys_sum = sorted(stats_summary.keys())

    for alpha in sorted_keys_sum:
        d = stats_summary[alpha]
        # 'Solved (Union)' is the count of unique problems solved by ANY temperature for this alpha
        solved_count = len(d['solved_ids'])
        trials = d['total_trials']

        coverage = (solved_count / len(total_problems)) * 100 if total_problems else 0.0
        efficiency = (solved_count / trials) * 100 if trials > 0 else 0.0

        print(f"{alpha:<8} | {solved_count:<15} | {coverage:<11.2f}% | {trials:<15} | {efficiency:<17.2f}%")

    print("=" * 90)


if __name__ == "__main__":
    calculate_solved_combined('gsm_table_v2.json')