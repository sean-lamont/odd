import wandb
import pandas as pd
import json
import os
import ast
import numpy as np
from collections import defaultdict
from tqdm import tqdm

ENTITY = "YOUR_WANDB_ENTITY"  # Update this
NEW_PROJECTS = [
    # "dream_gsm8k_eval", "dream_humaneval_eval",
    # "gsm8k_generalisation", "humaneval_generalisation",
    # "gsm8k_feature_ablation", "humaneval_feature_ablation"
] # all wandb projects from one or more of sweep_dream.py, feature_ablation.py and generalisation.py


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def fetch_and_save_project_data(entity, project):
    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project}")
    except Exception as e:
        print(f"Could not access {project}: {e}")
        return

    print(f"\n--- Processing {project} ({len(runs)} runs) ---")

    summary_list, config_list, name_list = [], [], []
    problem_map = defaultdict(list)

    for run in tqdm(runs, desc="Downloading Summaries & Tables"):
        # 1. Summaries and Configs for CSV
        summary_list.append(run.summary._json_dict)
        config_list.append(run.config)
        name_list.append(run.name)

        # 2. Results Table for JSON
        strategy = run.config.get('strategy', {})
        if isinstance(strategy, str):
            try:
                strategy = ast.literal_eval(strategy)
            except:
                strategy = {}

        strat_name = strategy.get('name', strategy.get('type', run.config.get('strategy', 'unknown')))
        alpha = strategy.get('alpha', run.config.get('alpha', None))
        temp = run.config.get('temperature', None)

        target_artifact = next((art for art in run.logged_artifacts() if "results_table" in art.name), None)
        if target_artifact and alpha is not None and temp is not None:
            try:
                table_dir = target_artifact.download()
                json_path = next(
                    (os.path.join(table_dir, f) for f in os.listdir(table_dir) if f.endswith(".table.json")), None)
                if json_path:
                    with open(json_path) as f:
                        table_data = json.load(f)
                    df = pd.DataFrame(table_data['data'], columns=table_data['columns'])
                    col_name = 'is_correct' if 'is_correct' in df.columns else 'passed'
                    group_key = 'question' if 'question' in df.columns else 'task_id'

                    df[col_name] = df[col_name].astype(int)
                    success_counts = df.groupby(group_key)[col_name].sum()

                    for task_id, count in success_counts.items():
                        problem_map[task_id].append({
                            'strategy': str(strat_name),
                            'alpha': float(alpha),
                            'temperature': float(temp),
                            'pass_count': int(count)
                        })
            except Exception as e:
                pass  # Skip runs with missing table data

    # Save CSV
    if summary_list:
        summary_df = pd.DataFrame.from_records(summary_list)
        config_df = pd.DataFrame.from_records(config_list)
        all_data = pd.concat([pd.DataFrame({'name': name_list}), config_df, summary_df], axis=1)
        all_data.to_csv(f"{project}.csv", index=False)
        print(f"Saved {project}.csv")

    # Save JSON
    if problem_map:
        with open(f"{project}_table.json", 'w') as f:
            json.dump(problem_map, f, cls=NumpyEncoder, indent=4)
        print(f"Saved {project}_table.json")


if __name__ == "__main__":
    for proj in NEW_PROJECTS:
        fetch_and_save_project_data(ENTITY, proj)