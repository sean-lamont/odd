import wandb
import pandas as pd
import json
import os
import ast
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def fetch_wandb_data(entity, project, table_key_name="results_table"):
    """
    Downloads data from W&B and builds the dictionary.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    problem_map = defaultdict(list)

    print(f"Found {len(runs)} runs. Starting download & processing...")

    for run in tqdm(runs, desc="Processing Runs"):
        config = run.config

        strategy = config.get('strategy', {})
        if isinstance(strategy, str):
            try:
                strategy = ast.literal_eval(strategy)
            except:
                strategy = {}

        alpha = strategy.get('alpha', None)

        strategy_name = strategy.get('name', strategy.get('type', 'unknown'))
        temperature = config.get('temperature', None)

        if alpha is None or temperature is None:
            continue

        target_artifact = None
        try:
            for artifact in run.logged_artifacts():
                if table_key_name in artifact.name:
                    target_artifact = artifact
                    break
        except Exception:
            continue

        if not target_artifact:
            continue

        try:
            table_dir = target_artifact.download()
            json_path = None
            for f in os.listdir(table_dir):
                if f.endswith(".table.json"):
                    json_path = os.path.join(table_dir, f)
                    break

            if not json_path:
                continue

            with open(json_path) as f:
                table_data = json.load(f)

            df = pd.DataFrame(table_data['data'], columns=table_data['columns'])

            col_name = 'is_correct' if 'is_correct' in df.columns else 'passed'

            if col_name not in df.columns:
                print(f"Skipping run {run.name}: Column '{col_name}' not found.")
                continue

            df[col_name] = df[col_name].astype(int)

            group_key = 'question' if 'question' in df.columns else 'task_id'

            success_counts = df.groupby(group_key)[col_name].sum()

            for task_id, count in success_counts.items():
                problem_map[task_id].append({
                    'strategy': str(strategy_name),  # <--- NEW FIELD
                    'alpha': float(alpha),
                    'temperature': float(temperature),
                    'pass_count': int(count)
                })

        except Exception as e:
            print(f"Skipping run {run.name} due to error: {e}")
            continue

    return dict(problem_map)


class NumpyEncoder(json.JSONEncoder):
    """ Special handler to convert Numpy types to Python types for JSON saving """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_results(data, filename="wandb_results.json"):
    """ Saves the dictionary to a JSON file. """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=4)
        print(f"Successfully saved data to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")


def load_results(filename="wandb_results.json"):
    """ Loads the dictionary from a JSON file. """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {filename}")
        return data
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

if __name__ == "__main__":

    # --- CONFIGURATION ---
    ENTITY = "sean-a-lamont"  # Update with your W&B Entity
    PROJECT = "odd_gsm8k"  # Update with your W&B Project Name
    TABLE_NAME = "results_table"  # The specific table name to look for
    FILENAME = "gsm8k_table.json"

    if os.path.exists(FILENAME):
        print("Found cached data. Loading...")
        results = load_results(FILENAME)
    else:
        print("No cache found. Fetching from W&B...")
        results = fetch_wandb_data(ENTITY, PROJECT, TABLE_NAME)
        save_results(results, FILENAME)

    count = 0
    print("\n--- Sample Entries ---")
    for problem, configs in results.items():
        print(f"\nTask: {problem}")
        for cfg in configs:
            print(
                f"  - Strategy: {cfg['strategy']}, Temp: {cfg['temperature']}, Alpha: {cfg['alpha']}, Passed: {cfg['pass_count']}/16")

        count += 1
        if count >= 2: break



    # --- CONFIGURATION ---
    ENTITY = "sean-a-lamont"  # Update with your W&B Entity
    PROJECT = "odd_humaneval"  # Update with your W&B Project Name
    TABLE_NAME = "results_table"  # The specific table name to look for
    FILENAME = "humaneval_table.json"

    if os.path.exists(FILENAME):
        print("Found cached data. Loading...")
        results = load_results(FILENAME)
    else:
        print("No cache found. Fetching from W&B...")
        results = fetch_wandb_data(ENTITY, PROJECT, TABLE_NAME)
        save_results(results, FILENAME)

    count = 0
    print("\n--- Sample Entries ---")
    for problem, configs in results.items():
        print(f"\nTask: {problem}")
        for cfg in configs:
            print(
                f"  - Strategy: {cfg['strategy']}, Temp: {cfg['temperature']}, Alpha: {cfg['alpha']}, Passed: {cfg['pass_count']}/16")

        count += 1
        if count >= 2: break




