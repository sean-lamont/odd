import wandb
import pandas as pd
import json
import os
import ast
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# --- 1. Core Logic (W&B Extraction) ---
def fetch_wandb_data(entity, project, table_key_name="evaluation_results_table"):
    """
    Downloads data from W&B and builds the dictionary.
    """
    api = wandb.Api()
    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project}")

    # Master dictionary: Key = task_id, Value = List of config dicts
    problem_map = defaultdict(list)

    print(f"Found {len(runs)} runs. Starting download & processing...")

    for run in tqdm(runs, desc="Processing Runs"):
        # -- A. Extract Config --
        config = run.config

        # Handle 'strategy' if it's a string representation of a dict
        strategy = config.get('strategy', {})
        if isinstance(strategy, str):
            try:
                strategy = ast.literal_eval(strategy)
            except:
                strategy = {}

        alpha = strategy.get('alpha', None)
        temperature = config.get('temperature', None)

        if alpha is None or temperature is None:
            continue

        # -- B. Find the Table Artifact --
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

        # -- C. Download & Read Table --
        try:
            # Download artifact directory
            table_dir = target_artifact.download()

            # Find the .table.json file
            json_path = None
            for f in os.listdir(table_dir):
                if f.endswith(".table.json"):
                    json_path = os.path.join(table_dir, f)
                    break

            if not json_path:
                continue

            with open(json_path) as f:
                table_data = json.load(f)

            # Create DataFrame
            df = pd.DataFrame(table_data['data'], columns=table_data['columns'])

            # -- D. Filter & Count Successes --
            # Ensure we count only where passed == True
            # passed_df = df[df['passed'] == True]
            # gsm8k
            passed_df = df[df['is_correct'] == True]

            if passed_df.empty:
                continue

            # Count successes per task_id
            success_counts = passed_df.groupby('question').size()
            # success_counts = passed_df.groupby('task_id').size()

            for task_id, count in success_counts.items():
                problem_map[task_id].append({
                    'alpha': float(alpha),  # Convert to native float
                    'temperature': float(temperature),
                    'pass_count': int(count)  # Convert numpy int to native int
                })

        except Exception as e:
            print(f"Skipping run {run.name} due to error: {e}")
            continue

    return dict(problem_map)


# --- 2. Save & Load Functions ---
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


# --- 3. Main Execution Block ---
if __name__ == "__main__":

    ENTITY = "tactic-zero"
    PROJECT = "gsm8k"
    TABLE_NAME = "results_table"  # The name visible in W&B UI
    FILENAME = "gsm8k_table.json"

    # Option A: Load from file if it exists
    if os.path.exists(FILENAME):
        print("Found cached data. Loading...")
        results = load_results(FILENAME)

    # Option B: Fetch fresh from W&B if no file exists
    else:
        print("No cache found. Fetching from W&B...")
        results = fetch_wandb_data(ENTITY, PROJECT, TABLE_NAME)

        # Save for next time
        save_results(results, FILENAME)

    # --- Example Usage of Loaded Data ---
    # Print first 2 problems and their successful configs
    count = 0
    for problem, configs in results.items():
        print(f"\nTask: {problem}")
        for cfg in configs:
            print(f"  - Temp: {cfg['temperature']}, Alpha: {cfg['alpha']}, Passed: {cfg['pass_count']}/16")

        count += 1
        if count >= 2: break