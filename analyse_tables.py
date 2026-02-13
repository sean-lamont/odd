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

        # Extract Params
        alpha = strategy.get('alpha', None)

        # --- NEW: Extract Strategy Name ---
        # We assume the key is 'name' or 'type' at the same level as 'alpha'
        strategy_name = strategy.get('name', strategy.get('type', 'unknown'))
        # ----------------------------------

        temperature = config.get('temperature', None)

        # Skip runs that don't have the required hyperparameters
        if alpha is None or temperature is None:
            continue

        # -- B. Find the Table Artifact --
        target_artifact = None
        try:
            # Look for the specific table artifact logged in this run
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
            # Download artifact directory (caches locally)
            table_dir = target_artifact.download()

            # Find the .table.json file inside the artifact dir
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

            # -- D. Sum Successes (Don't Filter) --

            # 1. Ensure 'is_correct' is treated as a number (True=1, False=0)
            col_name = 'is_correct' if 'is_correct' in df.columns else 'passed'

            if col_name not in df.columns:
                print(f"Skipping run {run.name}: Column '{col_name}' not found.")
                continue

            # Convert boolean/string to integer 1/0
            df[col_name] = df[col_name].astype(int)

            # 2. Group by Question and SUM the correct answers
            group_key = 'question' if 'question' in df.columns else 'task_id'

            # This series has Index=Question, Value=Sum of Correct
            success_counts = df.groupby(group_key)[col_name].sum()

            # 3. Store Data
            for task_id, count in success_counts.items():
                problem_map[task_id].append({
                    'strategy': str(strategy_name),  # <--- NEW FIELD
                    'alpha': float(alpha),
                    'temperature': float(temperature),
                    'pass_count': int(count)
                })

        except Exception as e:
            # print(f"Skipping run {run.name} due to error: {e}")
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

    # --- CONFIGURATION ---
    ENTITY = "tactic-zero"  # Update with your W&B Entity
    PROJECT = "gsm8k"  # Update with your W&B Project Name
    TABLE_NAME = "results_table"  # The specific table name to look for
    FILENAME = "gsm_table_v2.json"

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

    # --- Verification & Sample Output ---
    count = 0
    print("\n--- Sample Entries ---")
    for problem, configs in results.items():
        print(f"\nTask: {problem}")
        for cfg in configs:
            # Updated print statement to show Strategy Name
            print(
                f"  - Strategy: {cfg['strategy']}, Temp: {cfg['temperature']}, Alpha: {cfg['alpha']}, Passed: {cfg['pass_count']}/16")

        count += 1
        if count >= 2: break