import wandb
import pandas as pd
import json
from tqdm import tqdm


def export_project_data(project_path, output_csv):
    """
    Fetches run data, configs, summaries, and GPU metrics for a given W&B project.
    """
    api = wandb.Api()
    runs = api.runs(project_path)

    summary_list = []
    config_list = []
    name_list = []

    for run in tqdm(runs, desc=f"Processing {project_path}"):
        summary_list.append(run.summary._json_dict)
        config_list.append(run.config)
        name_list.append(run.name)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list})

    all_data = pd.concat([name_df, config_df, summary_df], axis=1)

    print(f"\nPreview of {project_path}:")
    print(all_data.head())
    all_data.to_csv(output_csv, index=False)
    print(f"Successfully saved {len(all_data)} runs to {output_csv}\n")


if __name__ == "__main__":
    export_project_data("tactic-zero/gsm8k", "gsm8k.csv")
    export_project_data("tactic-zero/humaneval", "humaneval.csv")
