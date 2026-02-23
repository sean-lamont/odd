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
    # gpu_list = []

    for run in tqdm(runs, desc=f"Processing {project_path}"):
        # run.summary contains the final output values
        summary_list.append(run.summary._json_dict)

        # run.config contains the hyperparameters
        config_list.append(run.config)

        # run.name is the human-readable name
        name_list.append(run.name)

    #     gpu_data = {}
    #
    #     # 1. Fetch GPU Hardware Model from Metadata
    #     try:
    #         # Download the hidden metadata file
    #         meta_file = run.file("wandb-metadata.json").download(replace=True)
    #         with open(meta_file.name, "r") as f:
    #             metadata = json.load(f)
    #             gpu_data["gpu_model"] = metadata.get("gpu")
    #             print(metadata.get("gpu"))
    #     except Exception:
    #         pass
    #
    #     # 2. Fetch GPU Utilization from Events Stream
    #     try:
    #         # Fetch the entire events stream safely (without 'keys' parameter)
    #         sys_metrics = run.history(stream="events", pandas=True)
    #         if not sys_metrics.empty:
    #             if "system.gpu.0.memory" in sys_metrics.columns:
    #                 gpu_data["gpu_0_mem_avg"] = sys_metrics["system.gpu.0.memory"].mean()
    #             if "system.gpu.1.memory" in sys_metrics.columns:
    #                 gpu_data["gpu_1_mem_avg"] = sys_metrics["system.gpu.1.memory"].mean()
    #     except Exception:
    #         pass
    #
    #     gpu_list.append(gpu_data)
    #
    # # Convert lists to DataFrames
    # gpu_df = pd.DataFrame.from_records(gpu_list)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list})

    # Concatenate all data horizontally
    # all_data = pd.concat([name_df, config_df, summary_df, gpu_df], axis=1)
    all_data = pd.concat([name_df, config_df, summary_df], axis=1)

    # Save and display
    print(f"\nPreview of {project_path}:")
    print(all_data.head())
    all_data.to_csv(output_csv, index=False)
    print(f"Successfully saved {len(all_data)} runs to {output_csv}\n")


# ==========================================
# Execute for both projects
# ==========================================
if __name__ == "__main__":
    # export_project_data("tactic-zero/gsm8k", "gsm8k.csv")
    export_project_data("tactic-zero/humaneval", "humaneval.csv")
