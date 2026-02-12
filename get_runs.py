import pandas as pd
from tqdm import tqdm
import wandb

api = wandb.Api()
runs = api.runs("tactic-zero/humaneval")

summary_list = []
config_list = []
name_list = []

for run in tqdm(runs):
    # run.summary contains the final output values (e.g., final accuracy)
    summary_list.append(run.summary._json_dict)

    # run.config contains the hyperparameters (e.g., learning_rate)
    config_list.append(run.config)

    # run.name is the human-readable name
    name_list.append(run.name)

summary_df = pd.DataFrame.from_records(summary_list)
config_df = pd.DataFrame.from_records(config_list)
name_df = pd.DataFrame({'name': name_list})

all_data = pd.concat([name_df, config_df, summary_df], axis=1)
print (all_data.head())
all_data.to_csv("human_eval.csv")

api = wandb.Api()
runs = api.runs("tactic-zero/gsm8k")

summary_list = []
config_list = []
name_list = []

for run in tqdm(runs):
    # run.summary contains the final output values (e.g., final accuracy)
    summary_list.append(run.summary._json_dict)

    # run.config contains the hyperparameters (e.g., learning_rate)
    config_list.append(run.config)

    # run.name is the human-readable name
    name_list.append(run.name)

summary_df = pd.DataFrame.from_records(summary_list)
config_df = pd.DataFrame.from_records(config_list)
name_df = pd.DataFrame({'name': name_list})

all_data = pd.concat([name_df, config_df, summary_df], axis=1)
print (all_data.head())
all_data.to_csv("gsm8k.csv")
