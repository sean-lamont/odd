import os
import sys
import json
import random
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Import your modular core
from odd_gen import load_model
from feature_extractor import FeatureExtractor
from strategies import get_strategy
from app_generator import AppGenerator

# HumanEval local import matching sweep_human_eval.py
sys.path.append(os.path.join(os.getcwd(), "human-eval"))
try:
    from human_eval.data import read_problems
except ImportError:
    print("Error: Could not import human_eval. Make sure the human-eval directory exists.")
    sys.exit(1)


def load_config():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")
    return cfg


def get_gsm8k_prompts(num_samples=20):
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    random.seed(42)
    indices = random.sample(range(min(200, len(dataset))), num_samples)

    prompts = []
    for idx, i in enumerate(indices):
        q = dataset[i]['question']
        formatted_prompt = f"Question: {q}\nLet's think step by step.\nAnswer:"
        prompts.append({"id": f"Problem_{idx + 1}", "prompt": formatted_prompt})
    return prompts


def get_humaneval_prompts(num_samples=20):
    print("Loading HumanEval dataset...")
    problems_dict = read_problems()
    problem_list = list(problems_dict.values())

    random.seed(42)
    indices = random.sample(range(min(164, len(problem_list))), num_samples)

    prompts = []
    for idx, i in enumerate(indices):
        problem = problem_list[i]
        safe_id = problem['task_id'].replace("/", "_")
        prompts.append({"id": safe_id, "prompt": problem['prompt']})
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate ODD Demo Grids")
    parser.add_argument("--strategy", type=str, choices=["baseline", "odd", "dpp", "all"], default="all",
                        help="Which strategy to run")
    parser.add_argument("--dataset", type=str, choices=["GSM8K", "HumanEval", "all"], default="all",
                        help="Which dataset to run")
    parser.add_argument("--temp", type=float, choices=[0.0, 1.0, 2.0, -1.0], default=-1.0,
                        help="Specific temperature to run (-1 for all)")
    args = parser.parse_args()

    out_dir = "demo_histories"
    os.makedirs(out_dir, exist_ok=True)

    print("Initializing model...")
    cfg = load_config()
    model, tokenizer, embedding_matrix, mask_token_id = load_model(cfg)

    # Grid Search Parameters
    batch_size = 4
    steps = 32
    alphas = [8.0, 16.0, 128.0]

    temps_to_run = [0.0, 1.0, 2.0] if args.temp == -1.0 else [args.temp]
    strats_to_run = ["baseline", "batched_orth", "joint"] if args.strategy == "all" else [args.strategy]

    datasets_to_run = {}
    if args.dataset in ["GSM8K", "all"]:
        datasets_to_run["GSM8K"] = {"prompts": get_gsm8k_prompts(20), "gen_len": 64}
    if args.dataset in ["HumanEval", "all"]:
        datasets_to_run["HumanEval"] = {"prompts": get_humaneval_prompts(20), "gen_len": 64}

    print("Initializing Extractor...")
    extractor = FeatureExtractor(embedding_matrix=embedding_matrix, kernel_target="logits", pooling_method="max")

    for ds_name, ds_info in datasets_to_run.items():
        print(f"\n================ Starting Dataset: {ds_name} ================")
        prompts = ds_info["prompts"]
        gen_len = ds_info["gen_len"]

        for p_dict in tqdm(prompts, desc=f"Processing {ds_name}"):
            prob_id = p_dict["id"]
            prompt = p_dict["prompt"]

            for temp in temps_to_run:
                for strat_name in strats_to_run:

                    # Determine alpha loop based on strategy
                    current_alphas = [0.0] if strat_name == "baseline" else alphas

                    for alpha in current_alphas:
                        # CREATE A UNIQUE FILENAME FOR EVERY COMBINATION
                        filename = os.path.join(out_dir, f"{ds_name}_{prob_id}_{strat_name}_a{alpha}_t{temp}.json")

                        if os.path.exists(filename):
                            continue  # Skip if already generated

                        strat_obj = get_strategy(strat_name, alpha, 1.0, extractor)
                        generator = AppGenerator(model, tokenizer, strat_obj, mask_token_id)
                        history, _ = generator.generate(prompt, batch_size, steps, gen_len, temp)

                        record = {
                            "id": f"{ds_name} | {prob_id} | {strat_name} | Alpha {alpha} | Temp {temp}",
                            "dataset": ds_name,
                            "problem_id": prob_id,
                            "strategy": strat_name,
                            "alpha": alpha,
                            "temp": temp,
                            "params": {"prompt": prompt, "batch": batch_size, "steps": steps, "alpha": alpha,
                                       "temp": temp, "strategy": strat_name},
                            "data": history
                        }

                        # Save the single configuration immediately
                        with open(filename, "w") as f:
                            json.dump([record], f)  # Wrapped in list for the app loader

    print(f"\n✅ Generation complete! Saved in ./{out_dir}/")


if __name__ == "__main__":
    main()