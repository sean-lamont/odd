import os
import json
import random
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


def load_config():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")
    return cfg


def get_gsm8k_prompts(num_samples=20):
    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    random.seed(42)
    # Randomly select out of the first 200
    indices = random.sample(range(min(200, len(ds))), num_samples)
    # Extracting 'question' specifically so the 'answer' isn't leaked into the prompt string
    return [{"id": f"Problem_{idx + 1}", "prompt": f"Question: {ds[i]['question']}\nLet's think step by step.\nAnswer:"}
            for idx, i in enumerate(indices)]


def get_humaneval_prompts(num_samples=20):
    print("Loading HumanEval dataset...")
    ds = load_dataset("openai_humaneval", split="test")
    random.seed(42)
    indices = random.sample(range(len(ds)), num_samples)
    return [{"id": f"Problem_{idx + 1}", "prompt": ds[i]["prompt"]} for idx, i in enumerate(indices)]


def main():
    out_dir = "demo_histories"
    os.makedirs(out_dir, exist_ok=True)

    print("Initializing model...")
    cfg = load_config()
    model, tokenizer, embedding_matrix, mask_token_id = load_model(cfg)

    # Grid Search Parameters
    batch_size = 4
    steps = 32
    alphas = [8.0, 16.0, 128.0]
    temps = [0.0, 1.0, 2.0]
    strategies = ["odd", "dpp"]

    print("Initializing Extractor...")
    extractor = FeatureExtractor(embedding_matrix=embedding_matrix, kernel_target="logits", pooling_method="max")

    # Map datasets to their specific generation lengths
    datasets_to_run = {
        "GSM8K": {"prompts": get_gsm8k_prompts(20), "gen_len": 64},
        "HumanEval": {"prompts": get_humaneval_prompts(20), "gen_len": 64}
    }

    # Total runs: 2 datasets * 20 problems * 3 temps * (1 baseline + 2 strats * 3 alphas) = 840 runs
    for ds_name, ds_info in datasets_to_run.items():
        print(f"\n================ Starting Dataset: {ds_name} ================")
        prompts = ds_info["prompts"]
        gen_len = ds_info["gen_len"]

        for p_dict in tqdm(prompts, desc=f"Processing {ds_name} Problems"):
            prob_id = p_dict["id"]
            prompt = p_dict["prompt"]
            problem_history = []

            filename = os.path.join(out_dir, f"{ds_name}_{prob_id}.json")
            if os.path.exists(filename):
                continue  # Skip if already generated (allows resuming if interrupted)

            for temp in tqdm(temps):
                # 1. Run Baseline (Standard Sampling)
                strat_obj = get_strategy("baseline", 0.0, 1.0, extractor)
                generator = AppGenerator(model, tokenizer, strat_obj, mask_token_id)
                history, _ = generator.generate(prompt, batch_size, steps, gen_len, temp)

                problem_history.append({
                    "id": f"{ds_name} | {prob_id} | Baseline | Temp {temp}",
                    "dataset": ds_name,
                    "problem_id": prob_id,
                    "strategy": "baseline",
                    "alpha": 0.0,
                    "temp": temp,
                    "params": {"prompt": prompt, "batch": batch_size, "steps": steps, "alpha": 0.0, "temp": temp,
                               "strategy": "baseline"},
                    "data": history
                })

                # 2. Run ODD & Joint DPP Strategies
                for strat_name in tqdm(strategies):
                    for alpha in tqdm(alphas):
                        strat_obj = get_strategy(strat_name, alpha, 1.0, extractor)
                        generator = AppGenerator(model, tokenizer, strat_obj, mask_token_id)
                        history, _ = generator.generate(prompt, batch_size, steps, gen_len, temp)

                        problem_history.append({
                            "id": f"{ds_name} | {prob_id} | {strat_name} | Alpha {alpha} | Temp {temp}",
                            "dataset": ds_name,
                            "problem_id": prob_id,
                            "strategy": strat_name,
                            "alpha": alpha,
                            "temp": temp,
                            "params": {"prompt": prompt, "batch": batch_size, "steps": steps, "alpha": alpha,
                                       "temp": temp, "strategy": strat_name},
                            "data": history
                        })

            # Save the entire grid for this specific problem
            with open(filename, "w") as f:
                json.dump(problem_history, f)

    print(f"\n✅ All generation grids complete! Saved in ./{out_dir}/")


if __name__ == "__main__":
    main()