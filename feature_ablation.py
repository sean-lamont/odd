import os
import re
import sys
import time

import hydra
import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer

import wandb
from feature_extractor import FeatureExtractor
from generator import DiverseGenerator
from strategies import get_strategy
from sweep_human_eval import clean_code_for_harness
from utils import load_model, calculate_diversity_score

sys.path.append(os.path.join(os.getcwd(), "human-eval"))
from human_eval.data import read_problems
from human_eval.execution import check_correctness

from tqdm import tqdm

# Set the target tasks
TARGET_TASK = ["gsm8k", "humaneval"]

# Fixed parameters for the ablation
STEPS = 32
GEN_LENGTH = 64
N_RUNS = 4

# The explicit ablation grid (Baseline removed as requested)
ABLATION_CONFIGS = [
    # 1. Top-k Ablations (Logits + Max Pool)
    {"target": "logits", "pool": "max", "top_k": 32},
    {"target": "logits", "pool": "max", "top_k": 64},
    {"target": "logits", "pool": "max", "top_k": 128},
    {"target": "logits", "pool": "max", "top_k": 256},

    # 2. Embedding Baseline
    {"target": "embeddings", "pool": "max", "top_k": 0},

    # 3. Positional Pooling
    {"target": "logits", "pool": "positional", "top_k": 0},
    {"target": "embeddings", "pool": "positional", "top_k": 0},
]

# Optimal hyperparameters derived from the primary sweep
BEST_PARAMS = {
    "gsm8k": {"alpha": 128.0, "temperature": 1.5, "n_problems": 200},
    "humaneval": {"alpha": 16.0, "temperature": 1.0, "n_problems": 164}
}


def extract_answer_num(text):
    try:
        text = text.replace(',', '')
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums: return float(nums[-1])
    except Exception:
        pass
    return None


def extract_gold_num(answer_str):
    if "####" in answer_str:
        try:
            val = answer_str.split("####")[1].strip()
            return float(val.replace(',', ''))
        except:
            pass
    return None


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(base_cfg):
    batch_size = 16
    model, tokenizer, embedding_matrix, mask_token_id = load_model(base_cfg)
    eval_model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f">>> Initializing Global Resources for Feature Ablation Eval...")

    # Pre-load datasets so we don't re-fetch them 4 times
    loaded_datasets = {}
    for TASK in TARGET_TASK:
        params = BEST_PARAMS[TASK]
        if TASK == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="test")
            problem_indices = range(len(dataset)) if params["n_problems"] == -1 else range(
                min(params["n_problems"], len(dataset)))
            loaded_datasets[TASK] = (dataset, problem_indices)
        else:
            problems_dict = read_problems()
            dataset = list(problems_dict.values())
            problem_indices = range(min(params["n_problems"], len(dataset)))
            loaded_datasets[TASK] = (dataset, problem_indices)

    # Outer loop: Run number (1 to 4)
    for run_idx in range(1, N_RUNS + 1):
        print(f"\n{'=' * 60}")
        print(f">>> STARTING STATISTICAL RUN {run_idx}/{N_RUNS}")
        print(f"{'=' * 60}")

        for TASK in TARGET_TASK:
            params = BEST_PARAMS[TASK]
            dataset, problem_indices = loaded_datasets[TASK]

            # Iterate over the grid
            for grid_cfg in ABLATION_CONFIGS:

                cfg = base_cfg.copy()
                if "strategy" not in cfg: cfg.strategy = {}
                cfg.strategy.name = "odd"
                cfg.strategy.alpha = params["alpha"]
                cfg.strategy.quality_scale = 1.0

                # Apply ablation specifics
                cfg.strategy.target = grid_cfg["target"]
                cfg.strategy.pool = grid_cfg["pool"]
                cfg.strategy.top_k = grid_cfg["top_k"]

                cfg.temperature = params["temperature"]
                cfg.batch_size = batch_size
                cfg.steps = STEPS
                cfg.ignore_pad = False

                run_name = f"run{run_idx}_{TASK}_{grid_cfg['target']}_{grid_cfg['pool']}_topk{grid_cfg['top_k']}"
                run = wandb.init(
                    project=f"{TASK}_feature_ablation",
                    group=f"dpp_run_{run_idx}",
                    name=run_name,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    reinit=True
                )

                results_table = wandb.Table(
                    columns=["question/task_id", "gold/prompt", "generated", "is_correct", "diversity"]
                )

                try:
                    print(f"\n{'-' * 40}")
                    print(
                        f">>> RUNNING: Task={TASK.upper()}, Target={grid_cfg['target']}, Pool={grid_cfg['pool']}, Top-k={grid_cfg['top_k']}")
                    print(f"{'-' * 40}")

                    feature_extractor = FeatureExtractor(
                        embedding_matrix=embedding_matrix,
                        kernel_target=cfg.strategy.target,
                        pooling_method=cfg.strategy.pool,
                        top_k=cfg.strategy.get("top_k", 0),
                        use_confidence_weighting=cfg.get('use_confidence_weighting', True),
                        ignore_token_ids=[tokenizer.pad_token_id] if cfg.get('ignore_pad', False) else []
                    )

                    current_strategy = get_strategy(
                        cfg.strategy.name,
                        cfg.strategy.alpha,
                        cfg.strategy.quality_scale,
                        feature_extractor
                    )

                    generator = DiverseGenerator(model, tokenizer, current_strategy, mask_token_id)

                    pass_at_k_totals = {k: [] for k in range(1, batch_size + 1)}
                    cumulative_totals = {k: 0 for k in range(1, batch_size + 1)}
                    diversity_scores = []
                    gen_times = []

                    for i in tqdm(problem_indices):
                        start_t = time.time()

                        if TASK == "gsm8k":
                            row = dataset[i]
                            q = row['question']
                            gold = extract_gold_num(row['answer'])
                            if gold is None: continue
                            prompt = f"Question: {q}\nLet's think step by step.\nAnswer:"
                            task_id = q
                            gold_data = gold
                        else:
                            problem = dataset[i]
                            task_id = problem['task_id']
                            prompt = problem['prompt']
                            gold_data = prompt

                        _, samples = generator.generate(
                            prompt=prompt,
                            batch_size=cfg.batch_size,
                            steps=cfg.steps,
                            gen_length=GEN_LENGTH,
                            temperature=cfg.temperature,
                        )

                        gen_times.append(time.time() - start_t)

                        correct_flags = []
                        if TASK == "gsm8k":
                            for s in samples:
                                val = extract_answer_num(s)
                                is_correct = (val is not None and abs(val - gold) < 1e-4)
                                correct_flags.append(is_correct)

                            div = calculate_diversity_score(eval_model, samples)
                            diversity_scores.append(div)
                            for s, is_correct in zip(samples, correct_flags):
                                results_table.add_data(task_id, gold_data, s, is_correct, div)

                        else:
                            batch_results = []
                            for s in samples:
                                cleaned_code = clean_code_for_harness(prompt, s)
                                res = check_correctness(problem, cleaned_code, timeout=3.0)
                                batch_results.append((s, cleaned_code, res))
                                correct_flags.append(res['passed'])

                            div = calculate_diversity_score(eval_model, samples)
                            diversity_scores.append(div)
                            for s, cleaned_s, res in batch_results:
                                results_table.add_data(task_id, gold_data, cleaned_s, res['passed'], div)

                        cumulative_correct = 0
                        for k in range(1, batch_size + 1):
                            score = 1.0 if any(correct_flags[:k]) else 0.0
                            cumulative_correct += score
                            pass_at_k_totals[k].append(score)
                            cumulative_totals[k] = cumulative_correct

                        # Commented out per-problem print to reduce spam with tqdm
                        # print(f"[{i + 1}/{len(problem_indices)}] Correct: {cumulative_correct} | Time: {gen_times[-1]:.2f}s")

                    avg_pass_at_k = {f"pass_at_{k}": np.mean(v) for k, v in pass_at_k_totals.items()}
                    avg_cumulative_at_k = {f"cumulative_at_{k}": np.mean(v) for k, v in cumulative_totals.items()}

                    avg_div = np.mean(diversity_scores) if diversity_scores else 0.0
                    std_div = np.std(diversity_scores) if diversity_scores else 0.0
                    avg_time = np.mean(gen_times) if gen_times else 0.0
                    std_time = np.std(gen_times) if gen_times else 0.0

                    target_metric = avg_pass_at_k[f"pass_at_{batch_size}"]

                    print(
                        f"\nRESULTS: Pass@1: {avg_pass_at_k['pass_at_1']:.4f} | Pass@{batch_size}: {target_metric:.4f} | Div: {avg_div:.4f}")

                    log_dict = {
                        "run_idx": run_idx,
                        "target": grid_cfg["target"],
                        "pool": grid_cfg["pool"],
                        "top_k": grid_cfg["top_k"],
                        "avg_diversity": avg_div,
                        "std_diversity": std_div,
                        "avg_time": avg_time,
                        "std_time": std_time,
                        "results_table": results_table,
                    }
                    log_dict.update(avg_pass_at_k)
                    log_dict.update(avg_cumulative_at_k)

                    wandb.log(log_dict)

                finally:
                    wandb.finish()


if __name__ == "__main__":
    main()