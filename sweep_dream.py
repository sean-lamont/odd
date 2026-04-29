import os
import re
import sys
import time

import hydra
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

import wandb
from feature_extractor import FeatureExtractor
from strategies import get_strategy
from sweep_human_eval import clean_code_for_harness
from utils import calculate_diversity_score

sys.path.append(os.path.join(os.getcwd(), "human-eval"))
from human_eval.data import read_problems
from human_eval.execution import check_correctness

from tqdm import tqdm

# --- Sweep Parameters ---
TARGET_TASKS = ["gsm8k"]#, "humaneval"]
ALGS = ["origin"]#["maskgit_plus", "origin"]
TEMPERATURES = [0.0, 0.5, 1.0, 1.5, 2.0]
ALPHAS = [8.0, 16.0, 64.0, 128.0]

STEPS = 32
GEN_LENGTH = 64
BATCH_SIZE = 16
N_RUNS = 4

BEST_PARAMS = {
    "gsm8k": {"n_problems": 200},
    "humaneval": {"n_problems": 164}
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
    print(">>> Loading DREAM Model...")
    model_path = "Dream-org/Dream-v0-Instruct-7B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModel.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        for t in ["<|mask|>", "[MASK]", "<mask>"]:
            if t in tokenizer.get_vocab():
                mask_token_id = tokenizer.get_vocab()[t]
                break

    embedding_matrix = model.model.embed_tokens.weight.detach()
    eval_model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f">>> Initializing Global Resources for Dream Sweep...")

    # Pre-load datasets
    loaded_datasets = {}
    for TASK in TARGET_TASKS:
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

    # 1. Feature Extractor (shared across runs, gradients fixed)
    feature_extractor = FeatureExtractor(
        embedding_matrix=embedding_matrix, kernel_target="logits",
        pooling_method="max", top_k=0, use_confidence_weighting=True,
        ignore_token_ids=[]  # Prevents the in-place softmax bug
    )

    # ---------------------------------------------------------
    # MAIN SWEEP LOOPS
    # ---------------------------------------------------------
    for run_idx in range(1, N_RUNS + 1):
        print(f"\n{'=' * 80}")
        print(f">>> STARTING STATISTICAL RUN {run_idx}/{N_RUNS}")
        print(f"{'=' * 80}")

        for TASK in TARGET_TASKS:
            dataset, problem_indices = loaded_datasets[TASK]

            for alg in ALGS:
                for temp in TEMPERATURES:

                    # Generate the config permutations for this Temp/Alg combination
                    # Baseline (no hook) + ODD sweeps
                    configs_to_run = [{"strategy": "baseline", "alpha": 0.0}]
                    for a in ALPHAS:
                        configs_to_run.append({"strategy": "odd", "alpha": a})

                    for cfg_dict in configs_to_run:
                        strat_name = cfg_dict["strategy"]
                        alpha_val = cfg_dict["alpha"]

                        run_name = f"r{run_idx}_{TASK}_{alg}_T{temp}_{strat_name}"
                        if strat_name == "odd":
                            run_name += f"_a{alpha_val}"

                        print(f"\n{'-' * 60}")
                        print(
                            f">>> RUNNING: Task={TASK.upper()} | Alg={alg} | Temp={temp} | Strat={strat_name.upper()} | Alpha={alpha_val}")
                        print(f"{'-' * 60}")

                        run = wandb.init(
                            project=f"dream_{TASK}_eval",
                            group=f"dream_sweep_run_{run_idx}",
                            name=run_name,
                            config={
                                "task": TASK,
                                "alg": alg,
                                "temperature": temp,
                                "strategy": strat_name,
                                "alpha": alpha_val,
                                "steps": STEPS,
                                "gen_length": GEN_LENGTH,
                                "batch_size": BATCH_SIZE
                            },
                            reinit=True
                        )

                        results_table = wandb.Table(
                            columns=["question/task_id", "gold/prompt", "generated", "is_correct", "diversity"]
                        )

                        odd_strategy = None
                        if strat_name == "odd":
                            odd_strategy = get_strategy("odd", alpha_val, 1.0, feature_extractor)

                        pass_at_k_totals = {k: [] for k in range(1, BATCH_SIZE + 1)}
                        cumulative_totals = {k: 0 for k in range(1, BATCH_SIZE + 1)}
                        diversity_scores = []
                        gen_times = []

                        try:
                            for i in tqdm(problem_indices):
                                start_t = time.time()

                                # Prepare Prompt
                                if TASK == "gsm8k":
                                    row = dataset[i]
                                    q = row['question']
                                    gold = extract_gold_num(row['answer'])
                                    if gold is None: continue
                                    raw_prompt = f"Question: {q}\nLet's think step by step.\nAnswer:"
                                    task_id = q
                                    gold_data = gold
                                else:
                                    problem = dataset[i]
                                    task_id = problem['task_id']
                                    raw_prompt = problem['prompt']
                                    gold_data = raw_prompt

                                # Format via tokenizer
                                messages = [{"role": "user", "content": raw_prompt}]
                                prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                                           tokenize=False)

                                encoded = tokenizer([prompt_str] * BATCH_SIZE, return_tensors="pt", padding=True,
                                                    add_special_tokens=False)
                                input_ids = encoded.input_ids.to(model.device)
                                attention_mask = encoded.attention_mask.to(model.device)
                                prompt_len = input_ids.shape[1]

                                # Define Hook dynamically for the current prompt length
                                def get_hook(current_alpha):
                                    def hook(step, x, logits):
                                        with torch.enable_grad():
                                            gen_x = x[:, prompt_len:]
                                            gen_logits = logits[:, prompt_len:, :].clone()
                                            gen_mask = (gen_x == mask_token_id)

                                            if not gen_mask.any():
                                                return logits

                                            step_alpha = current_alpha * (1.0 - (step / STEPS))
                                            odd_strategy.alpha = step_alpha

                                            if step_alpha > 0.0:
                                                guided_gen_logits, _ = odd_strategy.apply(
                                                    logits=gen_logits, mask_index=gen_mask, x=gen_x,
                                                    history_vecs=[], history_qualities=[], protected_tokens=None
                                                )
                                                logits[:, prompt_len:, :] = guided_gen_logits.detach()

                                            return logits

                                    return hook

                                active_hook = get_hook(alpha_val) if strat_name == "odd" else None

                                # Setup Generation Arguments
                                gen_kwargs = {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                    "max_new_tokens": GEN_LENGTH,
                                    "steps": STEPS,
                                    "temperature": temp if temp > 0.0 else 0.0,
                                    "alg": alg,
                                    "return_dict_in_generate": True,
                                    # "generation_logits_hook_func": active_hook
                                }
                                
                                if active_hook:
                                    gen_kwargs["generation_logits_hook_func"] = active_hook

                                # Inject proper sampling flags if temp > 0
                                # if temp > 0.0:
                                #     gen_kwargs["alg_temp"] = temp
                                #     gen_kwargs["top_p"] = 1.0

                                # Execute Generation
                                with torch.no_grad():
                                    # output = model.diffusion_generate(**gen_kwargs)

                                    if active_hook:
                                        output = model.diffusion_generate(
                                    input_ids, attention_mask=attention_mask,
                                    max_new_tokens=GEN_LENGTH, steps=STEPS,
                                    temperature=temp if temp > 0.0 else 0.0,
                                    top_p=1.0,
                                    alg=alg,
                                    return_dict_in_generate=True,
                                    generation_logits_hook_func=active_hook
                                    )
                                    else:
                                        output = model.diffusion_generate(
                                            input_ids, attention_mask=attention_mask,
                                            max_new_tokens=GEN_LENGTH, steps=STEPS,
                                            temperature=temp if temp > 0.0 else 0.0,
                                            top_p=1.0,
                                            alg=alg,
                                            return_dict_in_generate=True,
                                            # generation_logits_hook_func=active_hook
                                        )

                                gen_times.append(time.time() - start_t)

                                # Decode
                                samples = [
                                    tokenizer.decode(g[prompt_len:].tolist(), skip_special_tokens=True)
                                    for g in output.sequences
                                ]

                                # Evaluate
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
                                        cleaned_code = clean_code_for_harness(raw_prompt, s)
                                        res = check_correctness(problem, cleaned_code, timeout=3.0)
                                        batch_results.append((s, cleaned_code, res))
                                        correct_flags.append(res['passed'])

                                    div = calculate_diversity_score(eval_model, samples)
                                    diversity_scores.append(div)
                                    for s, cleaned_s, res in batch_results:
                                        results_table.add_data(task_id, gold_data, cleaned_s, res['passed'], div)

                                # Pass@K Math
                                cumulative_correct = 0
                                for k in range(1, BATCH_SIZE + 1):
                                    score = 1.0 if any(correct_flags[:k]) else 0.0
                                    cumulative_correct += score
                                    pass_at_k_totals[k].append(score)
                                    cumulative_totals[k] = cumulative_correct

                            # Log Final Metrics for Configuration
                            avg_pass_at_k = {f"pass_at_{k}": np.mean(v) for k, v in pass_at_k_totals.items()}
                            avg_cumulative_at_k = {f"cumulative_at_{k}": np.mean(v) for k, v in
                                                   cumulative_totals.items()}

                            avg_div = np.mean(diversity_scores) if diversity_scores else 0.0
                            std_div = np.std(diversity_scores) if diversity_scores else 0.0
                            avg_time = np.mean(gen_times) if gen_times else 0.0
                            std_time = np.std(gen_times) if gen_times else 0.0

                            target_metric = avg_pass_at_k[f"pass_at_{BATCH_SIZE}"]

                            print(
                                f"RESULTS: Pass@1: {avg_pass_at_k['pass_at_1']:.4f} | Pass@{BATCH_SIZE}: {target_metric:.4f} | Div: {avg_div:.4f}")

                            log_dict = {
                                "run_idx": run_idx,
                                "task": TASK,
                                "alg": alg,
                                "temperature": temp,
                                "strategy": strat_name,
                                "alpha": alpha_val,
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
