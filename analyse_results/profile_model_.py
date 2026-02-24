import os

# minimal time overhead, saves significant VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import csv
import time
import torch
import wandb
from types import SimpleNamespace
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

from dpp_core import FeatureExtractor, get_strategy, DPPGenerator

from human_eval.data import read_problems


def load_model(cfg):
    print(f"Loading {cfg.model.name}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        cfg.model.name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch.bfloat16
    )

    model.eval()
    tokenizer.padding_side = 'left'

    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        embedding_matrix = model.model.transformer.wte.weight
    else:
        embedding_matrix = None

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = cfg.model.mask_token_id

    return model, tokenizer, embedding_matrix, mask_token_id


def get_dataset_prompts(max_gsm_problems=200):
    """Extracts prompts from datasets, returning a list of (Dataset_Name, Problem_Index, Prompt_Text)."""
    prompts = []

    try:
        gsm_dataset = load_dataset("gsm8k", "main", split="test")
        n_problems = min(max_gsm_problems, len(gsm_dataset))
        for i in range(n_problems):
            q = gsm_dataset[i]['question']
            prompt_text = f"Question: {q}\nLet's think step by step.\nAnswer:"
            prompts.append(("GSM8K", i, prompt_text))
    except Exception as e:
        print(f"Failed to load GSM8K: {e}")

    try:
        problems_dict = read_problems()
        problem_list = list(problems_dict.values())
        for i, problem in enumerate(problem_list):
            prompts.append(("HumanEval", i, problem['prompt']))
    except Exception as e:
        print(f"Failed to load HumanEval: {e}")

    return prompts


BASE_CONFIG = SimpleNamespace(
    model=SimpleNamespace(
        name="GSAI-ML/LLaDA-8B-Instruct",
        load_in_4bit=True,
        mask_token_id=126336
    ),
    strategy=SimpleNamespace(
        target="logits",
        pool="max",
        top_k=0
    ),
    wandb=SimpleNamespace(
        project="dpp-profiling",
        run_name="profile-run-table-" + time.strftime("%Y%m%d-%H%M%S"),
        artifact_freq=20  # Log CSV artifact every K scenarios
    )
)

batch_sizes = [4, 16, 64]
steps = [4, 8, 16]
length = [8, 32, 128]


def run_single_pass(generator, batch, steps, length, prompt):
    gc.collect()
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_t = time.perf_counter()
    _ = generator.generate(prompt=prompt, batch_size=batch, steps=steps, gen_length=length, temperature=1.0)

    if is_cuda:
        torch.cuda.synchronize()
        alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        res_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    else:
        alloc_mb, res_mb = 0.0, 0.0

    return time.perf_counter() - start_t, alloc_mb, res_mb


def main():
    print(f"--- Starting Profiler ---")
    wandb.init(
        project=BASE_CONFIG.wandb.project,
        name=BASE_CONFIG.wandb.run_name,
        config={
            "model": BASE_CONFIG.model.name,
            "4bit": BASE_CONFIG.model.load_in_4bit,
            "batches": batch_sizes,
            "steps": steps,
            "lengths": length
        }
    )

    columns = [
        "Scenario", "Dataset", "Problem_Idx", "Prompt_Tokens", "Batch", "Steps", "Length",
        "Time_Base(s)", "Time_Strat(s)", "Time_Overhead(%)",
        "Alloc_Base(MB)", "Alloc_Strat(MB)", "Alloc_Overhead(MB)",
        "Res_Base(MB)", "Res_Strat(MB)", "Res_Overhead(MB)", "Status"
    ]
    wandb_table = wandb.Table(columns=columns)

    model, tokenizer, embedding_matrix, mask_token_id = load_model(BASE_CONFIG)

    feature_extractor = FeatureExtractor(
        embedding_matrix=embedding_matrix,
        kernel_target=BASE_CONFIG.strategy.target,
        pooling_method=BASE_CONFIG.strategy.pool,
        top_k=BASE_CONFIG.strategy.top_k
    )

    dataset_prompts = get_dataset_prompts(max_gsm_problems=200)

    SCENARIOS = []
    for ds_name, prob_idx, prompt_text in dataset_prompts:
        messages = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = tokenizer([prompt_str], return_tensors="pt", padding=True, add_special_tokens=False)
        p_len = encoded.input_ids.shape[1]

        for b in batch_sizes:
            for s in steps:
                for l in length:
                    if s <= l:
                        label = f"{ds_name[:3]}_{prob_idx}_B{b}"
                        SCENARIOS.append((label, ds_name, prob_idx, p_len, prompt_text, b, s, l))

    csv_filename = "profiler_results_expand.csv"

    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(columns)

        for i, (label, ds_name, prob_idx, p_len, prompt_text, batch, steps_val, gen_length) in enumerate(SCENARIOS):
            try:
                strat_base = get_strategy("baseline", 0.0, 0.0, feature_extractor)
                gen_base = DPPGenerator(model, tokenizer, strat_base, mask_token_id)
                if i == 0: run_single_pass(gen_base, 1, 2, 5, "warmup")
                time_base, alloc_base, res_base = run_single_pass(gen_base, batch, steps_val, gen_length, prompt_text)

                strat_ortho = get_strategy("batched_orth", 64, 1.0, feature_extractor)
                gen_ortho = DPPGenerator(model, tokenizer, strat_ortho, mask_token_id)
                time_strat, alloc_strat, res_strat = run_single_pass(gen_ortho, batch, steps_val, gen_length,
                                                                     prompt_text)

                time_oh_pct = ((time_strat - time_base) / time_base) * 100.0 if time_base > 0 else 0.0
                alloc_oh_mb = alloc_strat - alloc_base
                res_oh_mb = res_strat - res_base

                row_data = [
                    label, ds_name, prob_idx, p_len, batch, steps_val, gen_length,
                    f"{time_base:.4f}", f"{time_strat:.4f}", f"{time_oh_pct:.2f}",
                    f"{alloc_base:.2f}", f"{alloc_strat:.2f}", f"{alloc_oh_mb:.2f}",
                    f"{res_base:.2f}", f"{res_strat:.2f}", f"{res_oh_mb:.2f}", "Success"
                ]

                csv_writer.writerow(row_data)
                csv_file.flush()

                wandb_table.add_data(*row_data)
                wandb.log({
                    "time/base": time_base, "time/strat": time_strat, "time/overhead_pct": time_oh_pct,
                    "mem/alloc_base": alloc_base, "mem/alloc_strat": alloc_strat, "mem/alloc_oh": alloc_oh_mb,
                    "mem/res_base": res_base, "mem/res_strat": res_strat, "mem/res_oh": res_oh_mb,
                    "params/batch": batch, "params/steps": steps_val, "params/length": gen_length, "params/p_len": p_len
                })

            except RuntimeError as e:
                status = "OOM" if "out of memory" in str(e).lower() else f"Error: {e}"
                row_err = [label, ds_name, prob_idx, p_len, batch, steps_val, gen_length] + [""] * 9 + [status]

                csv_writer.writerow(row_err)
                csv_file.flush()
                wandb_table.add_data(*row_err)

                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    wandb.log({"status": "OOM", "params/batch": batch})
                else:
                    wandb.log({"status": "Error", "error_msg": str(e)})

            if (i + 1) % BASE_CONFIG.wandb.artifact_freq == 0:
                artifact = wandb.Artifact(name="profiling_results_partial", type="dataset")
                artifact.add_file(csv_filename)
                wandb.log_artifact(artifact)

    wandb.log({"profiling_summary_table": wandb_table})

    final_artifact = wandb.Artifact(name="profiling_results_final", type="dataset")
    final_artifact.add_file(csv_filename)
    wandb.log_artifact(final_artifact)

    wandb.finish()
    print(f"\nProfiling complete. Table and results logged to WandB.")


if __name__ == "__main__":
    main()