"""Shared plumbing for the rebuttal benchmark harnesses (sweep_mbpp.py / sweep_math500.py).

Unlike sweep_human_eval.py / sweep_gsm8k.py these harnesses do NOT use Optuna or a
Postgres storage backend: they run a plain grid loop over
strategies x alphas x temperatures x runs, driven by argparse, and log each run to
wandb (full-text results_table included, so diversity metrics can be recomputed
later) plus a local CSV + JSONL under --results-dir.

All torch / model / sentence-transformers imports are lazy so that --dry-run works
on a CPU-only machine with none of the heavy dependencies installed.
"""

import argparse
import csv
import json
import os
import time

import hydra
import numpy as np
import wandb
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser(description, default_project, default_gen_length):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--strategies", nargs="+", default=["baseline", "odd"],
                        help="Strategy config names under conf/strategy (default: baseline odd)")
    parser.add_argument("--alphas", nargs="+", type=float, default=[16.0],
                        help="Strategy alphas to sweep (ignored/collapsed to 0 for baseline)")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.5, 1.0, 1.5],
                        help="Sampling temperatures to sweep (0 and 2 are degenerate cases)")
    parser.add_argument("--n-problems", type=int, default=-1,
                        help="Number of problems to evaluate (-1 = all)")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of repeats of the full grid")
    parser.add_argument("--steps", type=int, default=32, help="Diffusion steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Samples per problem (max k for pass@k)")
    parser.add_argument("--gen-length", type=int, default=default_gen_length,
                        help="Generation length in tokens")
    parser.add_argument("--model-config", type=str, default="llada",
                        help="Model config name under conf/model")
    parser.add_argument("--alg", type=str, default="maskgit_plus",
                        choices=["origin", "maskgit_plus", "entropy", "topk_margin"],
                        help="Dream only: diffusion_generate remasking strategy "
                             "(paper setting: maskgit_plus; Dream's own eval.sh uses entropy)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Dream only: nucleus top_p for diffusion_generate "
                             "(default None = 1.0, i.e. sweep_dream.py behaviour)")
    parser.add_argument("--eos-conf-inf", action="store_true",
                        help="LLaDA only: floor the unmask confidence of EOS/PAD "
                             "predictions (official generate.py confidence_eos_eot_inf "
                             "analogue; default off = paper behaviour)")
    parser.add_argument("--wandb-project", type=str, default=default_project)
    parser.add_argument("--wandb-group", type=str, default="rebuttal")
    parser.add_argument("--wandb-mode", type=str, default=None,
                        choices=[None, "online", "offline", "disabled"],
                        help="wandb mode override (dry-run defaults to offline)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Local directory for CSV/JSONL outputs")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU-only smoke test: stub generator instead of the model")
    return parser


def iter_grid(args):
    """Yield (run_idx, strategy, alpha, temperature) combos.

    For the baseline strategy alpha has no effect (BaselineStrategy hardcodes
    alpha=0), so the alpha axis is collapsed to a single 0.0 entry to avoid
    duplicate runs.
    """
    for run_idx in range(args.n_runs):
        for strategy in args.strategies:
            alphas = [0.0] if strategy == "baseline" else args.alphas
            for alpha in alphas:
                for temperature in args.temperatures:
                    yield run_idx, strategy, alpha, temperature


# ---------------------------------------------------------------------------
# Hydra config
# ---------------------------------------------------------------------------

def compose_cfg(args, strategy, alpha, temperature):
    """Compose the repo hydra config with per-combo overrides.

    config_path is resolved relative to this file, so it works from any cwd.
    """
    overrides = [
        f"model={args.model_config}",
        f"strategy={strategy}",
        f"strategy.alpha={alpha}",
        f"temperature={temperature}",
        f"batch_size={args.batch_size}",
        f"steps={args.steps}",
        f"gen_length={args.gen_length}",
    ]
    if getattr(args, "alg", None):
        overrides.append(f"++model.alg={args.alg}")
    if getattr(args, "top_p", None) is not None:
        overrides.append(f"++model.top_p={args.top_p}")
    if getattr(args, "eos_conf_inf", False):
        overrides.append("++eos_conf_inf=true")
    with hydra.initialize(version_base=None, config_path="conf"):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


# ---------------------------------------------------------------------------
# Model / generator construction (lazy heavy imports)
# ---------------------------------------------------------------------------

def load_shared_resources(cfg):
    """Load the (4-bit) model once, plus the sentence-transformer used for the
    diversity metric. Mirrors the global setup of sweep_human_eval.py so memory
    behaviour is identical. Model configs with backend: "dream" are routed to
    the Dream loader (which mirrors sweep_dream.py)."""
    if cfg.model.get("backend", "llada") == "dream":
        return _load_dream_resources(cfg)
    if cfg.model.get("backend", "llada") == "llada2":
        return _load_llada2_resources(cfg)
    if cfg.model.get("backend", "llada") == "gemma_diffusion":
        return _load_gemma_diffusion_resources(cfg)

    from sentence_transformers import SentenceTransformer

    from utils import load_model

    model, tokenizer, embedding_matrix, mask_token_id = load_model(cfg)
    eval_model = SentenceTransformer("all-MiniLM-L6-v2")
    return {
        "backend": "llada",
        "model": model,
        "tokenizer": tokenizer,
        "embedding_matrix": embedding_matrix,
        "mask_token_id": mask_token_id,
        "eval_model": eval_model,
    }


def _load_dream_resources(cfg):
    """Dream model setup, lifted from sweep_dream.py: 4-bit nf4 bnb quant,
    trust_remote_code AutoModel, mask token resolved from the tokenizer with a
    vocab fallback, embedding matrix from model.model.embed_tokens.

    NOTE: Dream needs its own env (transformers 4.46.x, torch 2.5.x); run with
    PYTHONNOUSERSITE=1 if the box has stray user-site packages."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {cfg.model.name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModel.from_pretrained(
        cfg.model.name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    model.eval()

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        for t in ["<|mask|>", "[MASK]", "<mask>"]:
            if t in tokenizer.get_vocab():
                mask_token_id = tokenizer.get_vocab()[t]
                break

    embedding_matrix = model.model.embed_tokens.weight.detach()
    eval_model = SentenceTransformer("all-MiniLM-L6-v2")
    return {
        "backend": "dream",
        "model": model,
        "tokenizer": tokenizer,
        "embedding_matrix": embedding_matrix,
        "mask_token_id": mask_token_id,
        "eval_model": eval_model,
    }


def _load_llada2_resources(cfg):
    """LLaDA2.0 block-diffusion setup via llada2_generator.load_llada2
    (AutoModelForCausalLM + trust_remote_code, 4-bit nf4). Features are
    logit-space (llada2_smoke.py convention), so no embedding matrix is
    exposed; mask/eos ids come from the model config yaml."""
    from sentence_transformers import SentenceTransformer

    from llada2_generator import load_llada2

    print(f"Loading {cfg.model.name}...")
    model, tokenizer = load_llada2(cfg.model.name, cfg.model.load_in_4bit)
    eval_model = SentenceTransformer("all-MiniLM-L6-v2")
    return {
        "backend": "llada2",
        "model": model,
        "tokenizer": tokenizer,
        "embedding_matrix": None,
        "mask_token_id": cfg.model.mask_token_id,
        "eval_model": eval_model,
    }


def _load_gemma_diffusion_resources(cfg):
    """DiffusionGemma setup via gemma_diffusion_generator.load_diffusion_gemma
    (DiffusionGemmaForBlockDiffusion + 4-bit nf4, transformers >= 5.14).
    Features are logit-space; there is no mask token (random-init canvases)."""
    from sentence_transformers import SentenceTransformer

    from gemma_diffusion_generator import load_diffusion_gemma

    print(f"Loading {cfg.model.name}...")
    model, processor = load_diffusion_gemma(cfg.model.name, cfg.model.load_in_4bit)
    eval_model = SentenceTransformer("all-MiniLM-L6-v2")
    return {
        "backend": "gemma_diffusion",
        "model": model,
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "embedding_matrix": None,
        "mask_token_id": None,
        "eval_model": eval_model,
    }


def build_generator(cfg, shared):
    """Build FeatureExtractor + strategy + generator for one grid combo.

    LLaDA-style backends get DiverseGenerator exactly as sweep_human_eval.py
    does inside its objective(); backend "dream" gets a DreamGenerator that
    injects the strategy through diffusion_generate's logits hook, exactly as
    sweep_dream.py does."""
    from feature_extractor import FeatureExtractor
    from strategies import get_strategy

    if shared.get("backend") == "dream":
        # sweep_dream.py always builds the extractor with ignore_token_ids=[]
        # ("Prevents the in-place softmax bug") -- mirror that here.
        feature_extractor = FeatureExtractor(
            embedding_matrix=shared["embedding_matrix"],
            kernel_target=cfg.strategy.target,
            pooling_method=cfg.strategy.pool,
            top_k=cfg.strategy.get("top_k", 0),
            use_confidence_weighting=cfg.get("use_confidence_weighting", True),
            ignore_token_ids=[],
        )
        strategy = None
        if cfg.strategy.name != "baseline":
            strategy = get_strategy(
                cfg.strategy.name,
                cfg.strategy.alpha,
                cfg.strategy.quality_scale,
                feature_extractor,
            )
        return DreamGenerator(
            shared["model"], shared["tokenizer"], strategy,
            shared["mask_token_id"], cfg.model.alg,
            top_p=cfg.model.get("top_p", None),
        )

    if shared.get("backend") == "gemma_diffusion":
        # Dream-style: baseline passes NO processor at all; ODD rides the
        # public logits_processor API. Logit-space features as with llada2.
        from gemma_diffusion_generator import GemmaDiffusionDiverseGenerator

        strategy = None
        if cfg.strategy.name != "baseline":
            feature_extractor = FeatureExtractor(
                embedding_matrix=None,
                kernel_target=cfg.strategy.target,
                pooling_method=cfg.strategy.pool,
                top_k=cfg.strategy.get("top_k", 0),
                use_confidence_weighting=cfg.get("use_confidence_weighting", True),
                ignore_token_ids=[],
            )
            strategy = get_strategy(
                cfg.strategy.name,
                cfg.strategy.alpha,
                cfg.strategy.quality_scale,
                feature_extractor,
            )
        return GemmaDiffusionDiverseGenerator(shared["model"], shared["processor"], strategy)

    if shared.get("backend") == "llada2":
        # llada2_smoke.py conventions: logit-space features (no embedding
        # matrix), ignore_token_ids=[] -- eos/pad/mask protection happens
        # inside LLaDA2DiverseGenerator via protected_tokens.
        from llada2_generator import LLaDA2DiverseGenerator

        if cfg.strategy.name == "baseline":
            strategy = get_strategy("baseline", 0.0, 0.0, None)
        else:
            feature_extractor = FeatureExtractor(
                embedding_matrix=None,
                kernel_target=cfg.strategy.target,
                pooling_method=cfg.strategy.pool,
                top_k=cfg.strategy.get("top_k", 0),
                use_confidence_weighting=cfg.get("use_confidence_weighting", True),
                ignore_token_ids=[],
            )
            strategy = get_strategy(
                cfg.strategy.name,
                cfg.strategy.alpha,
                cfg.strategy.quality_scale,
                feature_extractor,
            )
        return LLaDA2DiverseGenerator(
            shared["model"], shared["tokenizer"], strategy,
            mask_token_id=cfg.model.mask_token_id,
            block_length=cfg.model.block_length,
            threshold=cfg.model.threshold,
            eos_token_id=cfg.model.eos_token_id,
        )

    from generator import DiverseGenerator

    feature_extractor = FeatureExtractor(
        embedding_matrix=shared["embedding_matrix"],
        kernel_target=cfg.strategy.target,
        pooling_method=cfg.strategy.pool,
        top_k=cfg.strategy.get("top_k", 0),
        use_confidence_weighting=cfg.get("use_confidence_weighting", True),
        ignore_token_ids=[shared["tokenizer"].pad_token_id] if cfg.get("ignore_pad", False) else [],
    )
    strategy = get_strategy(
        cfg.strategy.name,
        cfg.strategy.alpha,
        cfg.strategy.quality_scale,
        feature_extractor,
    )
    return DiverseGenerator(shared["model"], shared["tokenizer"], strategy, shared["mask_token_id"],
                            eos_conf_inf=bool(cfg.get("eos_conf_inf", False)))


class DreamGenerator:
    """Wraps Dream's model.diffusion_generate behind the same generate()
    interface as generator.DiverseGenerator, replicating sweep_dream.py:
      - prompt is wrapped in the chat template (add_generation_prompt=True),
      - batch_size independent samples per problem ([prompt] * B),
      - the ODD strategy is injected across the batch via
        generation_logits_hook_func with step-decayed alpha
        (alpha * (1 - step/steps)), applied to the generated slice only,
      - strategy=None means baseline (no hook passed at all),
      - temperature clamped at 0.0; alg from the model config / --alg
        (maskgit_plus = paper setting; origin/entropy/topk_margin selectable);
      - top_p defaults to 1.0 (sweep_dream.py behaviour) unless --top-p is
        given (e.g. 0.9 for Dream's native eval.sh-style config).
    torch is imported lazily so --dry-run works without it."""

    def __init__(self, model, tokenizer, strategy, mask_token_id, alg, top_p=None):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy  # None => baseline
        # Freeze the configured alpha now: the hook mutates strategy.alpha
        # every step (as sweep_dream.py does), so the decay must always start
        # from the original value.
        self.base_alpha = strategy.alpha if strategy is not None else 0.0
        self.mask_token_id = mask_token_id
        self.alg = alg
        self.top_p = 1.0 if top_p is None else top_p

    def generate(self, prompt, batch_size, steps, gen_length, temperature):
        import torch

        messages = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        encoded = self.tokenizer(
            [prompt_str] * batch_size, return_tensors="pt", padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded.input_ids.to(self.model.device)
        attention_mask = encoded.attention_mask.to(self.model.device)
        prompt_len = input_ids.shape[1]

        active_hook = None
        if self.strategy is not None:
            strategy = self.strategy
            base_alpha = self.base_alpha
            mask_token_id = self.mask_token_id

            def hook(step, x, logits):
                with torch.enable_grad():
                    gen_x = x[:, prompt_len:]
                    gen_logits = logits[:, prompt_len:, :].clone()
                    gen_mask = (gen_x == mask_token_id)

                    if not gen_mask.any():
                        return logits

                    step_alpha = base_alpha * (1.0 - (step / steps))
                    strategy.alpha = step_alpha

                    if step_alpha > 0.0:
                        guided_gen_logits, _ = strategy.apply(
                            logits=gen_logits, mask_index=gen_mask, x=gen_x,
                            history_vecs=[], history_qualities=[],
                            protected_tokens=None,
                        )
                        logits[:, prompt_len:, :] = guided_gen_logits.detach()

                    return logits

            active_hook = hook

        gen_kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": gen_length,
            "steps": steps,
            "temperature": temperature if temperature > 0.0 else 0.0,
            "top_p": self.top_p,
            "alg": self.alg,
            "return_dict_in_generate": True,
        }
        if active_hook is not None:
            gen_kwargs["generation_logits_hook_func"] = active_hook

        with torch.no_grad():
            output = self.model.diffusion_generate(input_ids, **gen_kwargs)

        samples = [
            self.tokenizer.decode(g[prompt_len:].tolist(), skip_special_tokens=True)
            for g in output.sequences
        ]
        return [], samples


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def make_diversity_fn(dry_run, shared=None):
    if dry_run:
        return _stub_diversity
    from utils import calculate_diversity_score
    eval_model = shared["eval_model"]
    return lambda texts: calculate_diversity_score(eval_model, texts)


def _stub_diversity(texts):
    """Cheap lexical stand-in for the embedding diversity score (dry-run only):
    1 - mean pairwise Jaccard similarity over whitespace tokens."""
    if len(texts) < 2:
        return 0.0
    sets = [set(t.split()) for t in texts]
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            sims.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return 1.0 - sum(sims) / len(sims)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def update_pass_at_k(correct_flags, batch_size, pass_at_k_totals):
    """Empirical pass@k by prefix slicing, identical to the existing sweeps:
    pass@k = 1 if any of the first k samples is correct. Returns the number of
    k values that passed for this problem (the 'cumulative correct' printout)."""
    cumulative_correct = 0
    for k in range(1, batch_size + 1):
        score = 1.0 if any(correct_flags[:k]) else 0.0
        cumulative_correct += score
        pass_at_k_totals[k].append(score)
    return cumulative_correct


def aggregate_metrics(pass_at_k_totals, diversity_scores, gen_times):
    metrics = {f"pass_at_{k}": float(np.mean(v)) for k, v in pass_at_k_totals.items()}
    metrics["avg_diversity"] = float(np.mean(diversity_scores)) if diversity_scores else 0.0
    metrics["std_diversity"] = float(np.std(diversity_scores)) if diversity_scores else 0.0
    metrics["avg_time"] = float(np.mean(gen_times)) if gen_times else 0.0
    metrics["std_time"] = float(np.std(gen_times)) if gen_times else 0.0
    metrics["n_problems_evaluated"] = len(gen_times)
    return metrics


# ---------------------------------------------------------------------------
# wandb + local persistence
# ---------------------------------------------------------------------------

def init_wandb(args, cfg, run_name):
    mode = args.wandb_mode
    if mode is None and args.dry_run and "WANDB_MODE" not in os.environ:
        mode = "offline"
    return wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode=mode,
    )


class RunWriter:
    """Belt-and-braces local persistence for a single run:
      <results_dir>/<bench>/<run_name>_<ts>.jsonl        one record per generated sample (full text)
      <results_dir>/<bench>/<run_name>_<ts>.csv          one row per problem
      <results_dir>/<bench>/<run_name>_<ts>_metrics.json aggregate metrics
    """

    def __init__(self, results_dir, bench_name, run_name, csv_fieldnames):
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(results_dir, bench_name)
        os.makedirs(out_dir, exist_ok=True)
        self.stem = os.path.join(out_dir, f"{run_name}_{ts}")
        self.jsonl_path = self.stem + ".jsonl"
        self.csv_path = self.stem + ".csv"
        self.metrics_path = self.stem + "_metrics.json"
        self._jsonl = open(self.jsonl_path, "w")
        self._csvf = open(self.csv_path, "w", newline="")
        self._csv = csv.DictWriter(self._csvf, fieldnames=csv_fieldnames)
        self._csv.writeheader()

    def add_sample(self, record):
        self._jsonl.write(json.dumps(record, default=str) + "\n")
        self._jsonl.flush()

    def add_problem(self, row):
        self._csv.writerow(row)
        self._csvf.flush()

    def finish(self, metrics):
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self._jsonl.close()
        self._csvf.close()
