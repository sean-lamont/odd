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
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 1.0],
                        help="Sampling temperatures to sweep")
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
    with hydra.initialize(version_base=None, config_path="conf"):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


# ---------------------------------------------------------------------------
# Model / generator construction (lazy heavy imports)
# ---------------------------------------------------------------------------

def load_shared_resources(cfg):
    """Load the (4-bit) model once, plus the sentence-transformer used for the
    diversity metric. Mirrors the global setup of sweep_human_eval.py so memory
    behaviour is identical."""
    from sentence_transformers import SentenceTransformer

    from utils import load_model

    model, tokenizer, embedding_matrix, mask_token_id = load_model(cfg)
    eval_model = SentenceTransformer("all-MiniLM-L6-v2")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "embedding_matrix": embedding_matrix,
        "mask_token_id": mask_token_id,
        "eval_model": eval_model,
    }


def build_generator(cfg, shared):
    """Build FeatureExtractor + strategy + DiverseGenerator for one grid combo,
    exactly as sweep_human_eval.py does inside its objective()."""
    from feature_extractor import FeatureExtractor
    from generator import DiverseGenerator
    from strategies import get_strategy

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
    return DiverseGenerator(shared["model"], shared["tokenizer"], strategy, shared["mask_token_id"])


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
