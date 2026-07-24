"""Shared helpers for loading, normalising and batching wandb results tables.

Table schemas seen in this project (see sweep_gsm8k.py / sweep_human_eval.py /
sweep_dream.py):

  GSM8K (LLaDA):      ["question", "gold", "generated", "is_correct", "diversity"]
  HumanEval (LLaDA):  ["task_id", "prompt", "completion", "result", "passed", "diversity"]
  Dream (both tasks): ["question/task_id", "gold/prompt", "generated", "is_correct", "diversity"]

All are normalised to canonical columns (originals are kept as well):

  problem_id        question / task_id
  ref               gold answer (GSM8K) or prompt (HumanEval)
  text              generated sample / completion
  correct           bool
  result            HumanEval harness result string (None elsewhere)
  diversity_logged  per-batch diversity score logged at generation time

Batch structure
---------------
Rows are logged one problem-batch at a time: for each problem, ``batch_size``
consecutive rows (one per sample), in generation order.  We do NOT blindly chunk
by ``batch_size``; instead batch boundaries are recovered from the data and then
*verified* against the configured batch size:

  * New-format tables (all cloud sweep runs) repeat the problem id on every row,
    so a change in ``problem_id`` marks a batch boundary.
  * Old-format tables (e.g. local wandb run dirs on the box) only fill the
    problem id on the FIRST row of each batch; subsequent rows contain "".
    We forward-fill and additionally use the ``diversity`` column, which is
    constant within a batch, as a boundary signal.

``assign_batch_ids`` implements this and ``verify_batches`` checks the result
against the run config (all batches of size == cfg batch_size, constant
diversity within batch), so any run violating the assumption is flagged rather
than silently mis-grouped.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from typing import Iterator, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

_CANONICAL_MAP = {
    "question": "problem_id",
    "task_id": "problem_id",
    "question/task_id": "problem_id",
    "gold": "ref",
    "prompt": "ref",
    "gold/prompt": "ref",
    "generated": "text",
    "completion": "text",
    "is_correct": "correct",
    "passed": "correct",
    "result": "result",
    "diversity": "diversity_logged",
}


def normalize_table(columns: list, data: list) -> pd.DataFrame:
    """Build a DataFrame from a wandb .table.json payload with canonical columns.

    Original columns are preserved; canonical aliases are added.
    """
    df = pd.DataFrame(data, columns=columns)
    for orig, canon in _CANONICAL_MAP.items():
        if orig in df.columns and canon not in df.columns:
            df[canon] = df[orig]
    if "result" not in df.columns:
        df["result"] = None
    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(bool)
    return df


def benchmark_from_columns(columns: list) -> str:
    """Best-effort benchmark tag from the raw table schema."""
    if "task_id" in columns:
        return "humaneval"
    if "question" in columns:
        return "gsm8k"
    return "unknown"  # Dream tables: benchmark comes from run config ("task")


# ---------------------------------------------------------------------------
# Batch recovery / verification
# ---------------------------------------------------------------------------

def assign_batch_ids(df: pd.DataFrame, batch_size: Optional[int] = None) -> pd.DataFrame:
    """Add a ``batch_id`` column grouping consecutive rows of one problem-batch.

    Boundary rules (in priority order):
      1. a non-empty ``problem_id`` different from the current one starts a new
         batch (new-format tables);
      2. if ``batch_size`` is known, a full batch is closed after ``batch_size``
         rows (safety net; also handles a problem repeated back-to-back);
      3. a change in ``diversity_logged`` starts a new batch (old-format tables
         where problem_id is only on the first row of a batch).

    ``problem_id`` is forward-filled across empty cells afterwards.
    """
    batch_ids = []
    cur_pid = None
    cur_div = None
    cur_len = 0
    bid = -1
    filled_pids = []
    for pid, div in zip(df["problem_id"].tolist(), df["diversity_logged"].tolist()):
        pid_empty = pid is None or (isinstance(pid, str) and pid.strip() == "")
        new_batch = False
        if bid < 0:
            new_batch = True
        elif not pid_empty and pid != cur_pid:
            new_batch = True
        elif batch_size and cur_len >= batch_size:
            new_batch = True
        elif pid_empty and div is not None and cur_div is not None and div != cur_div:
            new_batch = True
        if new_batch:
            bid += 1
            cur_len = 0
            cur_pid = None if pid_empty else pid
        elif not pid_empty:
            cur_pid = pid
        cur_div = div
        cur_len += 1
        batch_ids.append(bid)
        filled_pids.append(cur_pid)
    out = df.copy()
    out["batch_id"] = batch_ids
    out["problem_id"] = filled_pids
    return out


def verify_batches(df: pd.DataFrame, batch_size: Optional[int] = None) -> dict:
    """Sanity-check recovered batches; returns stats and a list of warnings."""
    warnings = []
    sizes = df.groupby("batch_id").size()
    size_counts = sizes.value_counts().to_dict()
    if batch_size:
        bad = int((sizes != batch_size).sum())
        if bad:
            warnings.append(
                f"{bad}/{len(sizes)} batches do not have the configured size {batch_size} "
                f"(size histogram: {size_counts})"
            )
    nunique_div = df.groupby("batch_id")["diversity_logged"].nunique(dropna=False)
    n_incons = int((nunique_div > 1).sum())
    if n_incons:
        warnings.append(f"{n_incons}/{len(nunique_div)} batches have non-constant diversity")
    nunique_pid = df.groupby("batch_id")["problem_id"].nunique(dropna=True)
    n_mixed = int((nunique_pid > 1).sum())
    if n_mixed:
        warnings.append(f"{n_mixed}/{len(nunique_pid)} batches mix multiple problem ids")
    return {
        "n_rows": int(len(df)),
        "n_batches": int(sizes.shape[0]),
        "size_histogram": {int(k): int(v) for k, v in size_counts.items()},
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Run config parsing (LLaDA nested strategy dict vs Dream flat config)
# ---------------------------------------------------------------------------

# Strategy-name conventions in the cloud runs:
#   LLaDA:  'batched_orth' = ODD, 'joint' = DPP, 'baseline' = baseline
#           ('orthogonal_projection' = an earlier ODD variant, excluded from
#            headline comparisons)
#   Dream:  'odd' = ODD, 'baseline' = baseline  (config also has 'alg';
#            the paper used alg='maskgit_plus')
ODD_NAMES = {"batched_orth", "odd"}
DPP_NAMES = {"joint"}
BASELINE_NAMES = {"baseline"}


def strategy_class(name: str) -> str:
    if name in ODD_NAMES:
        return "odd"
    if name in DPP_NAMES:
        return "dpp"
    if name in BASELINE_NAMES:
        return "baseline"
    return name


def parse_run_config(config: dict) -> dict:
    """Extract the fields we care about from a wandb run config dict.

    Handles both LLaDA sweeps (nested ``strategy`` dict, possibly stringified)
    and Dream sweeps (flat ``strategy``/``alpha``/``alg`` fields).
    """
    import ast as _ast

    strategy = config.get("strategy", {})
    if isinstance(strategy, str):
        try:
            strategy = _ast.literal_eval(strategy)
        except (ValueError, SyntaxError):
            # Dream style: strategy is a plain name string
            strategy = {"name": strategy}
    if not isinstance(strategy, dict):
        strategy = {"name": str(strategy)}

    name = strategy.get("name", strategy.get("type", "unknown"))
    alpha = strategy.get("alpha", config.get("alpha", None))
    model = config.get("model", None)
    if isinstance(model, dict):
        model = model.get("name")
    return {
        "strategy_name": str(name),
        "strategy_class": strategy_class(str(name)),
        "alpha": float(alpha) if alpha is not None else None,
        "temperature": _to_float(config.get("temperature")),
        "model": model,
        "alg": config.get("alg", None),           # Dream only
        "task": config.get("task", None),         # Dream only
        "batch_size": _to_int(config.get("batch_size")),
        "n_problems": _to_int(config.get("n_problems")),
        "steps": _to_int(config.get("steps")),
        "gen_length": _to_int(config.get("gen_length")),
    }


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Downloaded-run IO (jsonl.gz; first line is a meta record)
# ---------------------------------------------------------------------------

def run_output_path(out_dir: str, project: str, run_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{project}__{run_id}")
    return os.path.join(out_dir, safe + ".jsonl.gz")


def save_run(path: str, meta: dict, df: pd.DataFrame) -> None:
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, default=str) + "\n")
    os.replace(tmp, path)  # atomic -> resumability-safe


def load_run(path: str):
    """Return (meta, DataFrame) for one downloaded run file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = json.loads(f.readline())
        meta = first["_meta"]
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    if "correct" in df.columns:
        df["correct"] = df["correct"].map(
            lambda v: v if isinstance(v, bool) else str(v).lower() == "true"
        )
    return meta, df


def iter_runs(data_dir: str,
              projects: Optional[list] = None,
              strategy: Optional[str] = None,
              alpha: Optional[float] = None,
              temperature: Optional[float] = None) -> Iterator[tuple]:
    """Yield (path, meta, df) for downloaded runs matching the filters.

    ``strategy`` matches either the raw strategy name or its class
    ('odd'/'dpp'/'baseline').
    """
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith(".jsonl.gz"):
            continue
        path = os.path.join(data_dir, fn)
        meta, df = load_run(path)
        cfg = meta.get("config", {})
        if projects and meta.get("project") not in projects:
            continue
        if strategy and strategy not in (cfg.get("strategy_name"), cfg.get("strategy_class")):
            continue
        if alpha is not None and cfg.get("alpha") != float(alpha):
            continue
        if temperature is not None and cfg.get("temperature") != float(temperature):
            continue
        yield path, meta, df
