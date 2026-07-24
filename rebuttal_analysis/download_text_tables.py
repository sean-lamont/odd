"""Download per-run ``results_table`` artifacts from wandb, KEEPING the full
generation text (unlike analyse_results/download_tables.py, which aggregates to
pass counts and drops the text).

One ``<project>__<run_id>.jsonl.gz`` file is written per run into --out:
line 1 is a meta record (project, run id/name/group, parsed config: strategy
name, alpha, temperature, model, alg, batch_size, n_problems, plus batch
verification stats), each following line is one sample row with all original
table columns, canonical aliases (problem_id/ref/text/correct/result/
diversity_logged) and a recovered ``batch_id``.

Resumable: runs whose output file already exists are skipped.

Run this on the box with the logged-in wandb credential:

    python rebuttal_analysis/download_text_tables.py --out rebuttal_analysis/data

Pull a small subset first, e.g.:

    python rebuttal_analysis/download_text_tables.py --out rebuttal_analysis/data \
        --projects sean-a-lamont/odd_humaneval --filter-strategy baseline --max-runs 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from table_utils import (  # noqa: E402
    assign_batch_ids,
    benchmark_from_columns,
    normalize_table,
    parse_run_config,
    run_output_path,
    save_run,
    verify_batches,
)

# Projects holding the paper's runs, verified against the aggregate CSVs
# (run counts + strategy composition match exactly: 757 / 759 / 202 / 203).
# NOTE: the sweep scripts say project="gsm8k"/"humaneval", but on the wandb
# default entity (tactic-zero) those names hold only partial joint-only runs;
# the full LLaDA sweeps live under the sean-a-lamont entity as odd_gsm8k /
# odd_humaneval. A project may be given as "entity/project" to override
# --entity per project.
DEFAULT_PROJECTS = [
    "sean-a-lamont/odd_gsm8k",
    "sean-a-lamont/odd_humaneval",
    "tactic-zero/dream_gsm8k_eval",
    "tactic-zero/dream_humaneval_eval",
]
TABLE_KEY = "results_table"


def find_table_json(run, table_key: str):
    """Download the run's results table artifact; return path to .table.json."""
    target = None
    for artifact in run.logged_artifacts():
        if table_key in artifact.name:
            target = artifact
            break
    if target is None:
        return None
    table_dir = target.download()
    for root, _dirs, files in os.walk(table_dir):
        for fn in files:
            if fn.endswith(".table.json"):
                return os.path.join(root, fn)
    return None


def process_run(run, project: str, out_dir: str, table_key: str) -> str:
    out_path = run_output_path(out_dir, project, run.id)
    if os.path.exists(out_path):
        return "skipped (exists)"

    cfg = parse_run_config(run.config)
    json_path = find_table_json(run, table_key)
    if json_path is None:
        return "no results_table artifact"

    with open(json_path) as f:
        table = json.load(f)
    df = normalize_table(table["columns"], table["data"])
    df = assign_batch_ids(df, batch_size=cfg.get("batch_size"))
    batch_stats = verify_batches(df, batch_size=cfg.get("batch_size"))
    for w in batch_stats["warnings"]:
        print(f"    WARNING [{run.name}]: {w}")

    benchmark = cfg.get("task") or benchmark_from_columns(table["columns"])
    if benchmark == "unknown":
        benchmark = project

    meta = {
        "project": project,
        "run_id": run.id,
        "run_name": run.name,
        "group": getattr(run, "group", None),
        "benchmark": benchmark,
        "config": cfg,
        "table_columns": table["columns"],
        "batch_stats": batch_stats,
    }
    save_run(out_path, meta, df)
    return f"saved {batch_stats['n_batches']} batches / {batch_stats['n_rows']} rows -> {out_path}"


def matches_filters(cfg: dict, args) -> bool:
    if args.filter_strategy is not None and args.filter_strategy not in (
        cfg.get("strategy_name"), cfg.get("strategy_class")
    ):
        return False
    if args.filter_alpha is not None and cfg.get("alpha") != args.filter_alpha:
        return False
    if args.filter_temp is not None and cfg.get("temperature") != args.filter_temp:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=None,
                    help="wandb entity for bare project names (default: api.default_entity); "
                         "'entity/project' items in --projects take precedence")
    ap.add_argument("--projects", nargs="+", default=DEFAULT_PROJECTS,
                    help="project names, optionally entity-qualified as entity/project")
    ap.add_argument("--out", required=True, help="output directory for per-run jsonl.gz files")
    ap.add_argument("--filter-strategy", default=None,
                    help="only pull runs with this strategy name or class "
                         "(e.g. batched_orth / odd / joint / dpp / baseline)")
    ap.add_argument("--filter-alpha", type=float, default=None)
    ap.add_argument("--filter-temp", type=float, default=None)
    ap.add_argument("--max-runs", type=int, default=None, help="cap on runs downloaded per project")
    ap.add_argument("--table-key", default=TABLE_KEY)
    args = ap.parse_args()

    import wandb  # lazy: only needed on the machine with the credential

    api = wandb.Api(timeout=60)
    default_entity = args.entity or api.default_entity
    if not default_entity and any("/" not in p for p in args.projects):
        sys.exit("No wandb entity found; pass --entity or `wandb login` first.")
    print(f"Default entity: {default_entity}")
    os.makedirs(args.out, exist_ok=True)

    for spec in args.projects:
        if "/" in spec:
            entity, project = spec.split("/", 1)
        else:
            entity, project = default_entity, spec
        try:
            runs = api.runs(f"{entity}/{project}")
            n_total = len(runs)
        except Exception as e:
            print(f"Could not access {entity}/{project}: {e}")
            continue
        print(f"\n=== {entity}/{project}: {n_total} runs ===")
        n_done = 0
        for run in runs:
            if args.max_runs is not None and n_done >= args.max_runs:
                print(f"  reached --max-runs={args.max_runs}, stopping project")
                break
            cfg = parse_run_config(run.config)
            if not matches_filters(cfg, args):
                continue
            try:
                status = process_run(run, project, args.out, args.table_key)
            except Exception as e:
                status = f"ERROR: {e}"
            if "skipped" not in status:
                n_done += 1
            print(f"  [{run.name} | {cfg.get('strategy_name')} a={cfg.get('alpha')} "
                  f"T={cfg.get('temperature')}] {status}")


if __name__ == "__main__":
    main()
