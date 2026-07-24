"""Alpha robustness / transfer analysis for the rebuttal.

Question addressed: "does ODD's alpha need per-setting tuning?"  Answer: a
single fixed alpha=16 gains over baseline at EVERY (model, task, temperature),
and an alpha tuned on any one (model, task) transfers to every other with small
regret vs the per-setting oracle.

Inputs (aggregate run CSVs exported via analyse_results/download_runs.py /
new_results/download_data.py):
  * LLaDA:  gsm8k.csv / humaneval.csv — ``strategy`` column is a stringified
    dict {name, alpha, ...}; ODD='batched_orth', baseline='baseline'.
  * Dream:  dream_gsm8k_eval.csv / dream_humaneval_eval.csv — flat ``strategy``
    ('odd'/'baseline'), ``alpha`` and ``alg`` columns; filtered to
    alg='maskgit_plus' (the setting used in the paper).

Defaults: LLaDA CSVs from the in-repo analyse_results/; Dream CSVs live in the
supplementary bundle OUTSIDE this repo, so --dream-csv-dir is required unless
the conventional ../../supp/new_results location exists.

Metric: pass@16.  Gains reported in percentage points (x100).

Usage:
    python rebuttal_analysis/alpha_transfer.py \
        --dream-csv-dir ../supp/new_results \
        --out rebuttal_analysis/alpha_transfer_results.md
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import statistics as st
import sys
from collections import defaultdict

ALPHAS = [2, 8, 16, 32, 64, 128]
FIXED_ALPHA = 16.0
ODD_NAMES = {"batched_orth", "odd"}

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LLADA_DIR = os.path.normpath(os.path.join(_HERE, "..", "analyse_results"))
DEFAULT_DREAM_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "supp", "new_results"))


def load_llada(path):
    """LLaDA CSVs: strategy is a stringified dict."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("pass_at_16") or not r.get("temperature"):
                continue
            try:
                s = ast.literal_eval(r["strategy"])
            except (ValueError, SyntaxError):
                continue
            rows.append(dict(
                sname=s.get("name", "unknown"),
                alpha=float(s.get("alpha") or 0),
                temp=float(r["temperature"]),
                p16=float(r["pass_at_16"]),
            ))
    return rows


def load_dream(path, alg="maskgit_plus"):
    """Dream CSVs: flat strategy/alpha/alg columns; keep only the paper's alg."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("alg") != alg:
                continue
            if not r.get("pass_at_16") or not r.get("temperature"):
                continue
            rows.append(dict(
                sname=r["strategy"],
                alpha=float(r.get("alpha") or 0),
                temp=float(r["temperature"]),
                p16=float(r["pass_at_16"]),
            ))
    return rows


def aggregate(rows):
    """(class, alpha, temp) -> mean pass@16 over repeat runs."""
    g = defaultdict(list)
    for r in rows:
        cls = "odd" if r["sname"] in ODD_NAMES else r["sname"]
        g[(cls, r["alpha"], r["temp"])].append(r["p16"])
    return {k: st.mean(v) for k, v in g.items()}


def temps_of(m):
    return sorted({t for (c, _a, t) in m if c == "odd"})


def baseline_of(m, temps):
    return {t: m.get(("baseline", 0.0, t)) for t in temps}


def gains_for_alpha(m, a, temps=None):
    """Per-temp gains (pts) of ODD at alpha=a over baseline."""
    temps = temps or temps_of(m)
    base = baseline_of(m, temps)
    return [
        (m[("odd", float(a), t)] - base[t]) * 100
        for t in temps
        if ("odd", float(a), t) in m and base[t] is not None
    ]


def best_alpha(m):
    """Alpha maximizing mean gain over baseline across temps (the 'tuned' alpha)."""
    best, ba = -1e9, None
    for a in ALPHAS:
        g = gains_for_alpha(m, a)
        if g and st.mean(g) > best:
            best, ba = st.mean(g), a
    return ba


def oracle_per_temp(m, temps):
    """Best ODD pass@16 over alphas, per temp (the per-setting oracle)."""
    return {
        t: max(m[("odd", float(a), t)] for a in ALPHAS if ("odd", float(a), t) in m)
        for t in temps
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llada-csv-dir", default=DEFAULT_LLADA_DIR,
                    help=f"dir with gsm8k.csv / humaneval.csv (default: {DEFAULT_LLADA_DIR})")
    ap.add_argument("--dream-csv-dir", default=None,
                    help="dir with dream_gsm8k_eval.csv / dream_humaneval_eval.csv "
                         "(supplementary bundle, outside this repo; default tries ../../supp/new_results)")
    ap.add_argument("--dream-alg", default="maskgit_plus")
    ap.add_argument("--out", default=None, help="write the markdown report here")
    args = ap.parse_args()

    dream_dir = args.dream_csv_dir or (
        DEFAULT_DREAM_DIR if os.path.isdir(DEFAULT_DREAM_DIR) else None)
    if dream_dir is None:
        sys.exit("Dream CSVs not found: pass --dream-csv-dir (they live in the "
                 "supplementary bundle, e.g. <supp>/new_results).")

    files = {
        ("LLaDA", "GSM8K"): (os.path.join(args.llada_csv_dir, "gsm8k.csv"), load_llada),
        ("LLaDA", "HumanEval"): (os.path.join(args.llada_csv_dir, "humaneval.csv"), load_llada),
        ("Dream", "GSM8K"): (os.path.join(dream_dir, "dream_gsm8k_eval.csv"),
                             lambda p: load_dream(p, args.dream_alg)),
        ("Dream", "HumanEval"): (os.path.join(dream_dir, "dream_humaneval_eval.csv"),
                                 lambda p: load_dream(p, args.dream_alg)),
    }

    A = {}
    for key, (path, loader) in files.items():
        rows = loader(path)
        A[key] = aggregate(rows)
        print(f"{key}: {len(rows)} runs, strategies {sorted({r['sname'] for r in rows})}")

    md = ["# Alpha robustness and transfer (pass@16, percentage points)", ""]
    md.append(f"Dream runs filtered to `alg={args.dream_alg}` (paper setting). "
              "Gains are ODD minus baseline at the same temperature; each cell "
              "averages over the temperature grid {0.0, 0.5, 1.0, 1.5, 2.0}.")
    md.append("")

    # --- fixed alpha = 16 ---------------------------------------------------
    md.append(f"## Fixed alpha = {FIXED_ALPHA:g} (no per-setting tuning)")
    md.append("")
    md.append("| Model | Task | mean gain (pts) | min gain (pts) | oracle alpha per temp | regret vs oracle mean/max (pts) |")
    md.append("|---|---|---|---|---|---|")
    for key, m in A.items():
        temps = temps_of(m)
        g16 = gains_for_alpha(m, FIXED_ALPHA, temps)
        oracle = oracle_per_temp(m, temps)
        oracle_alpha = {
            t: max((m[("odd", float(a), t)], a) for a in ALPHAS if ("odd", float(a), t) in m)[1]
            for t in temps
        }
        fixed = {t: m.get(("odd", FIXED_ALPHA, t)) for t in temps}
        reg = [(oracle[t] - fixed[t]) * 100 for t in temps if fixed[t] is not None]
        md.append(
            f"| {key[0]} | {key[1]} | {st.mean(g16):+.2f} | {min(g16):+.2f} | "
            f"{', '.join(f'{t:g}->{oracle_alpha[t]}' for t in temps)} | "
            f"{st.mean(reg):.2f} / {max(reg):.2f} |"
        )
        print(f"{key}: alpha=16 mean gain {st.mean(g16):+.2f}, min {min(g16):+.2f}, "
              f"regret mean {st.mean(reg):.2f} max {max(reg):.2f}")
    md.append("")

    # --- 12-pair transfer matrix -------------------------------------------
    md.append("## Cross-(model, task) transfer of a single tuned alpha")
    md.append("")
    md.append("Alpha is tuned on the SOURCE setting (argmax of mean gain over its "
              "temperature grid) and applied unchanged to the TARGET setting "
              "(12 source-target pairs).")
    md.append("")
    md.append("| Source (alpha*) | Target | mean gain (pts) | min gain (pts) | mean regret vs target oracle (pts) |")
    md.append("|---|---|---|---|---|")
    keys = list(files)
    all_means, all_mins, all_regrets = [], [], []
    for src in keys:
        a = best_alpha(A[src])
        for tgt in keys:
            if src == tgt:
                continue
            m = A[tgt]
            temps = temps_of(m)
            g = gains_for_alpha(m, a, temps)
            oracle = oracle_per_temp(m, temps)
            reg = [
                (oracle[t] - m[("odd", float(a), t)]) * 100
                for t in temps if ("odd", float(a), t) in m
            ]
            all_means.append(st.mean(g))
            all_mins.append(min(g))
            all_regrets.append(st.mean(reg))
            md.append(
                f"| {src[0]}/{src[1]} (a*={a}) | {tgt[0]}/{tgt[1]} | "
                f"{st.mean(g):+.2f} | {min(g):+.2f} | {st.mean(reg):.2f} |"
            )
    grand_mean = st.mean(all_means)
    md.append("")
    md.append(f"**Across all 12 transfer pairs: mean gain {grand_mean:+.2f} pts "
              f"(worst pair mean {min(all_means):+.2f}; worst single temperature "
              f"{min(all_mins):+.2f}; mean regret vs per-setting oracle "
              f"{st.mean(all_regrets):.2f} pts).**")
    md.append("")
    md.append("*Takeaway: alpha transfers across models (LLaDA <-> Dream) and tasks "
              "(GSM8K <-> HumanEval) — ODD does not require per-setting hyperparameter "
              "tuning to beat the baseline.*")

    report = "\n".join(md)
    print("\n" + report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
