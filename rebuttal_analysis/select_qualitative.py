"""Select ~3 compelling qualitative problem-batches for the OpenReview response.

Looks for problems (HumanEval preferred) where, at the same temperature:
  * the BASELINE batch collapses: >= --collapse-count of its samples are
    near-identical (pairwise normalized token edit distance <= 0.1) and the
    batch has low embedding Vendi score, while
  * the ODD batch (at --alpha) contains >= --min-distinct structurally distinct
    solutions (normalized edit distance > 0.3 between cluster representatives),
    ideally >= 2 of them correct.

Emits a markdown snippet with truncated code blocks: the collapsed baseline
mode plus one representative per ODD cluster, annotated with pass/fail and the
batch-level stats.

Usage (after download_text_tables.py):
    python rebuttal_analysis/select_qualitative.py \
        --data-dir rebuttal_analysis/data --out rebuttal_analysis/qualitative_examples.md
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diversity_metrics import (  # noqa: E402
    embedding_vendi,
    get_embedder,
    n_distinct_at_threshold,
    pairwise_edit_matrix,
)
from table_utils import iter_runs  # noqa: E402


def cluster_labels(norm_dist: np.ndarray, threshold: float):
    """Connected-component labels where distance <= threshold means same."""
    n = norm_dist.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if norm_dist[i, j] <= threshold:
                parent[find(i)] = find(j)
    roots = {}
    labels = []
    for i in range(n):
        r = find(i)
        labels.append(roots.setdefault(r, len(roots)))
    return labels


def batch_view(g):
    texts = [str(t) for t in g["text"].tolist()]
    flags = [bool(c) for c in g["correct"].tolist()]
    return texts, flags


def truncate(code: str, max_lines: int, max_chars: int) -> str:
    lines = str(code).rstrip().splitlines()
    out = "\n".join(lines[:max_lines])
    if len(out) > max_chars:
        out = out[:max_chars] + " ..."
    elif len(lines) > max_lines:
        out += "\n# ... (truncated)"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="qualitative_examples.md")
    ap.add_argument("--benchmark", default="humaneval",
                    help="benchmark tag to prefer (humaneval / gsm8k)")
    ap.add_argument("--alpha", type=float, default=16.0, help="ODD alpha to use")
    ap.add_argument("--temperature", type=float, default=None,
                    help="restrict to one temperature (default: all)")
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--collapse-count", type=int, default=12,
                    help="baseline collapse: >= this many near-identical samples out of 16")
    ap.add_argument("--min-distinct", type=int, default=3,
                    help="ODD must have >= this many structurally distinct solutions")
    ap.add_argument("--embedder", choices=["minilm", "hash"], default="minilm")
    ap.add_argument("--max-lines", type=int, default=14)
    ap.add_argument("--max-chars", type=int, default=900)
    args = ap.parse_args()

    embedder = get_embedder(args.embedder)

    # (temperature, alg) -> problem_id -> {"baseline": [(texts, flags)], "odd": [...]}
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    prompts = {}

    for path, meta, df in iter_runs(args.data_dir):
        if meta.get("benchmark") != args.benchmark:
            continue
        cfg = meta["config"]
        scls = cfg.get("strategy_class")
        if scls == "odd" and cfg.get("alpha") != args.alpha:
            continue
        if scls not in ("odd", "baseline"):
            continue
        temp = cfg.get("temperature")
        if args.temperature is not None and temp != args.temperature:
            continue
        for _bid, g in df.groupby("batch_id"):
            pid = str(g["problem_id"].iloc[0])
            index[(temp, cfg.get("alg"))][pid][scls].append(batch_view(g))
            prompts.setdefault(pid, str(g["ref"].iloc[0]))

    candidates = []
    for (temp, alg), problems in index.items():
        for pid, sides in problems.items():
            if not sides["baseline"] or not sides["odd"]:
                continue
            b_texts, b_flags = sides["baseline"][0]
            o_texts, o_flags = sides["odd"][0]

            _, b_norm = pairwise_edit_matrix(b_texts)
            b_labels = cluster_labels(b_norm, 0.1)
            b_mode_size = Counter(b_labels).most_common(1)[0][1]
            if b_mode_size < args.collapse_count:
                continue

            _, o_norm = pairwise_edit_matrix(o_texts)
            o_distinct = n_distinct_at_threshold(o_norm, 0.3)
            if o_distinct < args.min_distinct:
                continue

            correct_texts = [t for t, c in zip(o_texts, o_flags) if c]
            n_dc = 0
            if len(correct_texts) >= 2:
                _, cn = pairwise_edit_matrix(correct_texts)
                n_dc = n_distinct_at_threshold(cn, 0.3)
            elif correct_texts:
                n_dc = 1

            b_vendi = embedding_vendi(b_texts, embedder)
            o_vendi = embedding_vendi(o_texts, embedder)
            score = (n_dc >= 2) * 100 + o_distinct * 5 + b_mode_size + (o_vendi - b_vendi)
            candidates.append({
                "pid": pid, "temp": temp, "alg": alg, "score": score,
                "b_texts": b_texts, "b_flags": b_flags, "b_labels": b_labels,
                "b_mode_size": b_mode_size, "b_vendi": b_vendi,
                "o_texts": o_texts, "o_flags": o_flags, "o_norm": o_norm,
                "o_distinct": o_distinct, "o_vendi": o_vendi, "n_dc": n_dc,
            })

    candidates.sort(key=lambda c: -c["score"])
    picks = candidates[: args.n_examples]
    if not picks:
        print("No batch met the strict criteria; relax --collapse-count/--min-distinct "
              "or check --alpha/--temperature coverage in the downloaded data.")

    lang = "python" if args.benchmark == "humaneval" else "text"
    lines = [f"# Qualitative examples: baseline collapse vs ODD ({args.benchmark}, "
             f"alpha={args.alpha:g})", ""]
    for ex in picks:
        lines.append(f"## {ex['pid']}  (T={ex['temp']}, "
                     f"baseline Vendi={ex['b_vendi']:.2f}, ODD Vendi={ex['o_vendi']:.2f})")
        lines.append("")
        lines.append(f"Baseline: {ex['b_mode_size']}/{len(ex['b_texts'])} samples "
                     f"near-identical (norm. edit dist <= 0.1). Dominant mode "
                     f"({'passes' if any(f for f, l in zip(ex['b_flags'], ex['b_labels']) if l == Counter(ex['b_labels']).most_common(1)[0][0]) else 'fails'}):")
        mode_label = Counter(ex["b_labels"]).most_common(1)[0][0]
        mode_idx = ex["b_labels"].index(mode_label)
        lines.append(f"```{lang}")
        lines.append(truncate(ex["b_texts"][mode_idx], args.max_lines, args.max_chars))
        lines.append("```")
        lines.append("")
        lines.append(f"ODD: {ex['o_distinct']} structurally distinct solutions "
                     f"(norm. edit dist > 0.3), {ex['n_dc']} distinct AND correct. "
                     f"One representative per cluster:")
        o_labels = cluster_labels(ex["o_norm"], 0.3)
        shown = set()
        for i, lab in enumerate(o_labels):
            if lab in shown:
                continue
            shown.add(lab)
            # prefer a correct member as the representative
            members = [j for j, l2 in enumerate(o_labels) if l2 == lab]
            rep = next((j for j in members if ex["o_flags"][j]), members[0])
            status = "PASS" if ex["o_flags"][rep] else "fail"
            lines.append(f"**Cluster {lab + 1}** ({len(members)} samples, {status}):")
            lines.append(f"```{lang}")
            lines.append(truncate(ex["o_texts"][rep], args.max_lines, args.max_chars))
            lines.append("```")
        lines.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(picks)} examples ({len(candidates)} candidates) -> {args.out}")


if __name__ == "__main__":
    main()
