"""Diversity / correctness metrics over downloaded results tables.

Given a directory of per-run ``.jsonl.gz`` files (from download_text_tables.py),
computes per (benchmark, strategy, alpha, temperature), aggregated over
problems:

  * Vendi score per problem-batch with a cosine-similarity kernel over
    all-MiniLM-L6-v2 embeddings: VS = exp(-sum lam_i log lam_i) where lam are
    the eigenvalues of K/n. Identical texts -> VS = 1; orthogonal -> VS = n.
  * An n-gram Vendi variant (TF-IDF character n-gram kernel) as a lexical
    complement to the semantic embedding kernel.
  * Solution-level joint stats per problem: #correct samples, #DISTINCT correct
    solutions at pairwise normalized token edit distance thresholds
    {0.1, 0.2, 0.3} (two solutions are "the same" iff distance <= t; distinct
    count = number of connected components), for HumanEval also an
    AST-normalized distinct count (parse, strip docstrings/comments, canonical
    ast.unparse, exact-match dedup), and #clusters among ALL samples
    (agglomerative clustering, average linkage, on embedding cosine distance
    with threshold --cluster-threshold, default 0.3).
  * "One-token-flip" refutation stats: among problems where a strategy finds
    >=2 correct solutions, the distribution of pairwise token-level edit
    distances between correct solutions (raw and normalized), incl. the
    fraction of pairs within <=2 raw token edits.
  * Novel-region stats: for problems ODD solves but baseline (same benchmark /
    temperature / alg, samples pooled over baseline runs) never solves, the min
    edit distance from each ODD-correct solution to ANY baseline sample.
  * GSM8K correctness recomputed from text (same extraction logic as
    sweep_gsm8k.py) and compared to the logged is_correct flag.

Outputs in --out-dir:
  per_batch_metrics.csv        one row per (run, problem-batch)
  summary_metrics.csv          tidy aggregate per (benchmark, model, strategy, alpha, temp)
  pairwise_edit_correct.csv    one-token-flip distributions per config
  novelty_vs_baseline.csv      ODD-only-solved distance-to-baseline stats
  summary.md                   markdown comparison at the headline settings

Embedding model is lazy and mockable: --embedder minilm (default; requires
sentence-transformers) or hash (deterministic char-n-gram hashing embedder,
no downloads — for tests/smoke runs).
"""

from __future__ import annotations

import argparse
import ast as pyast
import itertools
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from table_utils import iter_runs  # noqa: E402

EDIT_THRESHOLDS = (0.1, 0.2, 0.3)

# ---------------------------------------------------------------------------
# GSM8K answer extraction — copied from sweep_gsm8k.py (kept in sync by hand;
# importing that module would load the LLaDA model at import time).
# ---------------------------------------------------------------------------

def extract_answer_num(text):
    try:
        text = text.replace(",", "")
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            return float(nums[-1])
    except Exception as e:  # pragma: no cover
        print(e)
    return None


def extract_gold_num(answer_str):
    if "####" in str(answer_str):
        try:
            val = str(answer_str).split("####")[1].strip()
            return float(val.replace(",", ""))
        except Exception:
            pass
    return None


def gsm8k_recheck(texts, gold):
    """Recompute correctness flags from raw text; gold may be a float already
    (the sweeps log extract_gold_num(answer) as ``gold``) or a '#### x' string."""
    g = None
    try:
        g = float(gold)
    except (TypeError, ValueError):
        g = extract_gold_num(gold)
    if g is None:
        return None
    flags = []
    for t in texts:
        v = extract_answer_num(t)
        flags.append(v is not None and abs(v - g) < 1e-4)
    return flags


# ---------------------------------------------------------------------------
# Embedders (lazy / mockable)
# ---------------------------------------------------------------------------

class HashEmbedder:
    """Deterministic, download-free stand-in embedder (char n-gram hashing,
    L2-normalized). Only for tests and smoke runs — NOT for reported numbers."""

    name = "hash"

    def __init__(self, n_features: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), n_features=n_features, norm="l2"
        )

    def encode(self, texts):
        return np.asarray(self.vec.transform(list(texts)).todense())


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 sentence embeddings (same model the paper's diversity
    metric uses, see utils.calculate_diversity_score)."""

    name = "minilm"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts):
        emb = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(emb)


def get_embedder(name: str):
    if name == "hash":
        return HashEmbedder()
    if name == "minilm":
        return MiniLMEmbedder()
    raise ValueError(f"unknown embedder {name!r}")


class CachedEmbedder:
    """Wraps an embedder with a text->vector cache so each unique text in a run
    is encoded once (baseline batches are often mostly duplicates)."""

    def __init__(self, base):
        self.base = base
        self.name = base.name
        self.cache = {}

    def warm(self, texts):
        todo = list({t for t in texts if t not in self.cache})
        if todo:
            vecs = self.base.encode(todo)
            for t, v in zip(todo, vecs):
                self.cache[t] = np.asarray(v)

    def encode(self, texts):
        self.warm(texts)
        return np.stack([self.cache[t] for t in texts])

    def clear(self):
        self.cache.clear()


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# ---------------------------------------------------------------------------
# Vendi score
# ---------------------------------------------------------------------------

def vendi_from_kernel(K: np.ndarray) -> float:
    """VS = exp(von Neumann entropy of eigenvalues of K/n).

    K must be a PSD similarity kernel with unit diagonal (e.g. a cosine Gram
    matrix of L2-normalized features).  Identical items -> 1.0; mutually
    orthogonal items -> n.  (Friedman & Dieng, 2023 — implemented directly with
    numpy, no vendi-score dependency.)
    """
    n = K.shape[0]
    if n == 0:
        return 0.0
    lam = np.linalg.eigvalsh(np.asarray(K, dtype=np.float64) / n)
    lam = np.clip(lam, 0.0, None)
    s = lam.sum()
    if s <= 0:
        return 0.0
    lam = lam / s
    nz = lam[lam > 1e-12]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def embedding_vendi(texts, embedder) -> float:
    E = _l2_normalize(embedder.encode(texts))
    return vendi_from_kernel(E @ E.T)


def ngram_vendi(texts, ngram_range=(2, 4), analyzer="char_wb") -> float:
    """Lexical Vendi: cosine kernel of TF-IDF character n-grams."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [t if t and str(t).strip() else " " for t in texts]
    if len(set(texts)) == 1:
        return 1.0
    try:
        X = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, norm="l2").fit_transform(texts)
    except ValueError:  # empty vocabulary
        return 1.0
    K = np.asarray((X @ X.T).todense())
    np.fill_diagonal(K, 1.0)
    return vendi_from_kernel(K)


# ---------------------------------------------------------------------------
# Token-level edit distance & distinctness
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str):
    return _TOKEN_RE.findall(str(text))


def token_edit_distance(a_tokens, b_tokens) -> int:
    """Levenshtein distance over token sequences (insert/delete/substitute).

    Vectorized row DP: deletion/substitution terms are elementwise over the
    previous row; the insertion term cur[j-1]+1 (a sequential dependency) is
    resolved exactly via a prefix-min of (cur[k] - k) + j.
    """
    la, lb = len(a_tokens), len(b_tokens)
    if la == 0 or lb == 0:
        return la + lb
    # map tokens to int ids for fast comparison
    ids = {}
    a = np.fromiter((ids.setdefault(t, len(ids)) for t in a_tokens), dtype=np.int32, count=la)
    b = np.fromiter((ids.setdefault(t, len(ids)) for t in b_tokens), dtype=np.int32, count=lb)
    prev = np.arange(lb + 1, dtype=np.int32)
    idx = np.arange(lb + 1, dtype=np.int32)
    for i in range(1, la + 1):
        cost = (b != a[i - 1]).astype(np.int32)
        cur = np.empty(lb + 1, dtype=np.int32)
        cur[0] = i
        cur[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)  # delete / substitute
        cur = np.minimum.accumulate(cur - idx) + idx           # insert via prefix-min
        prev = cur
    return int(prev[lb])


def normalized_edit_distance(a: str, b: str):
    """(raw, normalized) token edit distance; normalized by max token length."""
    ta, tb = tokenize(a), tokenize(b)
    raw = token_edit_distance(ta, tb)
    denom = max(len(ta), len(tb), 1)
    return raw, raw / denom


def pairwise_edit_matrix(texts):
    """Symmetric matrices of (raw, normalized) token edit distances."""
    n = len(texts)
    toks = [tokenize(t) for t in texts]
    raw = np.zeros((n, n), dtype=float)
    norm = np.zeros((n, n), dtype=float)
    for i, j in itertools.combinations(range(n), 2):
        d = token_edit_distance(toks[i], toks[j])
        raw[i, j] = raw[j, i] = d
        norm[i, j] = norm[j, i] = d / max(len(toks[i]), len(toks[j]), 1)
    return raw, norm


def n_distinct_at_threshold(norm_dist: np.ndarray, threshold: float) -> int:
    """Connected components where an edge means distance <= threshold
    ("the same solution"); the component count is the number of DISTINCT
    solutions at that threshold."""
    n = norm_dist.shape[0]
    if n == 0:
        return 0
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in itertools.combinations(range(n), 2):
        if norm_dist[i, j] <= threshold:
            parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


# ---------------------------------------------------------------------------
# AST-normalized comparison (HumanEval)
# ---------------------------------------------------------------------------

class _DocstringStripper(pyast.NodeTransformer):
    def _strip(self, node):
        if (node.body and isinstance(node.body[0], pyast.Expr)
                and isinstance(node.body[0].value, pyast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [pyast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip(node)


def ast_normalize(code: str, prompt: str = None):
    """Canonical form of Python code: parse, strip docstrings (comments vanish
    in the AST), ast.unparse.  Completions that only parse when appended to the
    HumanEval prompt are retried as prompt+code.  Returns None if unparseable.
    """
    for src in ([code] + ([str(prompt) + str(code)] if prompt else [])):
        try:
            tree = pyast.parse(src)
        except (SyntaxError, ValueError):
            continue
        tree = _DocstringStripper().visit(tree)
        pyast.fix_missing_locations(tree)
        try:
            return pyast.unparse(tree)
        except Exception:
            continue
    return None


def n_distinct_ast(codes, prompt=None) -> int:
    """#distinct after AST normalization; unparseable codes fall back to
    whitespace-collapsed text so they still count."""
    forms = []
    for c in codes:
        f = ast_normalize(c, prompt=prompt)
        if f is None:
            f = "RAW::" + re.sub(r"\s+", " ", str(c)).strip()
        forms.append(f)
    return len(set(forms))


# ---------------------------------------------------------------------------
# Embedding clustering over ALL samples
# ---------------------------------------------------------------------------

def n_embedding_clusters(texts, embedder, distance_threshold: float = 0.3) -> int:
    """Agglomerative clustering (average linkage) on cosine distance of
    embeddings; returns the number of clusters at the given threshold."""
    from sklearn.cluster import AgglomerativeClustering

    n = len(texts)
    if n < 2:
        return n
    E = _l2_normalize(embedder.encode(texts))
    D = np.clip(1.0 - E @ E.T, 0.0, None)
    np.fill_diagonal(D, 0.0)
    if D.max() <= 1e-9:
        return 1
    cl = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold,
        metric="precomputed", linkage="average",
    ).fit(D)
    return int(cl.n_clusters_)


# ---------------------------------------------------------------------------
# Per-batch metric computation
# ---------------------------------------------------------------------------

def batch_metrics(texts, correct_flags, benchmark, embedder, prompt=None, gold=None,
                  cluster_threshold=0.3, edit_thresholds=EDIT_THRESHOLDS):
    n = len(texts)
    rec = {"n_samples": n, "n_correct": int(sum(correct_flags))}

    rec["vendi_embed"] = embedding_vendi(texts, embedder)
    rec["vendi_ngram"] = ngram_vendi(texts)
    rec["n_clusters_all"] = n_embedding_clusters(texts, embedder, cluster_threshold)

    correct_texts = [t for t, c in zip(texts, correct_flags) if c]
    if len(correct_texts) >= 2:
        raw, norm = pairwise_edit_matrix(correct_texts)
        iu = np.triu_indices(len(correct_texts), k=1)
        rec["_pair_raw"] = raw[iu].tolist()
        rec["_pair_norm"] = norm[iu].tolist()
        for t in edit_thresholds:
            rec[f"n_distinct_correct@{t}"] = n_distinct_at_threshold(norm, t)
    else:
        rec["_pair_raw"] = []
        rec["_pair_norm"] = []
        for t in edit_thresholds:
            rec[f"n_distinct_correct@{t}"] = len(correct_texts)

    if benchmark == "humaneval":
        rec["n_distinct_correct_ast"] = n_distinct_ast(correct_texts, prompt=prompt) \
            if correct_texts else 0

    if benchmark == "gsm8k" and gold is not None:
        re_flags = gsm8k_recheck(texts, gold)
        if re_flags is not None:
            agree = sum(int(a == bool(b)) for a, b in zip(re_flags, correct_flags))
            rec["gsm8k_recheck_agree"] = agree / n
            rec["gsm8k_recheck_n_correct"] = int(sum(re_flags))
    return rec


def _dist_stats(values, prefix):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {f"{prefix}_n_pairs": 0}
    return {
        f"{prefix}_n_pairs": int(v.size),
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_median": float(np.median(v)),
        f"{prefix}_p10": float(np.percentile(v, 10)),
        f"{prefix}_p90": float(np.percentile(v, 90)),
        f"{prefix}_min": float(v.min()),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def config_key(meta):
    cfg = meta["config"]
    return (
        meta.get("benchmark"),
        cfg.get("model") or ("Dream" if cfg.get("alg") else "unknown"),
        cfg.get("alg"),
        cfg.get("strategy_class"),
        cfg.get("alpha"),
        cfg.get("temperature"),
    )


KEY_COLS = ["benchmark", "model", "alg", "strategy", "alpha", "temperature"]


def process_all(data_dir, out_dir, embedder_name="minilm", cluster_threshold=0.3,
                headline_alpha=16.0, max_runs=None, projects=None, alphas=None):
    """``alphas``: if given, only process runs whose alpha is in the list
    (baseline runs are always kept)."""
    os.makedirs(out_dir, exist_ok=True)
    embedder = CachedEmbedder(get_embedder(embedder_name))

    per_batch_rows = []
    # For novelty: (benchmark, temperature, alg) -> problem -> list of baseline texts
    baseline_pool = defaultdict(lambda: defaultdict(list))
    odd_correct = defaultdict(lambda: defaultdict(list))  # same key -> problem -> correct texts
    baseline_solved = defaultdict(lambda: defaultdict(bool))

    n_runs = 0
    for path, meta, df in iter_runs(data_dir, projects=projects):
        cfg = meta["config"]
        if alphas is not None and cfg.get("strategy_class") != "baseline" \
                and cfg.get("alpha") not in [float(a) for a in alphas]:
            continue
        if max_runs is not None and n_runs >= max_runs:
            break
        n_runs += 1
        bench = meta.get("benchmark")
        key = config_key(meta)
        pool_key = (bench, cfg.get("temperature"), cfg.get("alg"))
        t0 = time.time()
        embedder.clear()  # cache per run: dedupe within, bound memory across
        embedder.warm([str(t) for t in df["text"].tolist()])
        print(f"[{n_runs}] {os.path.basename(path)}  {key}  "
              f"({len(embedder.cache)} unique / {len(df)} texts)", flush=True)

        for bid, g in df.groupby("batch_id"):
            texts = [str(t) for t in g["text"].tolist()]
            flags = [bool(c) for c in g["correct"].tolist()]
            pid = g["problem_id"].iloc[0]
            ref = g["ref"].iloc[0]
            prompt = ref if bench == "humaneval" else None
            gold = ref if bench == "gsm8k" else None
            rec = batch_metrics(texts, flags, bench, embedder, prompt=prompt, gold=gold,
                                cluster_threshold=cluster_threshold)
            pair_raw = rec.pop("_pair_raw")
            pair_norm = rec.pop("_pair_norm")
            rec.update({
                "benchmark": key[0], "model": key[1], "alg": key[2],
                "strategy": key[3], "alpha": key[4], "temperature": key[5],
                "project": meta["project"], "run_id": meta["run_id"],
                "problem_id": str(pid)[:120], "batch_id": bid,
                "diversity_logged": g["diversity_logged"].iloc[0],
                "pair_raw_edits": ";".join(f"{x:.0f}" for x in pair_raw),
                "pair_norm_edits": ";".join(f"{x:.4f}" for x in pair_norm),
            })
            per_batch_rows.append(rec)

            if cfg.get("strategy_class") == "baseline":
                baseline_pool[pool_key][str(pid)].extend(texts)
                if any(flags):
                    baseline_solved[pool_key][str(pid)] = True
            elif cfg.get("strategy_class") == "odd":
                ct = [t for t, c in zip(texts, flags) if c]
                if ct:
                    odd_correct[pool_key][(meta["run_id"], key[4], str(pid))] = ct
        print(f"    done in {time.time() - t0:.1f}s", flush=True)

    per_batch = pd.DataFrame(per_batch_rows)
    per_batch.to_csv(os.path.join(out_dir, "per_batch_metrics.csv"), index=False)
    print(f"wrote per_batch_metrics.csv ({len(per_batch)} rows)")

    summary = summarize(per_batch)
    summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)

    pairwise = pairwise_summary(per_batch)
    pairwise.to_csv(os.path.join(out_dir, "pairwise_edit_correct.csv"), index=False)

    novelty = novelty_vs_baseline(odd_correct, baseline_pool, baseline_solved)
    novelty.to_csv(os.path.join(out_dir, "novelty_vs_baseline.csv"), index=False)

    write_summary_md(os.path.join(out_dir, "summary.md"), summary, pairwise, novelty,
                     headline_alpha=headline_alpha, embedder_name=embedder_name)
    return per_batch, summary, pairwise, novelty


def summarize(per_batch: pd.DataFrame) -> pd.DataFrame:
    if per_batch.empty:
        return pd.DataFrame()
    agg = {
        "vendi_embed": "mean", "vendi_ngram": "mean", "n_clusters_all": "mean",
        "n_correct": "mean", "diversity_logged": "mean",
    }
    for t in EDIT_THRESHOLDS:
        agg[f"n_distinct_correct@{t}"] = "mean"
    for c in ("n_distinct_correct_ast", "gsm8k_recheck_agree"):
        if c in per_batch.columns:
            agg[c] = "mean"
    g = per_batch.groupby(KEY_COLS, dropna=False)
    out = g.agg(agg)
    out["n_problems"] = g.size()
    out["frac_solved"] = g["n_correct"].apply(lambda s: float((s > 0).mean()))
    out["frac_ge2_distinct_correct@0.3"] = g["n_distinct_correct@0.3"].apply(
        lambda s: float((s >= 2).mean()))
    return out.reset_index()


def pairwise_summary(per_batch: pd.DataFrame) -> pd.DataFrame:
    """One-token-flip refutation table: distribution of pairwise token edit
    distances between CORRECT solutions, per config."""
    rows = []
    if per_batch.empty:
        return pd.DataFrame()
    for key, g in per_batch.groupby(KEY_COLS, dropna=False):
        raw, norm = [], []
        for s_raw, s_norm in zip(g["pair_raw_edits"], g["pair_norm_edits"]):
            if isinstance(s_raw, str) and s_raw:
                raw.extend(float(x) for x in s_raw.split(";"))
                norm.extend(float(x) for x in s_norm.split(";"))
        row = dict(zip(KEY_COLS, key))
        row["n_problems_ge2_correct"] = int(
            g["pair_raw_edits"].apply(lambda s: isinstance(s, str) and bool(s)).sum())
        row.update(_dist_stats(raw, "raw"))
        row.update(_dist_stats(norm, "norm"))
        if raw:
            r = np.asarray(raw)
            row["frac_pairs_le2_tokens"] = float((r <= 2).mean())
            row["frac_pairs_norm_le0.05"] = float((np.asarray(norm) <= 0.05).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def novelty_vs_baseline(odd_correct, baseline_pool, baseline_solved) -> pd.DataFrame:
    """For problems ODD solves but baseline never solves (same benchmark /
    temperature / alg), min token edit distance from each ODD-correct solution
    to ANY baseline sample for that problem. Large distance = a genuinely new
    solution region, not a perturbation of a baseline sample."""
    rows = []
    for pool_key, runs in odd_correct.items():
        bench, temp, alg = pool_key
        pool = baseline_pool.get(pool_key, {})
        solved = baseline_solved.get(pool_key, {})
        for (run_id, alpha, pid), correct_texts in runs.items():
            base_texts = pool.get(pid)
            if not base_texts or solved.get(pid, False):
                continue  # need baseline coverage of this problem AND 0 baseline solves
            for t in correct_texts:
                tt = tokenize(t)
                best_raw, best_norm = None, None
                for b in base_texts:
                    bt = tokenize(b)
                    d = token_edit_distance(tt, bt)
                    dn = d / max(len(tt), len(bt), 1)
                    if best_norm is None or dn < best_norm:
                        best_raw, best_norm = d, dn
                rows.append({
                    "benchmark": bench, "temperature": temp, "alg": alg,
                    "alpha": alpha, "run_id": run_id, "problem_id": pid,
                    "n_baseline_samples": len(base_texts),
                    "min_raw_edit_to_baseline": best_raw,
                    "min_norm_edit_to_baseline": best_norm,
                })
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, cols, floatfmt="{:.3f}") -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary_md(path, summary, pairwise, novelty, headline_alpha=16.0,
                     embedder_name="minilm"):
    lines = ["# Diversity / correctness metrics summary", ""]
    lines.append(f"Embedder: `{embedder_name}` "
                 "(reported numbers must use `minilm`; `hash` is a test stub).")
    lines.append("")
    if not summary.empty:
        sel = summary[(summary["strategy"] == "baseline")
                      | (summary["alpha"] == headline_alpha)]
        for bench, g in sel.groupby("benchmark"):
            lines.append(f"## {bench} — baseline vs DPP vs ODD (alpha={headline_alpha:g})")
            lines.append("")
            cols = ["strategy", "alpha", "temperature", "n_problems", "vendi_embed",
                    "vendi_ngram", "n_clusters_all", "n_correct",
                    "n_distinct_correct@0.3", "frac_solved",
                    "frac_ge2_distinct_correct@0.3"]
            cols = [c for c in cols if c in g.columns]
            g = g.sort_values(["temperature", "strategy"])
            lines.append(_md_table(g, cols))
            lines.append("")
            lines.append("*Commentary (fill in): mean Vendi score per problem-batch "
                         "(semantic kernel); n-gram Vendi confirms the gain is not an "
                         "artifact of the embedding model; ODD raises the number of "
                         "DISTINCT correct solutions per problem, not just raw "
                         "correct-sample count.*")
            lines.append("")
    if not pairwise.empty:
        lines.append("## Pairwise token edit distance between correct solutions")
        lines.append("")
        cols = ["benchmark", "strategy", "alpha", "temperature",
                "n_problems_ge2_correct", "raw_n_pairs", "raw_mean", "raw_median",
                "norm_mean", "norm_median", "frac_pairs_le2_tokens"]
        cols = [c for c in cols if c in pairwise.columns]
        lines.append(_md_table(pairwise[pairwise["strategy"].isin(["baseline", "dpp", "odd"])]
                               .sort_values(["benchmark", "strategy", "temperature"]), cols))
        lines.append("")
        lines.append("*Commentary (fill in): `frac_pairs_le2_tokens` directly refutes the "
                     "'one token flipped' concern — pairs of ODD-correct solutions differ "
                     "by a median of N tokens (~X% of the sequence).*")
        lines.append("")
    if not novelty.empty:
        lines.append("## ODD-only solved problems: distance to nearest baseline sample")
        lines.append("")
        g = novelty.groupby(["benchmark", "temperature"], dropna=False).agg(
            n_solutions=("min_norm_edit_to_baseline", "size"),
            mean_min_norm=("min_norm_edit_to_baseline", "mean"),
            median_min_norm=("min_norm_edit_to_baseline", "median"),
            mean_min_raw=("min_raw_edit_to_baseline", "mean"),
        ).reset_index()
        lines.append(_md_table(g, list(g.columns)))
        lines.append("")
        lines.append("*Commentary (fill in): for problems the baseline never solves, "
                     "ODD's correct solutions are far (in token edit distance) from every "
                     "baseline sample — ODD reaches genuinely new solution regions.*")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="dir of downloaded .jsonl.gz runs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embedder", choices=["minilm", "hash"], default="minilm")
    ap.add_argument("--cluster-threshold", type=float, default=0.3,
                    help="cosine-distance threshold for agglomerative clustering over all samples")
    ap.add_argument("--headline-alpha", type=float, default=16.0)
    ap.add_argument("--projects", nargs="+", default=None)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--alphas", nargs="+", type=float, default=None,
                    help="only process runs with these alphas (baseline always kept)")
    args = ap.parse_args()
    process_all(args.data_dir, args.out_dir, embedder_name=args.embedder,
                cluster_threshold=args.cluster_threshold,
                headline_alpha=args.headline_alpha,
                max_runs=args.max_runs, projects=args.projects, alphas=args.alphas)


if __name__ == "__main__":
    main()
