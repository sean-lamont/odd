"""Offline tests for the rebuttal analysis pipeline (no wandb, no downloads).

Run:  python rebuttal_analysis/test_metrics.py
Optionally set FIXTURE=/path/to/old_format.table.json to also test parsing of
a real old-format wandb table (question only on first row of each batch).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diversity_metrics import (  # noqa: E402
    HashEmbedder,
    ast_normalize,
    batch_metrics,
    embedding_vendi,
    extract_answer_num,
    extract_gold_num,
    gsm8k_recheck,
    n_distinct_at_threshold,
    n_distinct_ast,
    n_embedding_clusters,
    ngram_vendi,
    normalized_edit_distance,
    pairwise_edit_matrix,
    vendi_from_kernel,
)
from table_utils import (  # noqa: E402
    assign_batch_ids,
    normalize_table,
    parse_run_config,
    verify_batches,
)

PASS = 0


def check(name, cond, detail=""):
    global PASS
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if cond:
        PASS += 1
    else:
        raise AssertionError(f"{name}: {detail}")


def test_vendi_kernel():
    print("vendi_from_kernel:")
    n = 8
    check("identical -> 1", abs(vendi_from_kernel(np.ones((n, n))) - 1.0) < 1e-9)
    check("orthogonal -> n", abs(vendi_from_kernel(np.eye(n)) - n) < 1e-9)
    # two groups of identical items -> 2
    K = np.kron(np.eye(2), np.ones((4, 4)))
    check("2 clusters of 4 -> 2", abs(vendi_from_kernel(K) - 2.0) < 1e-9)


def test_text_vendi():
    print("embedding/ngram vendi on texts:")
    emb = HashEmbedder()
    same = ["def f(x): return x + 1"] * 6
    vs_same = embedding_vendi(same, emb)
    check("identical texts VS ~= 1", abs(vs_same - 1.0) < 1e-6, f"got {vs_same:.4f}")
    diff = ["alpha beta gamma", "1234567890", "zzzz qqqq wwww", "the quick brown fox",
            "import numpy as np", "SELECT * FROM users"]
    vs_diff = embedding_vendi(diff, emb)
    check("distinct texts VS >> 1", vs_diff > 3.0, f"got {vs_diff:.4f} (n=6)")
    check("ngram identical ~= 1", abs(ngram_vendi(same) - 1.0) < 1e-6)
    vg = ngram_vendi(diff)
    check("ngram distinct >> 1", vg > 3.0, f"got {vg:.4f}")
    check("hash embedder deterministic",
          np.allclose(emb.encode(diff), HashEmbedder().encode(diff)))


def test_edit_distance():
    print("token edit distance:")
    a = "one two three four five six seven eight nine ten"
    raw, norm = normalized_edit_distance(a, a)
    check("same -> 0", raw == 0 and norm == 0.0)
    b = a.replace("five", "5")
    raw, norm = normalized_edit_distance(a, b)
    check("one token flip -> raw 1", raw == 1, f"got {raw}")
    check("one token flip norm = 0.1", abs(norm - 0.1) < 1e-9, f"got {norm}")
    raw, _ = normalized_edit_distance("", "a b c")
    check("empty vs 3 tokens -> 3", raw == 3)


def test_distinct():
    print("distinct-at-threshold:")
    texts = ["a b c d e f g h i j"] * 3 + ["completely different words entirely here now ok yes sure done"]
    _, norm = pairwise_edit_matrix(texts)
    check("3 same + 1 diff @0.2 -> 2", n_distinct_at_threshold(norm, 0.2) == 2)
    check("all distinct @0.0 counts singletons",
          n_distinct_at_threshold(norm, -0.1) == 4)


def test_ast():
    print("AST normalization:")
    v1 = 'def f(x):\n    """doc"""\n    # comment\n    return x + 1\n'
    v2 = "def f(x):\n    return x + 1  # different comment\n"
    v3 = "def f(x):\n    return 1 + x\n"
    check("comments/docstrings ignored", ast_normalize(v1) == ast_normalize(v2))
    check("real change detected", ast_normalize(v1) != ast_normalize(v3))
    check("n_distinct_ast", n_distinct_ast([v1, v2, v3]) == 2)
    # body-only completion parses when appended to the prompt
    prompt = "def g(y):\n"
    body = "    return y * 2\n"
    check("prompt+body fallback", ast_normalize(body, prompt=prompt) is not None)


def test_clusters():
    print("embedding clustering:")
    emb = HashEmbedder()
    texts = ["return the sum of a and b please"] * 5 + ["zzz qqq 999 xxx completely other"] * 3
    k = n_embedding_clusters(texts, emb, 0.3)
    check("two blocks -> 2 clusters", k == 2, f"got {k}")
    check("all identical -> 1", n_embedding_clusters(["x y z"] * 4, emb, 0.3) == 1)


def test_gsm8k_extract():
    print("GSM8K extraction:")
    check("answer num", extract_answer_num("so the answer is 1,234.5 dollars") == 1234.5)
    check("gold num", extract_gold_num("blah blah #### 42") == 42.0)
    flags = gsm8k_recheck(["makes 9*2 = 18", "total is 17"], 18.0)
    check("recheck flags", flags == [True, False], f"got {flags}")


def test_batch_assignment():
    print("batch id assignment:")
    # New format: problem id on every row
    cols = ["question", "gold", "generated", "is_correct", "diversity"]
    data = []
    for p in range(5):
        for s in range(4):
            data.append([f"Q{p}", 1.0, f"ans {p} {s}", s == 0, 0.5 + p * 0.01])
    df = assign_batch_ids(normalize_table(cols, data), batch_size=4)
    stats = verify_batches(df, batch_size=4)
    check("new format: 5 batches of 4", stats["size_histogram"] == {4: 5}, str(stats))
    check("new format: no warnings", not stats["warnings"], str(stats["warnings"]))

    # Old format: question only on first row of each batch (as in the fixture)
    data2 = []
    for p in range(5):
        for s in range(4):
            data2.append([f"Q{p}" if s == 0 else "", 1.0, f"ans {p} {s}", s == 0, 0.5 + p * 0.01])
    df2 = assign_batch_ids(normalize_table(cols, data2), batch_size=4)
    stats2 = verify_batches(df2, batch_size=4)
    check("old format: 5 batches of 4", stats2["size_histogram"] == {4: 5}, str(stats2))
    check("old format: pids forward-filled",
          df2["problem_id"].tolist() == [f"Q{p}" for p in range(5) for _ in range(4)])


def test_config_parsing():
    print("run config parsing:")
    llada = parse_run_config({
        "strategy": {"name": "batched_orth", "alpha": 16.0, "pool": "max"},
        "temperature": 1.0, "batch_size": 16, "n_problems": 200,
        "model": {"name": "GSAI-ML/LLaDA-8B-Instruct", "load_in_4bit": True},
    })
    check("llada strategy", llada["strategy_name"] == "batched_orth"
          and llada["strategy_class"] == "odd" and llada["alpha"] == 16.0)
    check("llada model", llada["model"] == "GSAI-ML/LLaDA-8B-Instruct")
    llada_str = parse_run_config({
        "strategy": "{'name': 'joint', 'alpha': 8.0}", "temperature": 0.5})
    check("stringified strategy dict", llada_str["strategy_class"] == "dpp"
          and llada_str["alpha"] == 8.0)
    dream = parse_run_config({
        "strategy": "odd", "alpha": 64.0, "alg": "maskgit_plus",
        "task": "gsm8k", "temperature": 2.0, "batch_size": 16,
    })
    check("dream flat strategy", dream["strategy_name"] == "odd"
          and dream["alpha"] == 64.0 and dream["alg"] == "maskgit_plus")


def test_batch_metrics():
    print("batch_metrics end-to-end (stub embedder):")
    emb = HashEmbedder()
    texts = ["She sells 16-3-4 = 9 eggs. She makes 9 * $2 = $18 per day."] * 3 + [
        "Total eggs 16. Eats 3, bakes 4, leaving 9. Income: 9 x 2 = 18.",
        "The answer is 20.",
    ]
    flags = [True, True, True, True, False]
    rec = batch_metrics(texts, flags, "gsm8k", emb, gold=18.0)
    check("n_correct", rec["n_correct"] == 4)
    check("distinct correct @0.3 = 2", rec["n_distinct_correct@0.3"] == 2,
          str({k: v for k, v in rec.items() if "distinct" in k}))
    check("recheck agrees", rec["gsm8k_recheck_agree"] == 1.0, str(rec.get("gsm8k_recheck_agree")))
    check("vendi in (1, n]", 1.0 < rec["vendi_embed"] <= 5.0, f"{rec['vendi_embed']:.3f}")


def test_fixture():
    fixture = os.environ.get("FIXTURE")
    if not fixture or not os.path.exists(fixture):
        print("fixture test: SKIPPED (set FIXTURE=/path/to/*.table.json)")
        return
    print(f"fixture parse ({os.path.basename(fixture)}):")
    with open(fixture) as f:
        table = json.load(f)
    df = normalize_table(table["columns"], table["data"])
    bs = int(os.environ.get("FIXTURE_BATCH_SIZE", "4"))
    df = assign_batch_ids(df, batch_size=bs)
    stats = verify_batches(df, batch_size=bs)
    print(f"  stats: {stats}")
    check("fixture: uniform batch size", stats["size_histogram"] == {bs: stats["n_batches"]},
          str(stats["size_histogram"]))
    check("fixture: no warnings", not stats["warnings"], str(stats["warnings"]))
    check("fixture: problem ids filled", df["problem_id"].notna().all()
          and (df["problem_id"].astype(str).str.len() > 0).all())
    # spot-check the batch metrics pipeline on the first two batches
    from diversity_metrics import batch_metrics as bm
    for bid in list(df["batch_id"].unique())[:2]:
        g = df[df["batch_id"] == bid]
        rec = bm([str(t) for t in g["text"]], [bool(c) for c in g["correct"]],
                 "gsm8k", HashEmbedder(), gold=g["ref"].iloc[0])
        print(f"  batch {bid}: n_correct={rec['n_correct']} vendi={rec['vendi_embed']:.3f} "
              f"recheck_agree={rec.get('gsm8k_recheck_agree')}")


if __name__ == "__main__":
    test_vendi_kernel()
    test_text_vendi()
    test_edit_distance()
    test_distinct()
    test_ast()
    test_clusters()
    test_gsm8k_extract()
    test_batch_assignment()
    test_config_parsing()
    test_batch_metrics()
    test_fixture()
    print(f"\nAll checks passed ({PASS}).")
