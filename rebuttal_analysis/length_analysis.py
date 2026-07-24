"""Length/EOS-behaviour analysis over downloaded WandB run files.

For each run (strategy, alpha, temperature, benchmark, alg), compute the
distribution of generated-output lengths. The decoded text has EOS/PAD
stripped, so token count of the text measures content emitted before EOS;
values at/near gen_length indicate the sample never emitted EOS (saturation).
Answers: does ODD systematically change EOS/length behaviour vs baseline?
"""
import gzip, json, glob, os, sys, csv, statistics as st
from collections import defaultdict

DATA = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/odd_rebuttal/odd/rebuttal_analysis/data")
OUT = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/odd_rebuttal/odd/rebuttal_analysis/metrics")
os.makedirs(OUT, exist_ok=True)

tok = None
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
except Exception as e:
    print("tokenizer unavailable, falling back to whitespace tokens:", e)

def ntokens(text):
    if not text:
        return 0
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return len(text.split())

rows = []
for f in sorted(glob.glob(os.path.join(DATA, "*.jsonl.gz"))):
    try:
        with gzip.open(f, "rt") as fh:
            meta = json.loads(fh.readline())["_meta"]
            cfg = meta.get("config", {})
            lens, empties, n = [], 0, 0
            gen_col = None
            for line in fh:
                r = json.loads(line)
                if gen_col is None:
                    for c in ("generated", "completion", "text"):
                        if c in r:
                            gen_col = c
                            break
                    if gen_col is None:
                        break
                t = r.get(gen_col) or ""
                L = ntokens(t)
                lens.append(L)
                empties += (L == 0)
                n += 1
    except Exception as e:
        print("SKIP", os.path.basename(f), e)
        continue
    if not lens:
        continue
    gl = cfg.get("gen_length") or 64
    lens.sort()
    q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
    rows.append({
        "project": meta.get("project"), "benchmark": meta.get("benchmark"),
        "strategy": cfg.get("strategy_class") or cfg.get("strategy_name"),
        "alpha": cfg.get("alpha"), "temperature": cfg.get("temperature"),
        "alg": cfg.get("alg"), "gen_length": gl, "n_samples": len(lens),
        "mean_len": round(st.mean(lens), 2), "median_len": q(0.5),
        "p10": q(0.10), "p90": q(0.90),
        "frac_at_cap": round(sum(l >= gl - 1 for l in lens) / len(lens), 4),
        "frac_empty": round(empties / len(lens), 4),
    })
    print("OK", os.path.basename(f), rows[-1]["strategy"], rows[-1]["alpha"], rows[-1]["temperature"], rows[-1]["mean_len"])

out_csv = os.path.join(OUT, "length_distribution.csv")
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# aggregate across runs of the same config
agg = defaultdict(list)
for r in rows:
    agg[(r["project"], r["strategy"], str(r["alpha"]), str(r["temperature"]), str(r["alg"]))].append(r)
with open(os.path.join(OUT, "length_summary.md"), "w") as fh:
    fh.write("| project | strategy | alpha | temp | alg | runs | mean_len | frac_at_cap | frac_empty |\n|---|---|---|---|---|---|---|---|---|\n")
    for k in sorted(agg):
        g = agg[k]
        fh.write(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {len(g)} | "
                 f"{st.mean(x['mean_len'] for x in g):.1f} | {st.mean(x['frac_at_cap'] for x in g):.3f} | "
                 f"{st.mean(x['frac_empty'] for x in g):.3f} |\n")
print("wrote", out_csv)
