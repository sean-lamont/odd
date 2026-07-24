"""Smoke test for the LLaDA2.0-mini ODD adapter (llada2_generator.py).

Phases (select with --phases, comma separated, default all):
  native   : native single-sample generate() at temp 0, seeds 0 and 1 (ground truth;
             note the native sampler is multinomial even at temp 0, so it is
             stochastic - the two seeds quantify its intrinsic variance).
  parity   : LLaDA2DiverseGenerator, batch=1, exact_native_sampling=True, seed 0.
             Must reproduce the seed-0 native output token-for-token if the batched
             loop is faithful.
  baseline : batch=4, strategy=baseline (alpha 0), temp 0 (greedy). All rows must be
             coherent, identical to each other, and close to the native output.
  odd      : batch=8, strategy=odd, alpha=16, temp 0. Expect diverse coherent rows.
  vram     : ODD at batch_size=16, gen_length=256, steps=32; on OOM, backs off to
             find the max feasible batch at gen 256, then tries batch 16 @ gen 128.

Run (GPU box):
  ~/miniconda3/envs/odd_rebuttal/bin/python llada2_smoke.py 2>&1 | tee smoke.log
"""

import argparse
import time

import torch

from feature_extractor import FeatureExtractor
from llada2_generator import (
    LLADA2_EOS_ID,
    LLADA2_MASK_ID,
    LLaDA2DiverseGenerator,
    load_llada2,
)
from strategies import get_strategy

# Prompt from conf/config.yaml
PROMPT = "Write a python function to compute fibonacci."
MODEL_NAME = "inclusionAI/LLaDA2.0-mini"


def gib(x):
    return x / (1024 ** 3)


def measure(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = fn()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak_alloc = gib(torch.cuda.max_memory_allocated())
    peak_reserved = gib(torch.cuda.max_memory_reserved())
    return result, dt, peak_alloc, peak_reserved


def show(tag, texts, max_chars=400):
    for i, t in enumerate(texts):
        t_disp = t.replace("\n", "\\n")
        if len(t_disp) > max_chars:
            t_disp = t_disp[:max_chars] + f"... [{len(t)} chars total]"
        print(f"[{tag} row {i}] {t_disp}")


def token_match(a, b):
    """(exact, frac_matching_prefixwise, len_a, len_b) for two 1-D id tensors."""
    a, b = a.flatten().tolist(), b.flatten().tolist()
    n = min(len(a), len(b))
    same = sum(1 for i in range(n) if a[i] == b[i])
    exact = (len(a) == len(b)) and same == n
    return exact, same / max(n, 1), len(a), len(b)


def lexical_diversity(texts):
    """Mean pairwise Jaccard distance over word sets (dependency-light proxy)."""
    sets = [set(t.split()) for t in texts]
    n = len(sets)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            u = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            dists.append(1.0 - (len(inter) / len(u) if u else 1.0))
    return sum(dists) / len(dists)


def make_generator(model, tokenizer, strategy_name, alpha):
    if strategy_name == "baseline":
        strategy = get_strategy("baseline", 0.0, 0.0, None)
    else:
        fe = FeatureExtractor(
            embedding_matrix=None,  # logit-space features (paper default)
            kernel_target="logits",
            pooling_method="max",
            top_k=0,
            use_confidence_weighting=True,
            ignore_token_ids=[],
        )
        strategy = get_strategy(strategy_name, alpha, 1.0, fe)
    return LLaDA2DiverseGenerator(
        model, tokenizer, strategy,
        mask_token_id=LLADA2_MASK_ID,
        block_length=32, threshold=0.95, eos_token_id=LLADA2_EOS_ID,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="native,parity,baseline,odd,vram")
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--batch_baseline", type=int, default=4)
    ap.add_argument("--batch_odd", type=int, default=8)
    args = ap.parse_args()
    phases = set(args.phases.split(","))

    print(f"=== LLaDA2.0-mini ODD smoke | gen_length={args.gen_length} steps={args.steps} ===")

    (loaded, load_t, _, _) = measure(lambda: load_llada2(MODEL_NAME, load_in_4bit=True))
    model, tokenizer = loaded
    print(f"[load] {load_t:.1f}s | VRAM allocated after load: {gib(torch.cuda.memory_allocated()):.2f} GiB")

    helper = make_generator(model, tokenizer, "baseline", 0.0)  # for prompt encoding
    prompt_ids = helper.encode_prompt(PROMPT, 1)
    print(f"[prompt] '{PROMPT}' -> {prompt_ids.shape[1]} tokens")

    native_ids = {}
    native_texts = {}

    # ---------------- native ----------------
    if "native" in phases:
        for seed in (0, 1):
            torch.manual_seed(seed)
            out, dt, pk, pkr = measure(lambda: model.generate(
                inputs=prompt_ids, temperature=0.0, block_length=32,
                steps=args.steps, gen_length=args.gen_length,
                threshold=0.95, eos_id=LLADA2_EOS_ID, mask_id=LLADA2_MASK_ID,
            ))
            native_ids[seed] = out[0].detach().cpu()
            native_texts[seed] = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
            print(f"\n[native seed={seed}] {dt:.1f}s | peak {pk:.2f} GiB alloc / {pkr:.2f} GiB reserved "
                  f"| {out.shape[1]} tokens")
            show(f"native s{seed}", [native_texts[seed]])
        exact, frac, la, lb = token_match(native_ids[0], native_ids[1])
        print(f"[native variance] seed0 vs seed1: exact={exact} prefix_token_match={frac:.3f} "
              f"lens=({la},{lb})  <- native temp-0 is multinomial, not greedy")

    # ---------------- parity ----------------
    if "parity" in phases and 0 in native_ids:
        gen = make_generator(model, tokenizer, "baseline", 0.0)
        torch.manual_seed(0)
        (res, dt, pk, _) = measure(lambda: gen.generate(
            PROMPT, batch_size=1, steps=args.steps, gen_length=args.gen_length,
            temperature=0.0, exact_native_sampling=True,
        ))
        _, texts = res
        exact, frac, la, lb = token_match(gen.last_sequences[0], native_ids[0])
        print(f"\n[parity batch=1 exact-native-sampling seed=0] {dt:.1f}s | peak {pk:.2f} GiB")
        print(f"[parity] token-for-token match with native seed0: exact={exact} "
              f"prefix_token_match={frac:.3f} lens=({la},{lb})")
        if not exact:
            show("parity", texts)

    # ---------------- baseline batched ----------------
    if "baseline" in phases:
        gen = make_generator(model, tokenizer, "baseline", 0.0)
        (res, dt, pk, pkr) = measure(lambda: gen.generate(
            PROMPT, batch_size=args.batch_baseline, steps=args.steps,
            gen_length=args.gen_length, temperature=0.0,
        ))
        _, texts = res
        print(f"\n[baseline batch={args.batch_baseline} temp=0 greedy] {dt:.1f}s | "
              f"peak {pk:.2f} GiB alloc / {pkr:.2f} GiB reserved")
        n_unique = len(set(texts))
        print(f"[baseline] unique rows: {n_unique}/{len(texts)} (expect 1: greedy rows are identical)")
        if 0 in native_texts:
            exact, frac, la, lb = token_match(gen.last_sequences[0], native_ids[0])
            print(f"[baseline] row0 vs native seed0: exact={exact} prefix_token_match={frac:.3f} "
                  f"lens=({la},{lb}) (drift only where native multinomial != argmax)")
        show("baseline", texts[:2])

    # ---------------- odd ----------------
    if "odd" in phases:
        gen = make_generator(model, tokenizer, "odd", args.alpha)
        (res, dt, pk, pkr) = measure(lambda: gen.generate(
            PROMPT, batch_size=args.batch_odd, steps=args.steps,
            gen_length=args.gen_length, temperature=0.0,
        ))
        _, texts = res
        print(f"\n[odd alpha={args.alpha} batch={args.batch_odd} temp=0] {dt:.1f}s | "
              f"peak {pk:.2f} GiB alloc / {pkr:.2f} GiB reserved")
        n_unique = len(set(texts))
        n_empty = sum(1 for t in texts if not t.strip())
        print(f"[odd] unique rows: {n_unique}/{len(texts)} | empty rows: {n_empty} | "
              f"lexical diversity (mean pairwise Jaccard dist): {lexical_diversity(texts):.3f}")
        show("odd", texts)

    # ---------------- vram sweep ----------------
    if "vram" in phases:
        print("\n[vram] ODD sweep (steps=32)")
        configs = [(16, 256), (12, 256), (8, 256), (16, 128)]
        done_256 = False
        for bs, gl in configs:
            if gl == 256 and done_256:
                continue
            gen = make_generator(model, tokenizer, "odd", args.alpha)
            try:
                (res, dt, pk, pkr) = measure(lambda: gen.generate(
                    PROMPT, batch_size=bs, steps=32, gen_length=gl, temperature=0.0,
                ))
                _, texts = res
                n_unique = len(set(texts))
                print(f"[vram] batch={bs} gen={gl}: OK | {dt:.1f}s | peak {pk:.2f} GiB alloc / "
                      f"{pkr:.2f} GiB reserved | unique {n_unique}/{bs}")
                if gl == 256:
                    done_256 = True
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                print(f"[vram] batch={bs} gen={gl}: OOM ({str(e).splitlines()[0][:120]})")

    print("\n=== smoke done ===")


if __name__ == "__main__":
    main()
