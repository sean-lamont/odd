# Rebuttal analysis runbook (P0)

Pipeline for the NeurIPS rebuttal: pull the full per-sample generation text
from the wandb `results_table` artifacts (the existing
`analyse_results/download_tables.py` drops the text), then compute Vendi
scores, solution-level joint diversity–correctness stats, and qualitative
evidence that ODD's diverse solutions are genuinely distinct.

## What serves which reviewer ask

| Output | Reviewer ask |
|---|---|
| `metrics/summary.md` + `summary_metrics.csv` — mean Vendi (embedding + n-gram kernels) per config | QcT2 + oRwm (Vendi score) |
| `metrics/summary_metrics.csv` (`n_distinct_correct@t`, `frac_ge2_distinct_correct@0.3`), `pairwise_edit_correct.csv`, `novelty_vs_baseline.csv` | oRwm (solution-level joint diversity–correctness; "not one token flipped") |
| `qualitative_examples.md` | QcT2 (concrete baseline-collapse vs ODD examples) |
| `alpha_transfer_results.md` (already generated, checked in) | alpha-robustness / no-per-setting-tuning ask |

## Where the runs actually live (verified against the cloud, 2026-07-24)

The sweep scripts say `project="gsm8k"/"humaneval"`, but on the credential's
default entity (`tactic-zero`) those projects hold only partial joint-only
runs. The full paper sweeps, matched to the aggregate CSVs by run count AND
strategy composition, are:

| Project (entity-qualified) | Runs | Strategies |
|---|---|---|
| `sean-a-lamont/odd_gsm8k` | 757 | joint 240, batched_orth 240, orthogonal_projection 237, baseline 40 |
| `sean-a-lamont/odd_humaneval` | 759 | joint 237, batched_orth 238, orthogonal_projection 244, baseline 40 |
| `tactic-zero/dream_gsm8k_eval` | 202 | odd 160, baseline 42 (alg maskgit_plus + origin) |
| `tactic-zero/dream_humaneval_eval` | 203 | odd 160, baseline 43 (alg maskgit_plus + origin) |

These are the `--projects` defaults in `download_text_tables.py` (items may be
`entity/project`; bare names use `--entity` / the default entity). Spot-checked:
runs carry a `run-<id>-results_table:v0` artifact; LLaDA GSM8K runs use
batch_size=16 with n_problems=300, HumanEval 164; Dream batch_size=16.

## 0. Prerequisites

* All downloads must run on the box (`ssh sean@192.168.0.23`) — that is where
  the logged-in wandb credential lives. **Do not start until the current model
  download there has finished (bandwidth is saturated; wandb API calls were
  already timing out at 9 s during verification).**
* Box python: `~/miniconda3/envs/odd/bin/python` (verified: wandb 0.13.5,
  pandas 2.3.3, scikit-learn, sentence-transformers 5.1.2 — all-MiniLM-L6-v2
  is the same model the paper's diversity metric uses).
* Offline sanity check first (works anywhere, no cloud):

```bash
python rebuttal_analysis/test_metrics.py
# optionally against a real old-format table on the box:
FIXTURE=~/Documents/text_dpp/wandb/run-20260130_173409-s1ralfr4/files/media/table/results_table_300_29060c4e6245f9265f38.table.json \
  FIXTURE_BATCH_SIZE=4 python rebuttal_analysis/test_metrics.py
```

## 1. Download (on the box, once bandwidth frees)

Get this branch onto the box, then smoke-test with ONE small run:

```bash
ssh sean@192.168.0.23
cd ~/Documents/odd && git fetch && git checkout rebuttal && git pull
PY=~/miniconda3/envs/odd/bin/python
$PY rebuttal_analysis/download_text_tables.py \
    --out rebuttal_analysis/data \
    --projects sean-a-lamont/odd_humaneval --filter-strategy baseline --max-runs 1
```

Check the printed batch verification (expect batches of 16, no warnings).
Then pull the headline subset (baseline + DPP/ODD at alpha=16, all temps —
enough for every headline table; ~120 runs per LLaDA project, ~50 per Dream
project):

```bash
$PY rebuttal_analysis/download_text_tables.py --out rebuttal_analysis/data \
    --filter-strategy baseline
$PY rebuttal_analysis/download_text_tables.py --out rebuttal_analysis/data \
    --filter-alpha 16
```

Finally (optional, for alpha-sweep versions of the metrics) the full pull:

```bash
$PY rebuttal_analysis/download_text_tables.py --out rebuttal_analysis/data
```

Notes:

* Defaults: `--projects sean-a-lamont/odd_gsm8k sean-a-lamont/odd_humaneval
  tactic-zero/dream_gsm8k_eval tactic-zero/dream_humaneval_eval` (see table
  above); bare names fall back to `--entity` /
  `wandb.Api().default_entity`.
* Resumable: re-running skips runs whose `.jsonl.gz` already exists, so it is
  safe to interrupt / re-launch.
* Expected volumes: LLaDA 757/759 runs/project x (300 problems x 16 samples,
  GSM8K; 164 x 16, HumanEval); Dream ~200 runs/project (~half alg=origin).
  Roughly 0.2–1 MB gzipped per run → order 100 MB for the headline subset,
  ~1–2 GB for everything. wandb also caches raw artifacts under
  `./artifacts/` — delete that dir afterwards to reclaim space.
* Strategy filter: `--filter-strategy odd` matches both `batched_orth`
  (LLaDA) and `odd` (Dream) via the class mapping; DPP = `joint` (or class
  `dpp`), LLaDA only. Dream runs with `alg=origin` are also pulled — the
  metrics keep `alg` as a grouping column, so restrict paper-facing numbers
  to `alg=maskgit_plus` rows (`summary_metrics.csv`).

## 2. Metrics (box or laptop, after syncing `rebuttal_analysis/data`)

```bash
# 1) Vendi + solution-level + one-token-flip + novelty stats
python rebuttal_analysis/diversity_metrics.py \
    --data-dir rebuttal_analysis/data --out-dir rebuttal_analysis/metrics \
    --embedder minilm

# 2) Qualitative examples (HumanEval preferred)
python rebuttal_analysis/select_qualitative.py \
    --data-dir rebuttal_analysis/data --out rebuttal_analysis/qualitative_examples.md \
    --benchmark humaneval --alpha 16

# 3) Alpha transfer (no cloud needed; already run and checked in)
python rebuttal_analysis/alpha_transfer.py \
    --dream-csv-dir ../supp/new_results \
    --out rebuttal_analysis/alpha_transfer_results.md
```

Order matters only in that (1) and (2) need the download; (3) is independent
(reads the aggregate CSVs in `analyse_results/` and the supp bundle).
`--embedder hash` runs everything without sentence-transformers for smoke
tests; reported numbers must use `minilm`.

## 3. Assumptions to verify against the real cloud tables

1. **Batch structure**: cloud tables are assumed to repeat the problem id on
   every row, 16 consecutive rows per problem (verified programmatically:
   `download_text_tables.py` prints a WARNING per run if recovered batch sizes
   != config `batch_size` or diversity is non-constant within a batch).
   The old local fixture instead leaves the id blank after the first row of a
   batch — both formats are handled.
2. **GSM8K `is_correct` agreement**: `gsm8k_recheck_agree` in
   `per_batch_metrics.csv` should be ~1.0 for cloud runs (they use exactly the
   `extract_answer_num`/`extract_gold_num` logic reused here). The old local
   fixture run (project `text_dpp`) shows ~0.79 agreement — it predates the
   final correctness check, which is why cloud-run verification matters.
3. **`gold` column format**: sweeps log `gold` as a float (already extracted);
   the recheck also accepts raw `#### x` strings.
4. **Dream runs**: keep `alg=maskgit_plus` only for paper-facing numbers.
5. **HumanEval completions**: logged completions are already
   markdown-stripped (`clean_code_for_harness`); AST normalization retries
   `prompt+completion` for body-only outputs and falls back to
   whitespace-collapsed text when unparseable.
6. **Excluded projects**: `tactic-zero/gsm8k` (175 runs) and
   `tactic-zero/humaneval` (107 runs) contain ONLY `joint` runs and appear to
   be partial early duplicates of the odd_* sweeps; they are excluded from the
   defaults to avoid double counting. Same for `gsm8k_sweep`/`humaneval_sweep`
   (earlier strategy variants: `orthogonal_projection`, `gram_schmidt`,
   `random_probe`). Verify with the user before ever merging them in.
7. **n_problems**: the checked-in `sweep_gsm8k.py` says `n_problems = 200`,
   but the cloud LLaDA GSM8K runs were launched with `n_problems = 300`
   (config spot-check). Per-run config is captured in each file's meta line —
   trust that, not the script.

## DiffusionGemma probe (added 2026-07-25)

Fresh env (transformers v5 API — do NOT touch the dream env):
```
conda create -y -n gemma_rebuttal python=3.12
conda run -n gemma_rebuttal pip install "transformers>=5.14" torch bitsandbytes accelerate sentence-transformers "datasets>=5" wandb "setuptools<81"
```
Checkpoint download is ~52GB bf16 (quantized to 4-bit nf4 on load, ~15GB VRAM).
Probe (single-canvas: gen_length <= 256 keeps the ODD anneal exact):
```
python sweep_gsm8k_plain.py --model-config diffusion_gemma --n-problems 3  --strategies baseline odd --alphas 4   --temperatures 1.0 --gen-length 256 --steps 32   # sanity + coherence check
python sweep_gsm8k_plain.py --model-config diffusion_gemma --n-problems 50 --strategies odd      --alphas 1 4 --temperatures 1.0 --gen-length 256 --steps 32   # alpha probe (LLaDA2 lesson: small alpha)
```
Open items before quoting numbers: verify the processor sees (B, canvas, V) logits
and canvas ids as input_ids (adapter is defensive but log-check the first run);
compare baseline at constant-theta vs native t_max->t_min schedule.
