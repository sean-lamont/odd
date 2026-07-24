# Alpha robustness and transfer (pass@16, percentage points)

Dream runs filtered to `alg=maskgit_plus` (paper setting). Gains are ODD minus baseline at the same temperature; each cell averages over the temperature grid {0.0, 0.5, 1.0, 1.5, 2.0}.

## Fixed alpha = 16 (no per-setting tuning)

| Model | Task | mean gain (pts) | min gain (pts) | oracle alpha per temp | regret vs oracle mean/max (pts) |
|---|---|---|---|---|---|
| LLaDA | GSM8K | +6.42 | +1.00 | 0->128, 0.5->128, 1->64, 1.5->128, 2->128 | 4.37 / 5.06 |
| LLaDA | HumanEval | +15.96 | +7.93 | 0->16, 0.5->16, 1->16, 1.5->8, 2->32 | 1.66 / 6.47 |
| Dream | GSM8K | +9.70 | -1.62 | 0->128, 0.5->64, 1->128, 1.5->128, 2->128 | 5.17 / 8.63 |
| Dream | HumanEval | +17.65 | +6.55 | 0->64, 0.5->64, 1->128, 1.5->8, 2->16 | 1.43 / 4.27 |

## Cross-(model, task) transfer of a single tuned alpha

Alpha is tuned on the SOURCE setting (argmax of mean gain over its temperature grid) and applied unchanged to the TARGET setting (12 source-target pairs).

| Source (alpha*) | Target | mean gain (pts) | min gain (pts) | mean regret vs target oracle (pts) |
|---|---|---|---|---|
| LLaDA/GSM8K (a*=128) | LLaDA/HumanEval | +13.17 | +2.41 | 4.45 |
| LLaDA/GSM8K (a*=128) | Dream/GSM8K | +14.33 | +3.75 | 0.55 |
| LLaDA/GSM8K (a*=128) | Dream/HumanEval | +14.33 | -2.44 | 4.76 |
| LLaDA/HumanEval (a*=16) | LLaDA/GSM8K | +6.42 | +1.00 | 4.37 |
| LLaDA/HumanEval (a*=16) | Dream/GSM8K | +9.70 | -1.62 | 5.17 |
| LLaDA/HumanEval (a*=16) | Dream/HumanEval | +17.65 | +6.55 | 1.43 |
| Dream/GSM8K (a*=128) | LLaDA/GSM8K | +10.72 | +5.25 | 0.06 |
| Dream/GSM8K (a*=128) | LLaDA/HumanEval | +13.17 | +2.41 | 4.45 |
| Dream/GSM8K (a*=128) | Dream/HumanEval | +14.33 | -2.44 | 4.76 |
| Dream/HumanEval (a*=16) | LLaDA/GSM8K | +6.42 | +1.00 | 4.37 |
| Dream/HumanEval (a*=16) | LLaDA/HumanEval | +15.96 | +7.93 | 1.66 |
| Dream/HumanEval (a*=16) | Dream/GSM8K | +9.70 | -1.62 | 5.17 |

**Across all 12 transfer pairs: mean gain +12.16 pts (worst pair mean +6.42; worst single temperature -2.44; mean regret vs per-setting oracle 3.43 pts).**

*Takeaway: alpha transfers across models (LLaDA <-> Dream) and tasks (GSM8K <-> HumanEval) — ODD does not require per-setting hyperparameter tuning to beat the baseline.*
