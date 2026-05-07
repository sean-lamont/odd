#!/bin/bash

# analyse results (commented out as we provide the results from our wandb experiments)
# python download_runs.py
# python download_tables.py

# note for the below scripts, strategy naming can be confusing:
# batched_orth is old name for odd, and 'joint' is the old name for dpp baseline
python gen_table.py > results_table.txt
python get_cumulative.py > cumulative_results.txt
python plot_diversity.py
python plot_pareto.py
python plot_passk.py

# if model has not been profiled, run in the root directory, with the appropriate filename for the overhead script.
# default runs in quantized mode but you can remove the bnb conf to test bf16.
# python profile_model_.py
# python plot_overhead.py