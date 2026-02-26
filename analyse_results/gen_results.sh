#!/bin/bash

# analyse results
python download_runs.py
python download_tables.py

python gen_table.py > results_table.txt
python get_cumulative.py > cumulative_results.txt
python plot_diversity.py
python plot_pareto.py
python plot_passk.py

# if model has not been profiled, run
# python profile_model_.py
python plot_overhead.py


