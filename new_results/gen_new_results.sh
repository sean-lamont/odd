#!/bin/bash

# download wandb results (commented out as we provide the results from our wandb experiments)
# python download_data.py
# python download_tables.py

python gen_tables.py > new_tables.txt
python get_cumulative.py > cumulative_results.txt