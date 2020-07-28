#!/bin/bash

#SBATCH --job-name=test_task

#SBATCH --output=test_task

#SBATCH -c 6

#SBATCH --time=00:10:00

#SBATCH --mem-per-cpu=15gb

#SBATCH --array 0-2

export PATH=~/miniconda3/bin:$PATH

source activate ~/miniconda3/envs/vivenvbtc/

cd ~/BitcoinMachineLearning/SklearnComponent

python3.6 HelperFunctions.py --task_idx $SLURM_ARRAY_TASK_ID  --csv '../inputs/bitfinex_tBTCUSD_1h.csv' --mlmethod 'Classification' --typeregression 'Difference' --intervalperiod 1 --hyperrelations 1 --rscv 'ON' --niters 1 --rfnest 10
