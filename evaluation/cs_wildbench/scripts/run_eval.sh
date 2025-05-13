#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --partition=cpufast
#SBATCH --mem-per-cpu 54G
#SBATCH --job-name wildbench-prepare-batches
#SBATCH --output /home/mlynatom/master-thesis-repository-tomas-mlynar/logs/wildbench/run_eval.%J.log


MODEL_1="B+IT->IT_(cs+en)" # example model name
MODEL_2="B+IT" # example model name

ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source /home/mlynatom/master-thesis-repository-tomas-mlynar/venv/wildbench_venv/bin/activate

export PYTHONPATH=src:$PYTHONPATH

bash /home/mlynatom/WildBench/evaluation/run_all_eval_batch.sh $MODEL_1 $MODEL_2