#!/bin/bash
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 12:00:00
#SBATCH --job-name benczechmark_eval_cpu
#SBATCH --output logs/benczechmark-logs.%j.out

ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source ~/venvs/benczechmark/bin/activate

cd benczechmark-leaderboard/

export PYTHONPATH=$(pwd)

python /home/mlynatom/benczechmark-leaderboard/leaderboard/compile_log_files.py -i "/home/mlynatom/master-thesis-repository-tomas-mlynar/results/eval_llama32-3B_instruct*" -o /home/mlynatom/master-thesis-repository-tomas-mlynar/llama3.2-3B-Instruct_submission.json