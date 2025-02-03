#!/bin/bash
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 12:00:00
#SBATCH --job-name benczechmark_eval_cpu
#SBATCH --output logs/benczechmark-logs-leaderboard.%j.out

ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source ~/venvs/benczechmark/bin/activate

cd ../benczechmark-leaderboard

echo $(pwd)

export PYTHONPATH=$(pwd)

python /home/mlynatom/benczechmark-leaderboard/leaderboard/compile_log_files.py -i "/home/mlynatom/master-thesis-repository-tomas-mlynar/results/eval_llama3.2-3b-instruct-1epoch-32batch-4gradacc-1e-4lr-merged*" -o /home/mlynatom/master-thesis-repository-tomas-mlynar/eval_llama3.2-3b-instruct-1epoch-32batch-4gradacc-1e-4lr-merged_submission.json