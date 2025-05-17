#!/bin/bash
#SBATCH --partition cpufast
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 1:00:00
#SBATCH --job-name benczechmark_eval_cpu
#SBATCH --output logs/benczechmark-logs-leaderboard.%j.out

ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source ~/venvs/benczechmark/bin/activate

cd ../benczechmark-leaderboard

echo $(pwd)

export PYTHONPATH=$(pwd)

MODEL_ID="b->nli_(cs+en)"

python /home/mlynatom/benczechmark-leaderboard/leaderboard/compile_log_files.py -i "/home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation_results/bcm/results/${MODEL_ID}/eval_${MODEL_ID}*" -o /home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation_results/bcm/submissions/${MODEL_ID}_submission.json