#!/bin/bash
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 24:00:00
#SBATCH --job-name benczechmark_eval_cpu
#SBATCH --output logs/benczechmark-cpu.%j.out

ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source ~/venvs/benczechmark/bin/activate

export PYTHONPATH=src:$PYTHONPATH

source ~/master-thesis-repository-tomas-mlynar/benczechmark_related/TASKS.sh

sbatch --array=0-$((${#TASKS[@]} - 1))  ~/master-thesis-repository-tomas-mlynar/benczechmark_related/script/eval_c4ai-command-r-08-2024.sh