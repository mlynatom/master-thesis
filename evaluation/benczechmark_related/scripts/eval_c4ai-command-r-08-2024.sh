#!/bin/bash
#SBATCH --partition amdgpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 96G
#SBATCH --gres gpu:2
#SBATCH --time 8:00:00
#SBATCH --job-name benczechmark_eval
#SBATCH --output logs/benczechmark-command-r-08-2024.%j.out

NAME='c4ai-command-r-08-2024-awq'
MODEL_NAME='AMead10/c4ai-command-r-08-2024-awq'

# set up run settings
CHAT_TEMPLATE="none"
TRUNCATE_STRATEGY="leave_description"

ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source ~/venvs/benczechmark/bin/activate

export PYTHONPATH=src:$PYTHONPATH

source ~/master-thesis-repository-tomas-mlynar/benczechmark_related/TASKS.sh
source ~/master-thesis-repository-tomas-mlynar/benczechmark_related/NUM_SHOT.sh

export CACHE_NAME="realrun_benczechmark_${NAME}_cache_${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Executing TASK: ${TASKS[$SLURM_ARRAY_TASK_ID]}"

# Set run script
SCRIPT="/home/mlynatom/master-thesis-repository-tomas-mlynar/benczechmark_related/script/eval_vllm.sh"
SUM_LOGP_FLAG="no"
for task in "${SUM_LOGPROBS[@]}"; do
  if [ "$task" == "${TASKS[$SLURM_ARRAY_TASK_ID]}" ]; then
    SUM_LOGP_FLAG="yes"
    CHAT_TEMPLATE="none"
    TRUNCATE_STRATEGY="none"
    NUM_FEWSHOT=0
    break
  fi
done

OUTPUT_PATH="results/eval_${NAME}_${TASKS[$SLURM_ARRAY_TASK_ID]}_chat_${CHAT_TEMPLATE}_trunc_${TRUNCATE_STRATEGY}"
LOGFILE="logs_benczechmark/eval_${NAME}_array_${TASKS[$SLURM_ARRAY_TASK_ID]}_chat_${CHAT_TEMPLATE}_trunc_${TRUNCATE_STRATEGY}.log"

set -x # enables a mode of the shell where all executed commands are printed to the terminal
# Run the script with the task specified by SLURM_ARRAY_TASK_ID
time $SCRIPT "${TASKS[$SLURM_ARRAY_TASK_ID]}" "$OUTPUT_PATH" "$SUM_LOGP_FLAG" "$CHAT_TEMPLATE" "$TRUNCATE_STRATEGY" "$NUM_FEWSHOT" "$MODEL_NAME" | tee -a "$LOGFILE"
set +x