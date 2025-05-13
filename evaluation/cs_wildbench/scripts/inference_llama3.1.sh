#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --partition=amdgpu --gres=gpu:1
#SBATCH --mem-per-cpu 54G
#SBATCH --job-name gpu_experiments
#SBATCH --output /home/mlynatom/master-thesis-repository-tomas-mlynar/logs/wildbench/inference.%J.log

HF_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" # example model id 
MODEL_PRETTY_NAME="B+IT" # example model name
NUM_GPUS=1 # depending on your hardwares;

ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source /home/mlynatom/master-thesis-repository-tomas-mlynar/venv/wildbench_venv/bin/activate

export PYTHONPATH=src:$PYTHONPATH

bash /home/mlynatom/WildBench/scripts/_common_vllm.sh $HF_MODEL_ID $MODEL_PRETTY_NAME $NUM_GPUS 