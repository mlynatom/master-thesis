#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --partition=amdgpu --gres=gpu:1
#SBATCH --mem-per-cpu 54G
#SBATCH --job-name gpu_experiments
#SBATCH --output /home/mlynatom/master-thesis-repository-tomas-mlynar/logs/perplexity/eval_ppl_amdgpu.%J.log


ml Python/3.12.3-GCCcore-13.3.0 

source /home/mlynatom/master-thesis-repository-tomas-mlynar/venv/master_venv/bin/activate

python /home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation/perplexity/eval_perplexity.py \
    --model_id "meta-llama/Llama-3.1-8B" \
    --device "cuda" \
    --load_in_4bit \
    --batch_size 4 \
    --max_length 1024 \
    --dataset_id "/mnt/personal/mlynatom/data/pretraining/fineweb_train_test_split" \
    --split "test" \
    --output_dir "/home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation/perplexity/" \
    --add_start_token \