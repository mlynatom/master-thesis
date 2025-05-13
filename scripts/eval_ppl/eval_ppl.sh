#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=2
#SBATCH --partition=amdgpufast --gres=gpu:1
#SBATCH --mem-per-cpu 54G
#SBATCH --job-name gpu_experiments
#SBATCH --output /home/mlynatom/master-thesis-repository-tomas-mlynar/logs/perplexity/eval_ppl_amdgpu.%J.log


ml Python/3.12.3-GCCcore-13.3.0 

source /home/mlynatom/master-thesis-repository-tomas-mlynar/venv/master_venv/bin/activate

MODEL_ID="/mnt/personal/mlynatom/thesis_models/it-Llama-3.1-8B-Instruct-mix_11_cs_en_alpaca_dolly/merge_16bit"
OUTPUT_NAME="b+it->it_(cs+en(ad))"

python /home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation/perplexity/eval_perplexity.py \
    --model_id  $MODEL_ID \
    --device "cuda" \
    --batch_size 4 \
    --max_length 1024 \
    --dataset_id "HuggingFaceFW/fineweb-2" \
    --split "test" \
    --subset "ces_Latn" \
    --output_dir "/home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation/perplexity/" \
    --add_start_token \
    --output_name $OUTPUT_NAME \

python /home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation/perplexity/eval_perplexity.py \
    --model_id $MODEL_ID \
    --device "cuda" \
    --batch_size 4 \
    --max_length 1024 \
    --dataset_id "/mnt/personal/mlynatom/data/pretraining/fineweb_train_test_split" \
    --split "test" \
    --output_dir "/home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation/perplexity/" \
    --add_start_token \
    --output_name $OUTPUT_NAME \