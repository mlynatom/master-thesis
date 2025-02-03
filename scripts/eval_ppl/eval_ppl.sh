#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --partition=amdgpu --gres=gpu:1
#SBATCH --mem-per-cpu 54G
#SBATCH --job-name gpu_experiments
#SBATCH --output /home/mlynatom/master-thesis-repository-tomas-mlynar/logs/models/perplexity/eval_ppl_amdgpu.%J.log


ml Python/3.12.3-GCCcore-13.3.0 

source /home/mlynatom/master-thesis-repository-tomas-mlynar/venv/master_venv/bin/activate

python /home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation/perplexity/eval_perplexity.py \
    --model_id "/home/mlynatom/master-thesis-repository-tomas-mlynar/models/llama3.2-3b-instruct-1epoch-32batch-4gradacc-1e-4lr-merged" \
    --device "cuda" \
    --load_in_16bit True \
    --batch_size 4 \
    --max_length 4096 \
    --dataset_id "HuggingFaceFW/fineweb-2" \
    --split "test" \
    --subset "ces_Latn" \
    --output_dir "/home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation/perplexity/" \