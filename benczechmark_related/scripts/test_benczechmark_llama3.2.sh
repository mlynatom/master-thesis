#!/bin/bash
#SBATCH --partition amdgpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 96G
#SBATCH --gres gpu:1
#SBATCH --time 24:00:00
#SBATCH --job-name benczechmark_eval
#SBATCH --output logs/benczechmark.%j.out


ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source ~/venvs/benczechmark/bin/activate

export PYTHONPATH=src:$PYTHONPATH

python -m lm_eval \
    --model vllm \
    --model_args pretrained=meta-llama/Llama-3.2-3B,tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.8,max_length=2048,normalize_log_probs=True,trust_remote_code=False,truncate_strategy=leave_description \
    --tasks benczechmark_ctkfacts_nli \
    --batch_size auto:4 \
    --output_path results_benczechmark/eval_Llama-3.2-3B_benczechmark_ctkfacts_nli_chat_none_trunc_leave_description_v2 \
    --log_samples \
    --verbosity DEBUG \
    --num_fewshot 3