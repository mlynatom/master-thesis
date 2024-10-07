#!/bin/bash
#SBATCH --partition amdgpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 96G
#SBATCH --gres gpu:2
#SBATCH --time 1:00:00
#SBATCH --job-name benczechmark_eval
#SBATCH --output logs/test_benczechmark_command-r.%j.out


ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source ~/venvs/benczechmark/bin/activate

export PYTHONPATH=src:$PYTHONPATH

python -m lm_eval \
    --model vllm \
    --model_args pretrained=AMead10/c4ai-command-r-08-2024-awq,tensor_parallel_size=2,dtype=float16,gpu_memory_utilization=0.5,max_length=2048,normalize_log_probs=True,trust_remote_code=False,truncate_strategy=leave_description,quantization=awq \
    --tasks benczechmark_ctkfacts_nli \
    --batch_size auto:4 \
    --output_path results_test_benczechmark/eval_command-r \
    --log_samples \
    --verbosity DEBUG \
    --num_fewshot 3