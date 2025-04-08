#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --partition=interactive --gres=gpu:1
#SBATCH --mem-per-cpu 54G
#SBATCH --job-name gpu_experiments
#SBATCH --output /home/mlynatom/master-thesis-repository-tomas-mlynar/logs/models/perplexity/eval_ppl_amdgpu.%J.log

#by Herbert Ullrich
unset LMOD_ROOT
unset MODULESHOME
unset LMOD_PKG
unset LMOD_CMD
unset LMOD_DIR
unset FPATH
unset __LMOD_REF_COUNT_MODULEPATH
unset __LMOD_REF_COUNT__LMFILES_
unset _LMFILES_
unset _ModuleTable001_
unset _ModuleTable002_

source /etc/profile.d/lmod.sh

ml Python/3.12.3-GCCcore-13.3.0 

source /home/mlynatom/master-thesis-repository-tomas-mlynar/venv/master_venv/bin/activate

python /home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation/perplexity/eval_perplexity.py \
    --model_id "/mnt/personal/mlynatom/thesis_models/cp_Llama-3.1-8B-full_cs_fineweb2_seed42_neptune_bs128_samples500000/final" \
    --device "cuda" \
    --batch_size 4 \
    --max_length 1024 \
    --dataset_id "HuggingFaceFW/fineweb-2" \
    --split "test" \
    --subset "ces_Latn" \
    --output_dir "/home/mlynatom/master-thesis-repository-tomas-mlynar/evaluation/perplexity/" \
    --add_start_token \