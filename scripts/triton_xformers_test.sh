#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --partition=interactive --gres=gpu:80gb:1
#SBATCH --mem-per-cpu 54G
#SBATCH --job-name jupyter_interactive_experiments
#SBATCH --output /home/mlynatom/master-thesis-repository-tomas-mlynar/logs/jupyter_interactive.%J.log

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

#standard loading of modules
ml xformers/0.0.29.post3-foss-2023b-CUDA-12.4.0 

python -m xformers.info