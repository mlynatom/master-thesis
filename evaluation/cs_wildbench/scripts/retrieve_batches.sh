#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --partition=cpufast
#SBATCH --mem-per-cpu 54G
#SBATCH --job-name wildbench-prepare-batches
#SBATCH --output /home/mlynatom/master-thesis-repository-tomas-mlynar/logs/wildbench/retrieve_batches.%J.log

ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source /home/mlynatom/master-thesis-repository-tomas-mlynar/venv/wildbench_venv/bin/activate

export PYTHONPATH=src:$PYTHONPATH

python /home/mlynatom/WildBench/src/openai_batch_eval/check_batch_status_with_id.py batch_68233d40b9388190ba8a977fda154cbd
python /home/mlynatom/WildBench/src/openai_batch_eval/check_batch_status_with_id.py batch_68233d47c0e88190852c3c53981bde57
python /home/mlynatom/WildBench/src/openai_batch_eval/check_batch_status_with_id.py batch_68233d4c54cc8190a4fbd840d147fe00
# repeat this command until all batch jobs are finished
