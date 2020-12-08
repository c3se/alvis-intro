#!/bin/bash
#SBATCH -A C3SE-STAFF -p alvis
#SBATCH -t 0-00:30:00
#SBATCH -J pytorch_dataset
#SBATCH --gpus-per-node=V100:3



singularity exec --nv /apps/hpc-ai-containers/PyTorch/PyTorch_v1.7.0-py3.sif python ./ex4_main.py > results_ex4.out

