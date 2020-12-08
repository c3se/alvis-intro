#!/bin/bash
#SBATCH -A C3SE-STAFF -p alvis
#SBATCH -t 0-00:10:00
#SBATCH -J pytorch_MNIST
#SBATCH --gpus-per-node=V100:1



singularity exec --nv /apps/hpc-ai-containers/PyTorch/PyTorch_v1.7.0-py3.sif python ./ex3_main.py > results_ex3.out

