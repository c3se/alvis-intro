#!/bin/env bash
#SBATCH -A C3SE-STAFF -p alvis # find your project with the "projinfo" command
#SBATCH -t 0-00:30:00
#SBATCH -J pytorch_dataset
#SBATCH --gpus-per-node=T4:2

CONTAINER=/apps/hpc-ai-containers/PyTorch/PyTorch_v1.7.0-py3.sif

# Don't forget the --nv flag, else your containers won't see the GPUs!
singularity exec --nv $CONTAINER python ./ex4_main.py

