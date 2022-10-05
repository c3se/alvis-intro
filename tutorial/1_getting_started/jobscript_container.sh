#!/bin/env bash

#SBATCH -A SNIC2021-7-120      # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-00:01:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:1  # choosing no. GPUs and their type
#SBATCH -J container           # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge

# Specify the path to the container
CONTAINER=/apps/containers/PyTorch/PyTorch-1.10-NGC-21.08.sif

# Print the PyTorch version then exit
singularity exec $CONTAINER python -c "import torch; print(torch.__version__)"
