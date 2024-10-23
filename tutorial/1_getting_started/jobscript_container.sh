#!/bin/env bash

#SBATCH -A NAISS2024-22-219      # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-00:01:00          # how long time it will take to run
#SBATCH --gpus-per-node=T4:1  # choosing no. GPUs and their type
#SBATCH -J container           # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge

# Specify the path to the container
CONTAINER=/apps/containers/PyTorch/PyTorch-NGC-latest.sif

# Print the PyTorch version then exit
singularity exec $CONTAINER python -c "import torch; print(torch.__version__)"
