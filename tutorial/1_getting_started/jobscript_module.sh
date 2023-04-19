#!/bin/env bash

#SBATCH -A SNIC2022-22-1064      # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-00:01:00          # how long time it will take to run
#SBATCH --gpus-per-node=T4:1  # choosing no. GPUs and their type
#SBATCH -J modules             # the jobname (not necessary)

# Load PyTorch using the module tree
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

# Print the PyTorch version then exit
python -c "import torch; print(torch.__version__)"

