#!/bin/env bash

#SBATCH -A SNIC2021-7-120      # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-00:01:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:1  # choosing no. GPUs and their type
#SBATCH -J modules             # the jobname (not necessary)

# Load PyTorch using the module tree
module purge
module load PyTorch/1.7.1-fosscuda-2020a

# Print the PyTorch version then exit
python -c "import torch; print(torch.__version__)"

