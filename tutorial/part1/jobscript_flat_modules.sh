#!/bin/env bash

#SBATCH -A C3SE-STAFF         # find your project with the "projinfo" command
#SBATCH -p alvis              # what partition to use (usually not necessary)
#SBATCH -t 0-00:01:00         # how long time it will take to run
#SBATCH --gpus-per-node=T4:1  # choosing no. GPUs and their type
#SBATCH -J flat_modules       # the jobname (not necessary)

# Load PyTorch using the flat module tree
# Note: you need to have switched by calling flat_modules
module purge
module load PyTorch/1.7.1-fosscuda-2020a-Python-3.8.2

# Print the PyTorch version then exit
python -c "import torch; print(torch.__version__)"
