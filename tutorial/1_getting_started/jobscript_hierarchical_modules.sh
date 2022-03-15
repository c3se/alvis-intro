#!/bin/env bash

#SBATCH -A SNIC2021-7-120            # find your project with the "projinfo" command
#SBATCH -p alvis                 # what partition to use (usually not necessary)
#SBATCH -t 0-00:01:00            # how long time it will take to run
#SBATCH --gpus-per-node=T4:1     # choosing no. GPUs and their type
#SBATCH -J hierarchical_modules  # the jobname (not necessary)

# Load using the hierarchical module tree
hierarchical_modules
module purge
module load GCC/9.3.0  CUDA/11.0.2  OpenMPI/4.0.3
mudule load PyTorch/1.7.1-Python-3.8.2

# Print the PyTorch version then exit
python -c "import torch; print(torch.__version__)"
