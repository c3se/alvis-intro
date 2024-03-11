#!/bin/env bash
#SBATCH -A NAISS2024-22-219        # find your project with the "projinfo" command
#SBATCH -p alvis                   # what partition to use (usually not needed)
#SBATCH -t 0-00:30:00              # how long time it will take to run
#SBATCH --gpus-per-node=A40:1       # choosing no. GPUs and their type
#SBATCH -J regr-torch              # the jobname (not needed)
#SBATCH -o regression-pytorch.out  # name of the output file

# Load modules
module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load matplotlib/3.5.2-foss-2022a
module load JupyterLab/3.5.0-GCCcore-11.3.0

# Interactive
#jupyter lab

# or you can instead use
#jupyter notebook

# Non-interactive
ipython -c "%run regression-pytorch.ipynb"

# or you can instead use
#jupyter nbconvert --to python regression-pytorch.ipynb &&
#python regression-pytorch.py
