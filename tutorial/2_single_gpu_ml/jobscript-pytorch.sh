#!/bin/env bash
#SBATCH -A SNIC2022-22-1064          # find your project with the "projinfo" command
#SBATCH -p alvis                   # what partition to use (usually not needed)
#SBATCH -t 0-00:30:00              # how long time it will take to run
#SBATCH --gpus-per-node=A40:1       # choosing no. GPUs and their type
#SBATCH -J regr-torch              # the jobname (not needed)
#SBATCH -o regression-pytorch.out  # name of the output file

# Load modules
ml purge
ml torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
ml matplotlib/3.4.2-foss-2021a
ml JupyterLab/3.0.16-GCCcore-10.3.0

# Interactive
#jupyter lab

# or you can instead use
#jupyter notebook

# Non-interactive
ipython -c "%run regression-pytorch.ipynb"

# or you can instead use
#jupyter nbconvert --to python regression-pytorch.ipynb &&
#python regression-pytorch.py
