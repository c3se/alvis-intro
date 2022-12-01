#!/bin/env bash

#SBATCH -A SNIC2022-22-1064  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:20:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J "Data PyTorch"

# Set-up environment
ml purge
ml torchdata/0.3.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
ml torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
ml matplotlib/3.4.2-foss-2021a
ml JupyterLab/3.0.16-GCCcore-10.3.0

# Interactive (but prefer Alvis OnDemand for interactive jupyter sessions)
#jupyter lab

# or you can instead use
#jupyter notebook

# Non-interactive
ipython -c "%run data-pytorch.ipynb"

# or you can instead use
#jupyter nbconvert --to python data-pytorch.ipynb &&
#python data-pytorch.py
