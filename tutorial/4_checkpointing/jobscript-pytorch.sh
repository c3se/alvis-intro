#!/bin/env bash

#SBATCH -A SNIC2021-7-120  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J "Checkpoint PyTorch"

# Set-up environment
ml purge
ml PyTorch/1.8.1-fosscuda-2020b torchvision/0.9.1-fosscuda-2020b-PyTorch-1.8.1 JupyterLab/2.2.8-GCCcore-10.2.0 matplotlib/3.3.3-fosscuda-2020b

# Interactive
jupyter lab

# Non-interactive
#ipython -c "%run checkpointing-pytorch.ipynb"
