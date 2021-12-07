#!/bin/env bash

#SBATCH -A SNIC2021-7-120  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=T4:1
#SBATCH -J "Checkpoint TensorFlow"

# Set-up environment
flat_modules
ml TensorFlow/2.5.0-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0 matplotlib/3.3.3-fosscuda-2020b

# Interactive
jupyter lab

# Non-interactive
#ipython -c "%run checkpointing-tensorflow.ipynb"
