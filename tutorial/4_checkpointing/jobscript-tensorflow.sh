#!/bin/env bash

#SBATCH -A NAISS2024-22-219  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J "Checkpoint TensorFlow"

# Set-up environment
ml purge
module TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
module matplotlib/3.7.2-gfbf-2023a
module JupyterLab/4.0.5-GCCcore-12.3.0

# Interactive
jupyter lab

# Non-interactive
#ipython -c "%run checkpointing-tensorflow.ipynb"
