#!/bin/env bash

#SBATCH -A SNIC2022-22-1064  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J "Profile TensorFlow"

# Set-up environment
module purge
ml purge
ml TensorFlow/2.5.0-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0 matplotlib/3.3.3-fosscuda-2020b

# Interactive
jupyter lab

# Non-interactive
#ipython -c "%run profiling-tensorflow.ipynb"
