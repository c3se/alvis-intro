#!/bin/env bash
#SBATCH -A NAISS2024-22-219           # find your project with the "projinfo" command
#SBATCH -p alvis                      # what partition to use (usually not needed)
#SBATCH -t 0-00:30:00                 # how long time it will take to run
#SBATCH --gpus-per-node=A40:1         # choosing no. GPUs and their type
#SBATCH -J regr-tf                    # the jobname (not needed)
#SBATCH -o regression-tensorflow.out  # name of the output file (not needed)

# Load modules
ml purge
ml TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
ml matplotlib/3.7.2-gfbf-2023a
ml JupyterLab/4.0.5-GCCcore-12.3.0

# Interactive
#jupyter lab

# or you can instead use
#jupyter notebook

# Non-interactive
ipython -c "%run regression-tensorflow.ipynb"

# or you can instead use
#jupyter nbconvert --to python regression-tensorflow.ipynb &&
#python regression-tensorflow.py
