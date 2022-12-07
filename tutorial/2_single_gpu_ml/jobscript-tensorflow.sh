#!/bin/env bash
#SBATCH -A SNIC2022-22-1064           # find your project with the "projinfo" command
#SBATCH -p alvis                      # what partition to use (usually not needed)
#SBATCH -t 0-00:30:00                 # how long time it will take to run
#SBATCH --gpus-per-node=A40:1         # choosing no. GPUs and their type
#SBATCH -J regr-tf                    # the jobname (not needed)
#SBATCH -o regression-tensorflow.out  # name of the output file (not needed)

# Load modules
ml purge
ml TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1
ml matplotlib/3.4.2-foss-2021a
ml JupyterLab/3.0.16-GCCcore-10.3.0

# Interactive
#jupyter lab

# or you can instead use
#jupyter notebook

# Non-interactive
ipython -c "%run regression-tensorflow.ipynb"

# or you can instead use
#jupyter nbconvert --to python regression-tensorflow.ipynb &&
#python regression-tensorflow.py
