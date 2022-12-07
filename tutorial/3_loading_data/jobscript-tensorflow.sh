#!/bin/env bash

#SBATCH -A SNIC2022-22-1064  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J "Data TensorFlow"

# Set-up environment
ml purge
ml TensorFlow-Datasets/4.7.0-foss-2021a-CUDA-11.3.1
ml matplotlib/3.3.3-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0
ml JupyterLab/2.2.8-GCCcore-10.2.0

# # Unpack data to TMPDIR
# # uncomment if you want to try to read data from directory instead of archive
# cd $TMPDIR
# tar -xzf "$SLURM_SUBMIT_DIR/data.tar.gz"
# cp "$SLURM_SUBMIT_DIR/data-tensorflow.ipynb" .

# Interactive
#jupyter lab

# or you can instead use
#jupyter notebook

# Non-interactive
ipython -c "%run data-tensorflow.ipynb"

# or you can instead use
#jupyter nbconvert --to python data-tensorflow.ipynb &&
#python data-tensorflow.py
