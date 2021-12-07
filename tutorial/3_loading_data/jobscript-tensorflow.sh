#!/bin/env bash

#SBATCH -A SNIC2021-7-120  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:1
#SBATCH -J "Data TensorFlow"

# Set-up environment
flat_modules
ml TensorFlow/2.5.0-fosscuda-2020b matplotlib/3.3.3-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0

# Unpack data to TMPDIR
cd $TMPDIR
tar -xzf "$SLURM_SUBMIT_DIR/data.tar.gz"
cp "$SLURM_SUBMIT_DIR/data-tensorflow.ipynb" .

# Interactive
#jupyter lab

# or you can instead use
#jupyter notebook

# Non-interactive
ipython -c "%run data-tensorflow.ipynb"

# or you can instead use
#jupyter nbconvert --to python data-tensorflow.ipynb &&
#python data-tensorflow.py
