#!/bin/env bash

#SBATCH -A SNIC2021-7-120  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:1
#SBATCH -J "Data PyTorch"

# Set-up environment
flat_modules
ml PyTorch/1.8.1-fosscuda-2020b torchvision/0.9.1-fosscuda-2020b-PyTorch-1.8.1 JupyterLab/2.2.8-GCCcore-10.2.0 matplotlib/3.3.3-fosscuda-2020b

# Unpack data to TMPDIR
cd $TMPDIR
tar -xzf "$SLURM_SUBMIT_DIR/data.tar.gz"
cp data-pytorch.ipynb .

# Interactive
#jupyter lab

# or you can instead use
#jupyter notebook

# Non-interactive
ipython -c "%run regression-pytorch.ipynb"

# or you can instead use
#jupyter nbconvert --to python regression-pytorch.ipynb &&
#python regression-pytorch.py
