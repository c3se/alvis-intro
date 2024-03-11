#!/bin/env bash

#SBATCH -A NAISS2024-22-219  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -J "Data TensorFlow"

# Set-up environment
ml purge
ml TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
ml matplotlib/3.5.2-foss-2022a
ml JupyterLab/3.5.0-GCCcore-11.3.0

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
