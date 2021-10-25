#!/bin/env bash

# This example shows how to use the available datasets to
# train a CNN  with TensorFlow. Investigate the cnn_with_cephyr_data_ex2.py
# file to find out how to directly access the available datasets from your code.

#SBATCH -A C3SE-STAFF  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 01-00:00:00
#SBATCH --gpus-per-node=T4:1
#SBATCH -e slurm-%j.err

ml GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5 TensorFlow/2.5.0 Pillow/8.0.1

# The u is to run python in unbuffered mode to get more feedback on the process
python -u cnn_with_cephyr_data_ex2.py

