#!/bin/env bash

# This example shows how to use the available datasets to
# train a CNN  with TensorFlow. Investigate the cnn_with_own_data_ex2.py file to find out how to directly access the available datasets from your code.

#SBATCH -A C3SE-STAFF  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 01-00:00:00
#SBATCH --gpus-per-node=T4:2

ml GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 TensorFlow/2.3.1-Python-3.7.4 Pillow/6.2.1

python cnn_with_cephyr_data_ex2.py

