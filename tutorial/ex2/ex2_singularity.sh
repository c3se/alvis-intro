#!/bin/sh

# This example shows how to use the available datasets to
# train a CNN  with TensorFlow from an NGC container image. Investigate the cnn_with_own_data_ex2.py file to find 
# out how to directly access the available datasets from your code.

#SBATCH -A C3SE-STAFF
#SBATCH -p alvis
#SBATCH -t 01-00:00:00
#SBATCH --gpus-per-node=V100:3


# Don't forget the --nv flag, else your containers won't see the GPUs!
singularity exec --nv /apps/hpc-ai-containers/TensorFlow/TensorFlow_v2.1.0-tf2-py3-NGC-R20.03.sif python cnn_with_own_data_ex2.py > res_ex2.out
