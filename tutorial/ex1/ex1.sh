#!/bin/sh

# This example shows how to prepare a job script and use the modules to
# run a training example with TensorFlow

#SBATCH -A C3SE-STAFF
#SBATCH -p alvis
#SBATCH -t 00-00:05:00

#SBATCH -n 1
#SBATCH --gpus-per-node=V100:1


ml GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 TensorFlow/2.3.1-Python-3.7.4
ml Pillow/6.2.1

python cnn_with_own_data_ex1.py > res_ex1.out
