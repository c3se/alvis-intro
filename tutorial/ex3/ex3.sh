#!/bin/env bash
#SBATCH -A C3SE-STAFF -p alvis # find your project with the "projinfo" command
#SBATCH -t 0-00:05:00
#SBATCH -J pytorch_MNIST
#SBATCH --gpus-per-node=T4:1

ml GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 PyTorch/1.4.0-Python-3.7.4 torchvision/0.7.0-Python-3.7.4-PyTorch-1.6.0 IPython

python ./ex3_main.py > results_ex3.out

