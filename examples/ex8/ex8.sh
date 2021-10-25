#!/bin/env bash
#SBATCH -A C3SE-STAFF # find your project with the "projinfo" command
#SBATCH --job-name="CIFAR with Horovod"
#SBATCH --time=1:00:0
# For a multi-node job all GPUs on a node must be requested
#SBATCH --gpus-per-node=T4:8
# For Horovod the number of tasks-per-node must match the number of GPUs per node
#SBATCH --ntasks-per-node=8
# Request multiple nodes
#SBATCH --nodes=3
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module load GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 Horovod/0.21.3-PyTorch-1.7.1 torchvision/0.8.2-PyTorch-1.7.1

# Note the use of srun to start an instance of the program for each task
srun python main.py --epochs=200
