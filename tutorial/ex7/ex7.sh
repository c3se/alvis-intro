#!/bin/env bash
#SBATCH -A C3SE-STAFF # find your project with the "projinfo" command
#SBATCH --job-name="Alvis ex7 with horovod"
#SBATCH --time=1:10:0
# For a multi-node job all GPUs on a node must be requested
#SBATCH --gpus-per-node=T4:8
# For Horovod the number of tasks-per-node must match the number of GPUs per node
#SBATCH --ntasks-per-node=8
# Request multiple nodes
#SBATCH --nodes=3
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

ml purge > /dev/null 2>&1
module load fosscuda/2019b
module load Horovod/0.20.3-TensorFlow-2.3.1-Python-3.7.4

# Download MNIST in pickled-format 
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
export OMPI_MCA_mpi_warn_on_fork=0

# Note the use of srun to start an instance of the program for each task
srun python ex7.py --epochs=160
