#!/bin/env bash
#SBATCH -A C3SE-STAFF # find your project with the "projinfo" command
#SBATCH --job-name="Alvis ex7 with horovod"
#SBATCH --time=1:10:0
# For a multi-node job all GPUs on a node must be requested
#SBATCH --gpus-per-node=T4:8
# For Horovod the number of tasks-per-node must match the number of GPUs per node
#SBATCH --ntasks-per-node=8
# Request multiple nodes
#SBATCH --nodes=1
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

CONTAINER=$HOME/horovod-torch-tensorboard.sif
echo $NP

# Note the use of srun to start an instance of the program for each task
srun singularity exec --nv $CONTAINER python3 ex7.py # --epochs=160
