#!/bin/env bash
#SBATCH -A C3SE-STAFF # find your project with the "projinfo" command
#SBATCH --job-name="Alvis ex8"
#SBATCH --time=0:10:0
# For a multi-node job all GPUs on a node must be requested
#SBATCH --gpus-per-node=T4:8
# For Horovod the number of tasks-per-node must match the number of GPUs per node
#SBATCH --ntasks-per-node=8
# Request multiple nodes
#SBATCH --nodes=3
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

ml purge > /dev/null 2>&1
module load GCC/10.1.0 \
            CUDA/11.1.1 \
            OpenMPI/4.0.5 \
            Horovod/0.21.3-PyTorch-1.7.1

# Download model if not available
if [ ! -f pytorch_model.bin ]; then
    wget https://huggingface.co/glasses/efficientnet_b2/blob/main/pytorch_model.bin
fi
#export OMPI_MCA_mpi_warn_on_fork=0

# Note the use of srun to start an instance of the program for each task
srun python ex8.py
