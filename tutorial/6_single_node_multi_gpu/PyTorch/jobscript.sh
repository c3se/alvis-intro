#!/bin/env bash
#SBATCH -A SNIC2022-22-1064
#SBATCH -p alvis
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=A40:2
#SBATCH -J "SNMG PyTorch"  # Single node, multiple GPUs

# Set-up environment
module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0

# Run DataParallel
#python dp.py

# Set up for the different multiprocessing alternatives
ngpus=$SLURM_GPUS_ON_NODE
export WORLD_SIZE=$ngpus

# Run DistributedDataParallel with run
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$ngpus \
    ddp.py

# Run DistributedDataParallel with srun (MPI)
#srun --ntasks=$ngpus python ddp.py --backend=mpi
