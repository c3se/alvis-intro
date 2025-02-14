#!/bin/env bash
#SBATCH -A NAISS2024-22-219
#SBATCH -p alvis
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=A40:2
#SBATCH -J "SNMG PyTorch"  # Single node, multiple GPUs

# Set-up environment
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

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
