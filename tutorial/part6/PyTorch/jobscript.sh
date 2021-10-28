#!/bin/env bash
#SBATCH -A SNIC2021-7-119  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:2
#SBATCH -J "SNMG PyTorch"  # Single node, multiple GPUs

# Set-up environment
flat_modules
ml PyTorch/1.8.1-fosscuda-2020b

# Run DataParallel
#python dp.py

# Set up for multiprocessing
export MASTER_ADDR="$HOSTNAME"
export MASTER_PORT="75324"
ngpus=$(python -c "import torch; print(torch.cuda.device_count())")

# Run DistributedDataParallel with torch.multiprocessing
#python ddp_mp.py

# Run DistributedDataParallel with torch.distributed.launch
#python -m torch.distributed.launch --nproc_per_node=$ngpus\
#    ddp_launch.py --world_size=$ngpus

# Run DistributedDataParallel with srun (MPI)
srun --ntasks=$ngpus python ddp_mpi.py
