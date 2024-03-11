#!/bin/env bash
#SBATCH -A NAISS2024-22-219
#SBATCH -p alvis
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=A100:4
#SBATCH -N 2
#SBATCH -J "MNMG PyTorch"  # multi node, multi GPU

echo $HOSTNAME
echo $SLURM_JOB_NODELIST

# Set-up environment
module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0

# Run DistributedDataParallel with srun (MPI backend)
srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=$SLURM_GPUS_ON_NODE python ddp_mpi.py

## Run DistributedDataParallel with srun (NCCL backend)
#srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=$SLURM_GPUS_ON_NODE python ddp_nccl.py
#
## Run DistributedDataParallel with torch.distributed.launch
#srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
#torchrun \
#    --node_rank="'$SLURM_NODEID'" \
#    --nnodes=$SLURM_JOB_NUM_NODES \
#    --nproc_per_node=$SLURM_GPUS_ON_NODE \
#    --rdzv_id=$SLURM_JOB_ID \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#    ddp_launch.py
#"
