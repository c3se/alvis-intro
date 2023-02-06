#!/bin/env bash
#SBATCH -A SNIC2022-22-1064  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=A100:4
#SBATCH -N 2
#SBATCH -J "MNMG PyTorch"  # multi node, multi GPU

echo $HOSTNAME
echo $SLURM_JOB_NODELIST

# Set-up environment
module purge
ml PyTorch/1.9.0-fosscuda-2020b

# Run DistributedDataParallel with srun (MPI backend)
srun -N $SLURM_NNODES --ntasks-per-node=$SLURM_GPUS_ON_NODE python ddp_mpi.py

## Run DistributedDataParallel with srun (NCCL backend)
#srun -N $SLURM_NNODES --ntasks-per-node=$SLURM_GPUS_ON_NODE python ddp_nccl.py
#
## Run DistributedDataParallel with torch.distributed.launch
#srun -N $SLURM_NNODES --ntasks-per-node=1 bash -c "
#python -m torch.distributed.run \
#    --node_rank="'$SLURM_NODEID'" \
#    --nnodes=$SLURM_NNODES \
#    --nproc_per_node=$SLURM_GPUS_ON_NODE \
#    --rdzv_id=$SLURM_JOB_ID \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
#    ddp_launch.py
#"
