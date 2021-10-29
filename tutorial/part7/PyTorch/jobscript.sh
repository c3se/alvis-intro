#!/bin/env bash
#SBATCH -A SNIC2021-7-120  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=T4:8
#SBATCH -N 2
#SBATCH -J "MNMG PyTorch"  # multi node, multi GPU

echo $HOSTNAME
echo $SLURM_JOB_NODELIST

# Set-up environment
flat_modules
ml PyTorch/1.9.0-fosscuda-2020b

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=12345
export JOB_ID=$SLURM_JOB_ID
export NGPUS_PER_NODE=$(echo "$SLURM_GPUS_PER_NODE" | sed 's/[A-Z0-9]*:\([0-9]*\)*/\1/')

# Run DistributedDataParallel with srun (MPI backend)
#srun --ntasks-per-node=8 python ddp_mpi.py

# Run DistributedDataParallel with srun (NCCL backend)
#srun --ntasks-per-node=8 python ddp_nccl.py

# Run DistributedDataParallel with torch.distributed.launch
srun --ntasks-per-node=1 bash -c "
python -m torch.distributed.run \
    --node_rank="'$SLURM_NODEID'" \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NGPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    ddp_launch.py
"
