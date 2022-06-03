#!/bin/env bash
#SBATCH -A SNIC2021-7-120  # find your project with the "projinfo" command
#SBATCH -p alvis
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=A100:4
#SBATCH -N 2
#SBATCH -J "MNMG PyTorch"  # multi node, multi GPU

echo $HOSTNAME
echo $SLURM_JOB_NODELIST

# Set-up environment
module purge
ml PyTorch/1.9.0-fosscuda-2020b

export NGPUS_PER_NODE=${SLURM_GPUS_PER_NODE#*:}

# Run DistributedDataParallel with srun (MPI backend)
srun --ntasks-per-node=$NGPUS_PER_NODE python ddp_mpi.py

# Run DistributedDataParallel with srun (NCCL backend)
srun --ntasks-per-node=$NGPUS_PER_NODE python ddp_nccl.py

# Run DistributedDataParallel with torch.distributed.launch
srun -N $SLURM_NNODES --ntasks-per-node=1 bash -c "
python -m torch.distributed.run \
    --node_rank="'$SLURM_NODEID'" \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NGPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    ddp_launch.py
"
