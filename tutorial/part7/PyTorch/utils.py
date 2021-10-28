# This function is mainly based on work by IDRIS
# http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html
import os
import subprocess

import torch


def get_hostnames():
    # An alternative would be to use hostlist from NSC
    return subprocess.run(
        'scontrol show hostnames "$SLURM_JOB_NODELIST"',
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\n")[:-1]


# Define useful variables
rank = int(os.environ["SLURM_PROCID"])
local_rank = int(os.environ["SLURM_LOCALID"])
world_size = int(os.environ["SLURM_NTASKS"])
gpus_per_node = int(os.environ["SLURM_GPUS_PER_NODE"].split(":")[-1])
hostnames = get_hostnames()

# Define MASTER_ADDR and MASTER_PORT
os.environ["MASTER_ADDR"] = hostnames[0]
os.environ["MASTER_PORT"] = "12345"
