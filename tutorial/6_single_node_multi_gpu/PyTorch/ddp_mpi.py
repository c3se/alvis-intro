import sys

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from model import Model
from dataset import RandomDataset


def setup(verbose=True):
    dist.init_process_group("mpi")
    rank = dist.get_rank()

    if verbose:
        print(f'''
=============================================
Rank: {dist.get_rank()}
World size: {dist.get_world_size()}
Master addres: {os.environ["MASTER_ADDR"]}
Master port: {os.environ["MASTER_PORT"]}
=============================================
        ''')
    
    return rank


def cleanup():
    dist.destroy_process_group()


def run_process():
    '''Run process

    This is what is actually run on each process.
    '''
    # Setup this process
    rank = setup(verbose=True)
    
    # Initialize data_loader
    input_size = 500
    output_size = 1
    batch_size = 30
    data_size = 100

    data_loader = DataLoader(
        dataset=RandomDataset(input_size, data_size),
        batch_size=batch_size,
        shuffle=True,
    )

    # Initialize model and attach to optimizer
    model = Model(input_size, output_size, verbose=True)

    device = torch.device(f"cuda:{rank}")
    model.to(device)

    opt = optim.SGD(model.parameters(), lr=0.01)

    # Parallelize
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # Actual training
    n_epochs = 3
    for epoch in range(n_epochs):
        model.train()
        for data, target in data_loader:
            opt.zero_grad()

            input = data.to(device)
            target = target.to(device)
            output = model(input)

            loss = (output - target).pow(2).mean(0)
            loss.backward()
            opt.step()
        
        if rank==0:
            print(f"Epoch {epoch}")
            sys.stdout.flush()  # to compare between processes

    # Cleanup process
    cleanup()

    return model


def main():
    # Spawn processes
    run_process()


if __name__=="__main__":
    main()