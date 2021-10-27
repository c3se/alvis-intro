import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from model_pytorch import Model
from dataset_pytorch import RandomDataset


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--world_size", type=int)
args = parser.parse_args()


def setup(rank, world_size, verbose=False):
    if verbose:
        print(f'''
=============================================
Rank: {rank}
World size: {world_size}
Master addres: {os.environ["MASTER_ADDR"]}
Master port: {os.environ["MASTER_PORT"]}
=============================================
        ''')
    
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_process(rank, world_size):
    '''Run process

    This is what is actually run on each process.
    '''
    # Setup this process
    setup(rank, world_size, verbose=True)
    
    # Initialize data_loader
    input_size = 5
    output_size = 1
    batch_size = 30
    data_size = 100

    data_loader = DataLoader(
        dataset=RandomDataset(input_size, data_size),
        batch_size=batch_size,
        shuffle=True,
    )

    # Initialize model and attach to optimizer
    model = Model(input_size, output_size, verbose=False)

    device = torch.device(f"cuda:{rank}")
    model.to(device)

    opt = optim.SGD(model.parameters(), lr=0.01)

    # Parallelize
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # Actual training
    n_epochs = 10
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
        
        print(epoch)

    # Cleanup process
    cleanup()

    return model


def main():
    # Spawn processes
    rank = args.local_rank
    world_size = torch.cuda.device_count()
    run_process(rank, world_size)


if __name__=="__main__":
    main()