import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from model import GPT
from dataset import RandomCorpus


def setup(verbose=False):
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
    context_size = 1024
    context_size
    batch_size = 64
    corpus_length = 1024

    data_loader = DataLoader(
        dataset=RandomCorpus(input_size, data_size),
        batch_size=batch_size,
        shuffle=True,
    )

    # Initialize model and attach to optimizer
    model = GPT(input_size, output_size, verbose=False)

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
        
        if rank==0:
            print(epoch)

    # Cleanup process
    cleanup()

    return model


def main():
    # Spawn processes
    run_process()


if __name__=="__main__":
    main()