import os

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from model import Model
from dataset import RandomDataset


torch.set_float32_matmul_precision("high")

def setup(verbose=False):

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if verbose:
        print(f'''
=============================================
Rank: {rank}
Local rank: {local_rank}
World size: {world_size}
Master addres: {os.environ["MASTER_ADDR"]}
Master port: {os.environ["MASTER_PORT"]}
=============================================
        ''')

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    return local_rank, rank, world_size

def cleanup():
    dist.destroy_process_group()


def run_process(verbose=False):
    '''Run process

    This is what is actually run on each process.
    '''
    # Setup this process
    local_rank, rank, world_size= setup(verbose=verbose)
    
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

    device = torch.device(f"cuda:{local_rank}")
    model.to(device)

    opt = optim.SGD(model.parameters(), lr=0.01)

    # Parallelize
    model = DistributedDataParallel(model, device_ids=[device], output_device=device)

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
    run_process(verbose=True)


if __name__=="__main__":
    main()
