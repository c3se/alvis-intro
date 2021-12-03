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
from logger import BenchmarkWriter


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
    context_size = 512
    batch_size = 32
    corpus_length = 1024
    vocab_size = 2**8

    data_loader = DataLoader(
        dataset=RandomCorpus(corpus_length, context_size, vocab_size),
        batch_size=batch_size,
        shuffle=True,
    )

    # Initialize model and attach to optimizer
    model = GPT(vocab_size, context_size, verbose=False)

    device = torch.device(f"cuda:{rank}")
    model.to(device)

    learning_rate = 6e-4 * 5e5 / (batch_size * context_size)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    # Parallelize
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Initialize logger instance to see performance
    writer = BenchmarkWriter()

    # Actual training
    global_step = 0
    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        for sequence in data_loader:
            opt.zero_grad()

            # Shift so that prediction is next token for each token
            sequence = sequence.to(device)
            logits = model(sequence[..., :-1].contiguous())
            target = sequence[..., 1:].contiguous()

            # Flatten the tokens when calculating loss
            loss = loss_func(
                logits.flatten(end_dim=-2),
                target.flatten(),
            )
            loss.backward()
            opt.step()
            
            # This will also log the wall time
            if rank==0:
                global_step += batch_size
                writer.add_scalar("Loss", loss.item(), global_step=global_step)
        
        if rank==0:
            print("Epoch:", epoch)

    if rank==0:
        writer.benchmark_results(burn_in=12, step_unit="seq")
    writer.close()

    # Cleanup process
    cleanup()

    return model


def main():
    # Spawn processes
    run_process()


if __name__=="__main__":
    main()