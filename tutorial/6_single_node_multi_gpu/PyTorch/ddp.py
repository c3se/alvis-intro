import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.elastic.multiprocessing.errors import record

from model import GPT
from dataset import RandomCorpus
from logger import BenchmarkWriter


torch.set_float32_matmul_precision("high")

def setup(backend, verbose=False):
    dist.init_process_group(backend)
    if verbose:
        print(f'''
=============================================
  Rank:          {dist.get_rank()}
  World size:    {dist.get_world_size()}
  Master addres: {os.environ["MASTER_ADDR"]}
  Master port:   {os.environ["MASTER_PORT"]}
=============================================
        ''')


def cleanup():
    dist.destroy_process_group()


@record  # to get traceback for "Root Cause"
def run_process():
    '''Run process

    This is what is actually run on each process.
    '''
    # Get distributed parameters
    rank = dist.get_rank()
    local_rank = int(os.getenv("LOCAL_RANK", os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", os.environ["SLURM_LOCALID"])))
    world_size = dist.get_world_size()
    
    # Initialize data_loader
    context_size = 512
    batch_size = 32
    corpus_length = 1024
    vocab_size = 2**8

    dataset = RandomCorpus(corpus_length, context_size, vocab_size)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
    )

    # Initialize model
    model = GPT(vocab_size, context_size, verbose=False)

    device = torch.device(f"cuda:{local_rank}")
    model.to(device)

    # Prepare for distributed data parallelism 
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # The learning rate is adapted for the total batch_size in tokens
    learning_rate = 6e-4 * (batch_size * world_size * context_size / 5e5)
    # ZeroRedundancyOptimizer reduces the memory footprint of the Optimizer
    opt= ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=optim.Adam,
        lr=learning_rate,
    )
    loss_func = nn.CrossEntropyLoss()

    # Initialize logger instance to see performance
    writer = BenchmarkWriter()

    # Actual training
    global_step = 0
    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        sampler.set_epoch(epoch)  # for correct shuffling
        for sequence, in data_loader:
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
                global_step += batch_size * world_size
                writer.add_scalar("Loss", loss.item(), global_step=global_step)
        
        if rank==0:
            print("Epoch:", epoch)

    if rank==0:
        pass
        #writer.benchmark_results(burn_in_steps=2*corpus_length, step_unit="seq")
    writer.close()

    return model


def main(args):
    # Run processes
    setup(args.backend, verbose=True)
    run_process()
    cleanup()


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch DDP example for Alvis Intro tutorial",
    )
    parser.add_argument(
        "--backend",
        choices=["nccl", "mpi", "gloo"],
        default="nccl",
        help="Choice of backend to torch.distributed.init_process_group",
    )
    args = parser.parse_args()

    main(args)
