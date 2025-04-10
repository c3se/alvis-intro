import sys

import torch
from torch import nn, optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from model import GPT
from dataset import RandomCorpus
from logger import BenchmarkWriter


torch.set_float32_matmul_precision("high")

def run_process():
    '''Run process

    This is what is actually run on each process and in this case for
    DataParallel this is the only process being run.
    '''
    
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

    device = torch.device("cuda:0")
    model.to(device)

    learning_rate = 6e-4 * 5e5 / (batch_size * context_size)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    # Parallelize
    if torch.cuda.device_count() > 1:
        # This is were the magic happens. This line is the only difference
        # between running on a single GPU or multiple GPUs.
        model = DataParallel(model)

    # Initialize logger instance to see performance
    writer = BenchmarkWriter()

    # Actual training
    global_step = 0
    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
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
            global_step += batch_size
            writer.add_scalar("Loss", loss.item(), global_step=global_step)
        
        print("Epoch:", epoch)

    writer.benchmark_results(burn_in=12, step_unit="seq")
    writer.close()

    return model


def main():
    run_process()


if __name__=="__main__":
    main()
