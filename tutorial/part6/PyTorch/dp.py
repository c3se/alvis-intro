import torch
from torch import nn, optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from model import GPT
from dataset import RandomCorpus


def run_process():
    '''Run process

    This is what is actually run on each process and in this case for
    DataParallel this is the only process being run.
    '''
    
    # Initialize data_loader
    context_size = 1024
    batch_size = 32
    corpus_length = 128
    vocab_size = 2**8

    data_loader = DataLoader(
        dataset=RandomCorpus(corpus_length, context_size, vocab_size),
        batch_size=batch_size,
        shuffle=True,
    )

    # Initialize model and attach to optimizer
    model = GPT(vocab_size, context_size, verbose=True)

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

    # Actual training
    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        for sequence in data_loader:
            opt.zero_grad()

            sequence = sequence.to(device)
            logits = model(sequence)

            # Shift so that prediction is next token for each token
            logits = logits[..., :-1, :].contiguous()
            target = sequence[..., 1:].contiguous()

            # Flatten the tokens
            loss = loss_func(
                logits.flatten(end_dim=-2),
                target.flatten(),
            )
            loss.backward()
            opt.step()
        
        print(epoch)

    return model


def main():
    run_process()


if __name__=="__main__":
    main()