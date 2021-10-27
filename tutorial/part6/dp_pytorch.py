import torch
from torch import nn, optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from model_pytorch import Model
from dataset_pytorch import RandomDataset


def run_process():
    '''Run process

    This is what is actually run on each process and in this case for
    DataParallel this is the only process being run.
    '''
    
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

    device = torch.device("cuda:0")
    model.to(device)

    opt = optim.SGD(model.parameters(), lr=0.01)

    # Parallelize
    if torch.cuda.device_count() > 1:
        # This is were the magic happens. This line is the only difference
        # between running on a single GPU or multiple GPUs.
        model = DataParallel(model)
    

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

    return model


def main():
    run_process()


if __name__=="__main__":
    main()