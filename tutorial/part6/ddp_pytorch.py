import importlib

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model_pytorch import Model
from dataset_pytorch import RandomDataset


def train(
    model,
    data_loader,
    opt,
    loss_func=nn.MSELoss(),
    n_epochs=10,
):
    device = torch.device("cuda:0")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)


    model.train()

    for epoch in range(n_epochs):
        for data, target in data_loader:
            opt.zero_grad()

            input = data.to(device)
            target = target.to(device)
            output = model(input)

            loss = loss_func(output, target)
            loss.backward()
            opt.step()
        
        print(epoch)


def main():
    input_size = 5
    output_size = 1

    batch_size = 30
    data_size = 100

    data_loader = DataLoader(
        dataset=RandomDataset(input_size, data_size),
        batch_size=batch_size,
        shuffle=True,
    )

    model = Model(input_size, output_size)

    opt = optim.SGD(model.parameters(), lr=0.01)

    train(model, data_loader, opt)


if __name__=="__main__":
    main()