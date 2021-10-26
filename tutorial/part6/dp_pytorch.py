import importlib

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model_pytorch import Model
from dataset_pytorch import RandomDataset

input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0")

# Load data
data_loader = DataLoader(
    dataset=RandomDataset(input_size, data_size),
    batch_size=batch_size,
    shuffle=True,
)


model = Model(input_size, output_size)

#if torch.cuda.device_count() > 1:
#    model = nn.DataParallel(model)

model.to(device)

for data in data_loader:
    input = data.to(device)
    output = model(input)