import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision.models as models
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np 

import platform,psutil
import time,os
import pandas as pd



def train(model, device, train_loader, optimizer, epoch, dry_run=False):
    model.train()
    log_interval = 10
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



use_cuda = True

device = torch.device("cuda" if use_cuda else "cpu")

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

train_set = datasets.MNIST('local_dataset', download=True, train=True, transform=transform)
test_set = datasets.MNIST('local_dataset', train=False, transform=transform)


pinning = True   # Control pinning of the data to the memory location

train_data = DataLoader(train_set, batch_size=64, num_workers=1, pin_memory=pinning, 
                        shuffle=True)
test_data = DataLoader(test_set, batch_size=64, num_workers=1, pin_memory=pinning, 
                       shuffle=True)



# Setting up the model
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)   # learning rate = 1.0
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)


epochs = 14
save_model = False

for epoch in range(1, epochs + 1):
    train(model, device, train_data, optimizer, epoch, dry_run=False)
    test(model, device, test_data)
    scheduler.step()

if save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")
