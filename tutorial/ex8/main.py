#!/usr/bin/env python

import argparse
import os
import pickle
from filelock import FileLock
from datetime import datetime

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
import horovod.torch as hvd

import resnet

# Training settings
parser = argparse.ArgumentParser(description='CIFAR')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--base-lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--data-dir', default='/cephyr/NOBACKUP/Datasets/CIFAR',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')


def get_dataset(data_dir, train=True, fine=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transfs = []
    if train:
        transfs +=  [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
        ]
    transfs = [
        transforms.ToTensor(),
        normalize,
    ]
    transf = transforms.Compose(transfs)

    Dataset = datasets.CIFAR100 if fine else datasets.CIFAR10
    return Dataset(root=data_dir, train=train, transform=transf)


def train(epoch, model, train_sampler, train_loader, optimizer, loss_function, args):
    running_loss = 0.0
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test(model, test_sampler, test_loader):
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum().item()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))
    
    return test_loss, test_accuracy


def main():
    args = parser.parse_args()

    # Set-up tensorboard

    # Horovod: initialize library.
    seed = 42
    hvd.init()
    torch.manual_seed(seed)

    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (hasattr(mp, '_supports_context') and
        mp._supports_context and
        'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    data_dir = args.data_dir
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_dataset = get_dataset(data_dir, train=True)

    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_dataset = get_dataset(data_dir, train=False)

    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)

    model = resnet.resnet110()
    loss_function = nn.CrossEntropyLoss()
    running_loss = 0.0

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr * lr_scaler)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average)

    # Profile training
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_sampler, train_loader, optimizer, loss_function, args)
        test_loss, test_acc = test(model, test_sampler, test_loader)

        if hvd.rank()==0:
            print("Epoch: ", epoch, "Test loss:", test_loss, ", Test acc.", test_acc)


if __name__ == '__main__':
    main()