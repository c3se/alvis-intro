from typing import Iterable

import torch
from torch.utils.data import TensorDataset


class RandomDataset(TensorDataset):

    def __init__(self, data_size, length, n_classes=10):
        if isinstance(data_size, int):
            data_size = (data_size,)
        data = torch.randn(length, *data_size)
        target = torch.randint(high=n_classes, size=(length,1))
        super().__init__(data, target)
