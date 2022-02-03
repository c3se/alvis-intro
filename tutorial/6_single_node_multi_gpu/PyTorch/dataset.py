import torch
from torch.utils.data import Dataset, TensorDataset


class RandomCorpus(TensorDataset):

    def __init__(self, n_sentences, context_size, vocab_size):
        super().__init__(torch.randint(size=(n_sentences, context_size), high=vocab_size))


class RandomDataset(TensorDataset):

    def __init__(self, data_size, length, target_dim=1):
        if isinstance(data_size, int):
            data_size = (data_size,)
        data = torch.randn(length, *data_size)
        target = torch.randn(length, target_dim)
        super().__init__(data, target)

