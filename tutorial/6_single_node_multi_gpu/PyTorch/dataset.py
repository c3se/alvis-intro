import torch
from torch.utils.data import Dataset


class RandomCorpus(Dataset):

    def __init__(self, n_sentences, context_length, vocab_size):
        self.len = n_sentences
        self.corpus = torch.randint(size=(n_sentences, context_length), high=vocab_size)

    def __getitem__(self, index):
        return self.corpus[index]

    def __len__(self):
        return self.len


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
        self.target = self.data.max(1, keepdim=True)[0].sin()

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.len
