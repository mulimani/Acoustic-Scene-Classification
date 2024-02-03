import torch
from torch.utils.data import Dataset


class BatchData(Dataset):
    def __init__(self, mels, labels):
        self.mels = mels
        self.labels = labels

    def __getitem__(self, index):
        mels = self.mels[index]
        label = self.labels[index]
        return mels, label

    def __len__(self):
        return len(self.mels)