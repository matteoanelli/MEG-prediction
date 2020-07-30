from torch.utils.data import Dataset
from DL.DL_utils import import_ECoG_Tensor

import torch


class ECoG_Dataset(Dataset):
    def __init__(self, datadir, filename, finger=0, window_size=0.5, sample_rate=1000, transform=None):
        """
        Args:

        """
        self.data, self.target = import_ECoG_Tensor(datadir, filename, finger, window_size, sample_rate)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        sample_data = self.data[idx, :, :]
        sample_target = self.target[idx]

        if self.transform:
            sample_data, sample_target = self.transform(sample_data, sample_target)

        return sample_data, sample_target