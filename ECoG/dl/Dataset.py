from torch.utils.data import Dataset
from DL_utils import import_ECoG_Tensor, split_data

import torch


class ECoG_Dataset(Dataset):
    def __init__(
        self, datadir, filename, train=True, finger=0, window_size=0.5, sample_rate=1000, overlap=0.0, transform=None,
    ):
        """
        Args:

        """
        # TODO improve training import
        if train:
            self.data, _, self.target, _ = split_data(
                *import_ECoG_Tensor(datadir, filename, finger, window_size, sample_rate, overlap)
            )
        else:
            _, self.data, _, self.target = split_data(
                *import_ECoG_Tensor(datadir, filename, finger, window_size, sample_rate, overlap)
            )

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        sample_data = self.data[idx, :, :]
        sample_target = self.target[idx]

        if self.transform:
            sample_data, sample_target = self.transform(sample_data, sample_target)

        return sample_data, sample_target
