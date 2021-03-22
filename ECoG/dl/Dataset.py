from torch.utils.data import Dataset
from ECoG.Utils.utils import import_ECoG_Tensor

import torch


class ECoG_Dataset(Dataset):
    def __init__(
        self,
        datadir,
        filename,
        finger=0,
        duration=1.0,
        overlap=0.0,
        sample_rate=1000,
        transform=None,
        rps=True,
    ):
        """
        Args:

        """
        self.rps = rps
        if self.rps:
            self.data, self.target, self.bp = import_ECoG_Tensor(
                datadir,
                filename,
                finger,
                duration,
                sample_rate,
                overlap,
                rps=self.rps,
            )
        else:
            self.data, self.target = import_ECoG_Tensor(
                datadir,
                filename,
                finger,
                duration,
                sample_rate,
                overlap,
                rps=self.rps,
            )

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        if self.rps:
            sample_data = self.data[idx, :, :]
            sample_target = self.target[idx]
            sample_bp = self.bp[idx, :, :]
        else:
            sample_data = self.data[idx, :, :]
            sample_target = self.target[idx]

        if self.transform:
            sample_data, sample_target = self.transform(
                sample_data, sample_target
            )

        if self.rps:
            return sample_data, sample_target, sample_bp
        else:
            return sample_data, sample_target
