from torch.utils.data import Dataset, DataLoader
from DL.DL_utils import import_ECoG_Tensor



class ECoG_Dataset(Dataset):
    def __init__(self, datadir, filename, finger=0, window_size=0.5, sample_rate=1000, transform=None):
        """
        Args:

        """
        self.data, self.target = import_ECoG_Tensor(datadir, filename, finger, window_size, sample_rate)
        self.transform = transform

    def __len__(self):
        return self.data.shape

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform:
            sample = self.transform(sample)

        return sample