from torch.utils.data import Dataset

from MEG.Utils.utils import import_MEG_Tensor


class MEG_Dataset(Dataset):
    def __init__(self, raw_fnames, duration=1., overlap=0.0, transform=None):
        """
        Args:

        """
        # TODO improve training import SubsetRandomSampler
        # if train:
        #     self.data, _, self.target, _ = split_data(
        #         *import_MEG_Tensor(raw_fnames, duration, overlap), test_size=test_size
        #     )
        # else:
        #     _, self.data, _, self.target = split_data(*import_MEG_Tensor(raw_fnames, duration, overlap), test_size=test_size)

        self.data, self.target = import_MEG_Tensor(raw_fnames, duration, overlap)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        sample_data = self.data[idx, :, :]
        sample_target = self.target[idx, :]

        if self.transform:
            sample_data, sample_target = self.transform(sample_data, sample_target)

        return sample_data, sample_target
