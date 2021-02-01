"""
    Custom MEG datasets.

    TODO: add and test dataset without bandpower.
"""

from torch.utils.data import Dataset

from MEG.Utils.utils import import_MEG_Tensor, import_MEG_Tensor_form_file, import_MEG_Tensor_2


class MEG_Dataset(Dataset):
    def __init__(self, raw_fnames, duration=1., overlap=0.0, y_measure="movement", transform=None, normalize_input=True,
                 data_dir=None):
        """

        Args:
        raw_fnames [list]:
            List of path of files to import. (The main file format used is fif, however, it accept all the files
            accepted by mne.io.raw().
        duration (float):
            Length of the windows.
        overlap (float):
            Length of the overlap.
        y_measure (string):
            Measure used to reshape the y. Values in [mean, movement, velocity, position]
        transform (function):
            None, no transformation. Otherwise, the transformation is applied.
        normalize_input (bool):
            True, if apply channel-wise standard scaling to the input data.
            False, does not apply ay normalization.
        data_dir (str): default=None
            The path to import the pre-epoched data. If none, the epochs are generated from the raw data.
        """
        # TODO improve training import SubsetRandomSampler
        # if train:
        #     self.data, _, self.target, _ = split_data(
        #         *import_MEG_Tensor(raw_fnames, duration, overlap), test_size=test_size
        #     )
        # else:
        #     _, self.data, _, self.target = split_data(*import_MEG_Tensor(raw_fnames, duration, overlap), test_size=test_size)


        self.raw_fnames = raw_fnames
        self.duration = duration
        self.overlap = overlap
        self.normalize_input = normalize_input
        self.data_dir = data_dir

        if duration == 1. and overlap == 0.8 and data_dir is not None:
            # Import already epoched MEG data from file
            self.data, self.target = import_MEG_Tensor_form_file(data_dir, normalize_input=self.normalize_input,
                                                                 y_measure=y_measure)
        else:
            # Generate dataset from raw MEG data
            self.data, self.target, self.bp = import_MEG_Tensor(self.raw_fnames, self.duration, self.overlap,
                                                       normalize_input=self.normalize_input, y_measure=y_measure)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        sample_data = self.data[idx, :, :]
        sample_target = self.target[idx, :]
        sample_bp = self.bp[idx, :, :]

        if self.transform:
            sample_data, sample_target, sample_bp = self.transform(sample_data, sample_target, sample_bp)

        return sample_data, sample_target, sample_bp

class MEG_Dataset_no_bp(Dataset):
    def __init__(self, raw_fnames, duration=1., overlap=0.0, y_measure="movement", transform=None, normalize_input=True,
                 data_dir=None):
        """

        Args:
        raw_fnames [list]:
            List of path of files to import. (The main file format used is fif, however, it accept all the files
            accepted by mne.io.raw().
        duration (float):
            Length of the windows.
        overlap (float):
            Length of the overlap.
        y_measure (string):
            Measure used to reshape the y. Values in [mean, movement, velocity, position]
        transform (function):
            None, no transformation. Otherwise, the transformation is applied.
        normalize_input (bool):
            True, if apply channel-wise standard scaling to the input data.
            False, does not apply ay normalization.
        data_dir (str): default=None
            The path to import the pre-epoched data. If none, the epochs are generated from the raw data.
        """
        # TODO improve training import SubsetRandomSampler
        # if train:
        #     self.data, _, self.target, _ = split_data(
        #         *import_MEG_Tensor(raw_fnames, duration, overlap), test_size=test_size
        #     )
        # else:
        #     _, self.data, _, self.target = split_data(*import_MEG_Tensor(raw_fnames, duration, overlap), test_size=test_size)


        self.raw_fnames = raw_fnames
        self.duration = duration
        self.overlap = overlap
        self.normalize_input = normalize_input
        self.data_dir = data_dir

        if duration == 1. and overlap == 0.8 and data_dir is not None:
            # Import already epoched MEG data from file
            self.data, self.target = import_MEG_Tensor_form_file(data_dir, normalize_input=self.normalize_input,
                                                                 y_measure=y_measure)
        else:
            # Generate dataset from raw MEG data
            self.data, self.target = import_MEG_Tensor(self.raw_fnames, self.duration, self.overlap,
                                                       normalize_input=self.normalize_input, y_measure=y_measure,
                                                       rps=False)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        sample_data = self.data[idx, :, :]
        sample_target = self.target[idx, :]

        if self.transform:
            sample_data, sample_target = self.transform(sample_data, sample_target)

        return sample_data, sample_target


class MEG_Dataset2(Dataset):
    def __init__(self, raw_fnames, duration=1., overlap=0.0, y_measure="movement", transform=None, normalize_input=True,
                 data_dir=None):
        """

        Args:
        raw_fnames [list]:
            List of path of files to import. (The main file format used is fif, however, it accept all the files
            accepted by mne.io.raw().
        duration (float):
            Length of the windows.
        overlap (float):
            Length of the overlap.
        y_measure (string):
            Measure used to reshape the y. Values in [mean, movement, velocity, position]
        transform (function):
            None, no transformation. Otherwise, the transformation is applied.
        normalize_input (bool):
            True, if apply channel-wise standard scaling to the input data.
            False, does not apply ay normalization.
        data_dir (str): default=None
            The path to import the pre-epoched data. If none, the epochs are generated from the raw data.
        """
        # TODO improve training import SubsetRandomSampler
        # if train:
        #     self.data, _, self.target, _ = split_data(
        #         *import_MEG_Tensor(raw_fnames, duration, overlap), test_size=test_size
        #     )
        # else:
        #     _, self.data, _, self.target = split_data(*import_MEG_Tensor(raw_fnames, duration, overlap), test_size=test_size)


        self.raw_fnames = raw_fnames
        self.duration = duration
        self.overlap = overlap
        self.normalize_input = normalize_input
        self.data_dir = data_dir

        if duration == 1. and overlap == 0.8 and data_dir is not None:
            # Import already epoched MEG data from file
            self.data, self.target = import_MEG_Tensor_form_file(data_dir, normalize_input=self.normalize_input,
                                                                 y_measure=y_measure)
        else:
            # Generate dataset from raw MEG data
            self.data, self.target, self.bp = import_MEG_Tensor_2(self.raw_fnames, self.duration, self.overlap,
                                                       normalize_input=self.normalize_input, y_measure=y_measure)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        sample_data = self.data[idx, :, :]
        sample_target = self.target[idx, :]
        sample_bp = self.bp[idx, :, :]

        if self.transform:
            sample_data, sample_target, sample_bp = self.transform(sample_data, sample_target, sample_bp)

        return sample_data, sample_target, sample_bp


class MEG_Cross_Dataset(Dataset):
    def __init__(self, data_dir, file_name, sub, mlp=False):
        """

        Args:
        data_dir (string):
            Path of the data directory.
        file_name (string):
            Data file name. file.hdf5.
        sub (int):
            Number of the test subject.
        mlp (bool):
            True if mlp_rps else otherwise.
        """
        self.data_dir = data_dir
        self.file_name = file_name
        self.sub = sub

        if mlp:
            # Import already epoched MEG data from file
            self.data, self.target = import_MEG_Tensor_form_file(data_dir, normalize_input=self.normalize_input,
                                                                 y_measure=y_measure)
        else:
            # Generate dataset from raw MEG data
            self.data, self.target, self.bp = import_MEG_Tensor_2(self.raw_fnames, self.duration, self.overlap,
                                                                  normalize_input=self.normalize_input,
                                                                  y_measure=y_measure)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        sample_data = self.data[idx, :, :]
        sample_target = self.target[idx, :]
        sample_bp = self.bp[idx, :, :]

        if self.transform:
            sample_data, sample_target, sample_bp = self.transform(sample_data, sample_target, sample_bp)

        return sample_data, sample_target, sample_bp
