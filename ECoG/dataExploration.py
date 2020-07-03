import os
import mne

import scipy.io as sio

data_dir  = 'C:\\Users\\anellim1\Develop\Econ\BCICIV_4_mat\\'
file_name = 'sub1_comp.mat'

print(os.path.exists(data_dir))
print(os.path.exists(os.path.join(data_dir, file_name)))

dataset = sio.loadmat(os.path.join(data_dir, file_name))
print(type(dataset))
print(dataset.keys())

X = dataset['train_data']
y = dataset['train_dg']

print(X.shape) # 400000 time samples x 62 channel
print(y.shape) # 400000 time samples x 5 fingers

print('Example of fingers position : {}'.format(y[0]))








