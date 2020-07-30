#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...

@author: Matteo Anelli
"""
from DL.DL_utils import *
from DL.Dataset import ECoG_Dataset
from torch.utils.data import DataLoader
#%%
if __name__ == '__main__':
    # data_dir  = os.environ['DATA_PATH']
    figure_path = 'ECoG\Figures'
    model_path = 'ECoG\Models'
    data_dir = 'C:\\Users\\anellim1\Develop\Econ\BCICIV_4_mat\\'
    file_name = 'sub1_comp.mat'
    sampling_rate = 1000

    dataset = ECoG_Dataset(data_dir, file_name, 0, 0.5, 1000)

    trainloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=1)

    for idx, samples in enumerate(trainloader):
        print(' batch number {}, X shape {}, y shape {}'.format(idx, samples[0].shape, samples[1].shape))



