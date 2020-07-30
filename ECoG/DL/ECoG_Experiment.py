#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...

@author: Matteo Anelli
"""
from DL_utils import *
#%%
# data_dir  = os.environ['DATA_PATH']
figure_path = 'ECoG\Figures'
model_path = 'ECoG\Models'
data_dir = 'C:\\Users\\anellim1\Develop\Econ\BCICIV_4_mat\\'
file_name = 'sub1_comp.mat'
sampling_rate = 1000

X, y = import_ECoG_Tensor(data_dir, file_name, 0, 0.5, 1000)

