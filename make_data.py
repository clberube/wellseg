# @Author: charles
# @Date:   2020-04-21T09:41:06-04:00
# @Last modified by:   charles
# @Last modified time: 2020-04-21T15:40:56-04:00


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils import data
from scipy.misc import electrocardiogram
from geoml.preprocessing import make_nd_windows
from smooth import smooth

fpath = ('/Users'
         '/charles'
         '/Goldspot Discoveries Corp'
         '/Group Ten Metals - 06_Machine_Learning'
         '/04_Petrophysics'
         '/data'
         '/NN_aiSIRIS_ASSAY_MPP_SCIP_SGinf_LITH_ROCKGRP.csv')

df = pd.read_csv(fpath, low_memory=False)

values = np.log10(df.loc[df['holeid'].isin(df['holeid'].mode()), 'Scpt:0.001_SI']).values
plt.plot(values)
np.random.seed(1)

L = 1e5
Nw = 784
Ns = Nw/2

time = np.arange(L)
values = np.random.random(time.shape)

spike = int(np.random.choice(time))

values[spike:spike+10000] += 5

values = smooth(values, int(Ns))

# values = electrocardiogram()[10000:20000]

# plt.plot(data)

X = make_nd_windows(values, Nw, steps=Ns)

tensor_x = torch.Tensor(X)  # transform to torch tensor
syn_dataset = data.TensorDataset(tensor_x) # create your datset
syn_dataloader = data.DataLoader(syn_dataset, batch_size=1) # create your dataloader
