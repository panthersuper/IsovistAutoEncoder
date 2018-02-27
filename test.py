#!/home/design_heritage/platform/dh_upload_ACL/lib/miniconda3/bin/python3.6

import numpy as np
import torch
from DataLoader import *

from torch.autograd import Variable
import torch.nn as nn
from model.AutoEncoder import Net
import apiFunctions as API

# Dataset Parameters

# Training Parameters
learning_rate = 0.01
training_epoches = 10
step_display = 10
step_save = 2
path_save = 'test'
start_from = 'test/Epoch10'#'./alexnet64/Epoch28'
starting_num = 1

batch_size = 64

opt_data_test = {
    'img_root': 'data/',   # MODIFY PATH ACCORDINGLY
    'file_lst': [0],   # MODIFY PATH ACCORDINGLY
    'randomize': True,
}

loader = DataLoaderDisk(**opt_data_test)

net = Net()

net.load_state_dict(torch.load(start_from, map_location={'cuda:0': 'cpu'}))

data = loader.next_batch(1)[0][0]
data2 = loader.next_batch(1)[0][0]

result = API.label(data,net,3)
z = API.encode(data,net)
z2 = API.encode(data2,net)
z3 = z + (z - z2)
xx = API.decode(z3,net)

print(result)
result = API.label(xx,net,3)

print(result)
