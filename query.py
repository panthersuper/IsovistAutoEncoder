#!/home/design_heritage/platform/dh_upload_ACL/lib/miniconda3/bin/python3.6

import numpy as np
import torch
from DataLoader import *

from torch.autograd import Variable
import torch.nn as nn
from model.AutoEncoder import Net
import apiFunctions as API
import sys
import json

# Dataset Parameters

# Training Parameters
learning_rate = 0.01
training_epoches = 10
step_display = 10
step_save = 2
path_save = 'test'
start_from = 'test3/Epoch20'#'./alexnet64/Epoch28'
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

# data = loader.next_batch(1)[0][0]
# data2 = loader.next_batch(1)[0][0]

# result = API.label(data,net,3)
# z = API.encode(data,net)
# z2 = API.encode(data2,net)
# z3 = z + (z - z2)
# xx = API.decode(z3,net)

# print(result)
# result = API.label(xx,net,3)

# print(result)

inputs = json.loads(sys.argv[1])

a_ = inputs["a"]
b_ = inputs["b"]
c_ = inputs["c"]

a = np.array(a_)
b = np.array(b_)
c = np.array(c_)

a_encode = API.encode(a,net) if len(a_) else np.array([])
b_encode = API.encode(b,net) if len(b_) else np.array([])
c_encode = API.encode(c,net) if len(c_) else np.array([])

a_minus_b_encode = a_encode - b_encode if len(a_) and len(b_) else np.array([])
a_minus_c_encode = a_encode - c_encode if len(a_) and len(c_) else np.array([])
b_minus_c_encode = b_encode - c_encode if len(c_) and len(b_) else np.array([])

b_minus_c_plus_a_encode = b_encode - c_encode + a_encode if len(a_) and len(b_) and len(c_) else np.array([])
b_minus_c_plus_a = API.decode(b_minus_c_plus_a_encode,net) if len(a_) and len(b_) and len(c_) else np.array([])
b_minus_c_plus_a_label = API.labelByZ(b_minus_c_plus_a_encode,net,5) if len(a_) and len(b_) and len(c_) else np.array([])

# the top 5 prediction labels
a_label = API.label(a,net,5) if len(a_) else np.array([])
b_label = API.label(b,net,5) if len(b_) else np.array([])
c_label = API.label(c,net,5) if len(c_) else np.array([])

output = {}

output["a"] = {
    "value":a.tolist() ,
    "encode":a_encode.tolist() ,
    "label":a_label.tolist() 
}

output["b"] = {
    "value":b.tolist() ,
    "encode":b_encode.tolist() ,
    "label":b_label.tolist() 
}

output["c"] = {
    "value":c.tolist() ,
    "encode":c_encode.tolist() ,
    "label":c_label.tolist() 
}

output["a+b-c"] = {
    "value":b_minus_c_plus_a.tolist() ,
    "encode":b_minus_c_plus_a_encode.tolist() ,
    "label":b_minus_c_plus_a_label.tolist() 
}

# print(API.encode(a,net)-API.encode(b,net))
print(json.dumps(output))

