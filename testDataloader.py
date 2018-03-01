import numpy as np
import torch
from DataLoader import *

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as s

batch_size =3

# Construct dataloader
opt_data_train = {
    'img_root': 'data/',   # MODIFY PATH ACCORDINGLY
    'file_lst': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],   # MODIFY PATH ACCORDINGLY
    'randomize': True,
}

loader_train = DataLoaderDisk(**opt_data_train)

data = loader_train.next_batch(batch_size)

print(data)

