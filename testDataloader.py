import numpy as np
import torch
from DataLoader import *

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as s
from PIL import Image

batch_size =3

# Construct dataloader
opt_data_train = {
    'img_root': 'data/',   # MODIFY PATH ACCORDINGLY
    'randomize': True,
}

loader_train = DataLoaderSegmentation(**opt_data_train)

data = loader_train.next_batch(batch_size)

print(data[0],np.shape(data[0]))

print(data[1],np.shape(data[1]))

im = Image.fromarray(data[0][0])
im.show()
