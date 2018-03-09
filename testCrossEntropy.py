
import numpy as np
import torch
from DataLoader import *

from torch.autograd import Variable
import torch.nn as nn
from model.AutoEncoderResidualSeg import Net
import torch.optim as optim
import torch.optim.lr_scheduler as s
import torch.nn.functional as F
import sys


loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5)
input = Variable(input)
target = torch.LongTensor(3).random_(5)
target = Variable(target)
output = loss(input, target)

print(input)

print(target)