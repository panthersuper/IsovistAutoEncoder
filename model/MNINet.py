import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append('model/')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5,stride=1, padding=2) #
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5,stride=1, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2100, 50)
        self.fc2 = nn.Linear(50, 15)

    def forward(self, x):
        #1/60/30

        #10/30/15
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        #20/15/7
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 2100)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x) #batch*15
        return F.log_softmax(x) 

