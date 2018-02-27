import numpy as np
import torch
from DataLoader import *

from torch.autograd import Variable
import torch.nn as nn
from model.AutoEncoder import Net
import sys
import json
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
    'file_lst': [1],   # MODIFY PATH ACCORDINGLY
    'randomize': True,
}

loader_test = DataLoaderDisk(**opt_data_test)

net = Net()

net.load_state_dict(torch.load(start_from, map_location={'cuda:0': 'cpu'}))


def get_label(loader, size, net):
    top_1_correct = 0
    top_5_correct = 0

    out = []
    for i in range(size):
        inputs, labels = loader.next_batch(1)

        inputs = np.reshape(inputs,(inputs.shape[0],inputs.shape[1],inputs.shape[2],1))
        inputs = np.swapaxes(inputs,1,3)
        inputs = np.swapaxes(inputs,2,3)
        inputs = torch.from_numpy(inputs).float()

        net.eval()
        y = net.getLabel(Variable(inputs))

        _, predicted = torch.max(y.data, 1)

        out.append(predicted.cpu().numpy()[0])
    
    out = np.array(out)
    return out

def get_encode(loader, size, net):
    top_1_correct = 0
    top_5_correct = 0

    out = []
    for i in range(size):
        inputs, labels = loader.next_batch(1)

        inputs = np.reshape(inputs,(inputs.shape[0],inputs.shape[1],inputs.shape[2],1))
        inputs = np.swapaxes(inputs,1,3)
        inputs = np.swapaxes(inputs,2,3)
        inputs = torch.from_numpy(inputs).float()
        print(inputs)

        net.eval()
        z = net.encode(Variable(inputs)).data

        out.append(z.cpu().numpy()[0])
    
    out = np.array(out)
    return out


# def single_Result(input,net):
#     input = np.array(input)
#     input = input.astype(float)
#     input = np.reshape(input,(1,input.shape[0],input.shape[1],1))
#     input = np.swapaxes(input,1,3)
#     input = np.swapaxes(input,2,3)
#     input = torch.from_numpy(input).float()

#     net.eval()
#     outputs = net(Variable(input))

#     # _, predicted = torch.max(outputs.data, 1)
#     _, predicted = torch.topk(outputs.data, 5)

#     return predicted.cpu().numpy()[0].tolist()

# myinput = json.loads(sys.argv[1])

# print(single_Result(myinput,net))

enco = get_encode(loader_test,1,net)

deco = net.decode(Variable(torch.from_numpy(enco))).data

print(deco)