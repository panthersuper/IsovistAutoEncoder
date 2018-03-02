# train auto-encoder

import numpy as np
import torch
from DataLoader import *

from torch.autograd import Variable
import torch.nn as nn
from model.AutoEncoderResidual import Net
import torch.optim as optim
import torch.optim.lr_scheduler as s
import torch.nn.functional as F

# Dataset Parameters

# Training Parameters
learning_rate = 0.01
training_epoches = 20
step_display = 50
step_save = 2
path_save = 'test3'
start_from = ''#./test/Epoch20'
starting_num = 1

batch_size = 64

# Construct dataloader
opt_data_train = {
    'img_root': 'data/',   # MODIFY PATH ACCORDINGLY
    'file_lst': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],   # MODIFY PATH ACCORDINGLY
    'randomize': True,
}

def get_accuracy(loader, size, net):
    top_1_correct = 0
    top_5_correct = 0
    decode_error = 0

    for i in range(size):
        inputs, labels = loader.next_batch(1)
        labels = np.asarray(labels,dtype=np.float32)

        inputs = np.reshape(inputs,(inputs.shape[0],inputs.shape[1],inputs.shape[2],1))
        inputs = np.swapaxes(inputs,1,3)
        inputs = np.swapaxes(inputs,2,3)
        inputs = torch.from_numpy(inputs).float().cuda()
        labels = torch.from_numpy(labels).long().cuda()

        net.eval()
        outputs = net(Variable(inputs))[1]
        xx = net(Variable(inputs))[0]

        err = np.sum(np.absolute(xx.data.cpu().numpy()[0][0] - inputs.cpu().numpy()[0][0]))
        decode_error += err

        # print("---------------")
        # print(xx.data.cpu().numpy()[0][0])
        # print("+")
        # print(inputs.cpu().numpy()[0][0])
        # print("+")
        # print(xx.data.cpu().numpy()[0][0] - inputs.cpu().numpy()[0][0])


        _, predicted = torch.max(outputs.data, 1)

        top_1_correct += (predicted == labels).sum()
        _, predicted = torch.topk(outputs.data, 5)
        for i in range(5):
            top_5_correct += (predicted[:,i] == labels).sum()

    return 100 * top_1_correct / float(size), 100 * top_5_correct / float(size), decode_error

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        pass #nn.init.kaiming_uniform(m.weight.data)

loader_train = DataLoaderDisk(**opt_data_train)
# loader_val = DataLoaderDisk(**opt_data_val)

net = Net()
net = net.cuda()
if start_from != '':
    net.load_state_dict(torch.load(start_from))
else:
    net.apply(weights_init)

criterion_x = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5) 

scheduler = s.StepLR(optimizer, step_size=4, gamma=0.075)

running_loss = 0.0

if start_from == '':
    with open('./' + path_save + '/log.txt', 'w') as f:
        f.write('')

for epoch in range(training_epoches):
    scheduler.step()
    net.train()

    for i in range(600):  # loop over the dataset multiple times
        data = loader_train.next_batch(batch_size)

        # get the inputs
        inputs, labels = data
        labels = np.asarray(labels,dtype=np.float32)

        inputs = np.reshape(inputs,(inputs.shape[0],inputs.shape[1],inputs.shape[2],1))
        inputs = np.swapaxes(inputs,1,3)
        inputs = np.swapaxes(inputs,2,3)
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).long()

        # wrap them in Variable
        # inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels= Variable(inputs.cuda()), Variable(labels.cuda())


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output_x,output_y = net(inputs) # places output

        loss = 10*criterion_x(output_x, inputs) + F.nll_loss(output_y,labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % step_display == step_display - 1:    # print every 100 mini-batches
            print('TRAINING Epoch: %d %d loss: %.3f' %
                  (epoch + starting_num, i + 1, running_loss/step_display))
            with open('./' + path_save + '/log.txt', 'a') as f:
                f.write('TRAINING Epoch: %d %d loss: %.3f\n' %
                  (epoch + starting_num, i + 1, running_loss/step_display))

            running_loss = 0.0

    if epoch % step_save == 1:
       torch.save(net.state_dict(), './' + path_save + '/Epoch'+str(epoch+starting_num))

    net.eval()
    with open('./' + path_save + '/log.txt', 'a') as f:
        accs = get_accuracy(loader_train, 600, net)
        f.write("Epoch: %d Training set: Top-1 %.3f Top-5 %.3f\n" %(epoch + starting_num, accs[0], accs[1]))
        print("Epoch:", epoch + starting_num, "Training set: Top-1", accs[0], "Top-5", accs[1], "err", accs[2])
    #     accs = get_accuracy(loader_val, 10000, net)
    #     print("Epoch:", epoch + starting_num, "Validation set: Top-1",accs[0], "Top-5", accs[1])
    #     f.write("Epoch: %d Validation set: Top-1 %.3f Top-5 %.3f\n" %(epoch + starting_num, accs[0], accs[1]))
print('Finished Training')
