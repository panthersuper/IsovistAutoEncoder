import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append('model/')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        self.fc1 = nn.Linear(2100, 50)
        self.fc2 = nn.Linear(50, 15)

        self.encoder = Encoder(50)
        self.decoder = Decoder(50)
    
    def zToLabels(self, x):
        x = self.fc2(x)
        return F.log_softmax(x)

    def getLabel(self,x):
        # get label with image input
        
        z = self.encoder(x)
        y = self.zToLabels(z)
        return y
    
    def encode(self,x):
        # get latent vector from image input
        return self.encoder(x)

    def decode(self,z):
        # get generated image from latent vector

        return self.decoder(z)

    def forward(self, x):

        z = self.encoder(x)
        y = self.zToLabels(z)
        xx = self.decoder(z)

        return xx,y

class Encoder(nn.Module):
    def __init__(self,z_size):
        super(Encoder, self).__init__() 
        self.fc1 = nn.Linear(2100, z_size)

        self.encoder = nn.Sequential(
            #1/60/30
            
            nn.Conv2d(1, 10, kernel_size=5,stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(), #10/30/15

            nn.Conv2d(10, 20, kernel_size=5,stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(), #20/15/7
        )

    def forward(self,x):
        x = self.encoder(x)

        x = x.view(-1, 2100)
        x = self.fc1(x)

        return x





class Decoder(nn.Module):
    def __init__(self,z_size):
        super(Decoder, self).__init__() 
        self.fc3 = nn.Linear(z_size, 20*15*8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 20, 1, stride=2, padding=(0,1),output_padding=(1,2),dilation=3),  # b, 40, 30, 15
            nn.ReLU(True),

            nn.ConvTranspose2d(20, 10, 5, stride=1, padding=2),  # b, 16, 5, 5
            nn.ReLU(True),

            nn.ConvTranspose2d(10, 10, 1, stride=2, padding=(0,0),output_padding=(1,1)),  # b, 40, 60, 30
            nn.ReLU(True),

            # # nn.UpsamplingNearest2d(size=(60,30)),

            nn.ConvTranspose2d(10, 1, 5, stride=1, padding=2),  # b, 1, 28, 28
            nn.Tanh()
        )
    def to2D(self, z):
        z = self.fc3(z)
        z = z.view(-1,20,15,8) #20/16/8
        return z

    def forward(self,z):
        z = self.to2D(z)

        return self.decoder(z)




