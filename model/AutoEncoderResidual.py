import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append('model/')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        self.fc1 = nn.Linear(2100, 100)
        self.fc2 = nn.Linear(100, 15)

        self.encoder = Encoder(100)
        self.decoder = Decoder(100)
    
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
            
            nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3), #60,30
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),#30,15
            nn.Conv2d(8, 16, kernel_size=5, padding=2),#30,15
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),#30,15
            nn.LeakyReLU(inplace=True),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.Conv2d(32, 24, kernel_size=3, padding=1),#30,15
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(24, 20, kernel_size=3, padding=1),#30,15
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(2, stride=2),#20/15/7
            nn.Dropout2d(),
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

            nn.ConvTranspose2d(20, 20, 1, stride=2, padding=(0,1),output_padding=(1,2),dilation=3),  # b,  20 x 30 x 15
            nn.BatchNorm2d(20),
            nn.ReLU(True),

            ConvLayer(20, 24, kernel_size=3, stride=1), # x 24 x 30 x 15
            nn.BatchNorm2d(24),
            nn.ReLU(True),

            ConvLayer(24, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32),

            ConvLayer(32, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            ConvLayer(16, 8, kernel_size=5, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            UpsampleConvLayer(8, 1, kernel_size=5, stride=1, upsample=2),
        )

    def to2D(self, z):
        z = self.fc3(z)
        z = z.view(-1,20,15,8) #20/16/8
        return z

    def forward(self,z):
        z = self.to2D(z)

        return self.decoder(z)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


