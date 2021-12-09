# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:20:49 2021

@author: haves
"""

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "pics/archive/socal2"

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 32

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 9), #N, 32, 120, 120
            nn.ReLU(),
            nn.MaxPool2d(2), #N, 32, 60, 60
            nn.Conv2d(32, 64, 9), #N, 64, 52, 52
            nn.ReLU(),
            nn.MaxPool2d(2), #N, 64, 26, 26
            nn.Conv2d(64, 256, 7), #N, 256, 24, 24
            nn.ReLU(),
            nn.MaxPool2d(2), #N, 256, 12, 12
            #nn.Conv2d(256, 512, 5), #N, 256, 8, 8
            #nn.ReLU(),
            #nn.MaxPool2d(2), #N, 512, 4, 4
        )


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 9, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 9, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 8, stride=2),
            #nn.ReLU(),
            #nn.ConvTranspose2d(32, 3, 8, stride=2),
        )
    
    def forward(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded
model = Autoencoder().to('cuda')
model.load_state_dict(torch.load("auto10x10_model_dict.pt"))
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 16, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf*16, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, 256, 3,1,0,bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
generator = Generator(ngpu)
generator.load_state_dict(torch.load('generator/epoch159.pth'))
generator = generator.to("cuda")
noise = torch.randn(64,100,1,1,device = 'cuda')
with torch.no_grad():
    fake = model.decoder(generator(noise)).detach().cpu()
    
plt.imshow(fake[63].permute(1,2,0))
fakegrid = vutils.make_grid(fake,padding=2,normalize=True)
plt.imshow(np.transpose(fakegrid,(1,2,0)))
counter = 0
for i in fake:
    if(counter%1==0):
        plt.figure()
        plt.axis('off')
        plt.imshow(np.transpose((i+1)/2,(1,2,0)),vmin=0,vmax=255)
    
    counter+=1
