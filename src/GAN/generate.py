# -*- coding: utf-8 -*-
#Used to generate images given a specific model .pth file.

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


# Number of channels in the output image. Typically 3
outChan = 3

# 
inVector = 100

# Size of feature maps in generator
genFeatures = 128

# Number of GPUs available. If none, use 0 to run solely on CPU.
ngpu = 1

#generator network. Needs to be here to load .pth state dict.
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( inVector, genFeatures * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(genFeatures * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d( genFeatures *16, genFeatures * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(genFeatures * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(genFeatures * 8, genFeatures * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(genFeatures * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( genFeatures * 4, genFeatures * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(genFeatures * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( genFeatures * 2, genFeatures, 4, 2, 1, bias=False),
            nn.BatchNorm2d(genFeatures),
            nn.ReLU(True),
            nn.ConvTranspose2d( genFeatures, outChan, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
generator = Generator(ngpu)
#load the pth file from the given directory.
generator.load_state_dict(torch.load('generator/epoch199.pth'))
generator = generator.to("cuda")
noise = torch.randn(64,inVector,1,1,device = 'cuda')
with torch.no_grad():
    fake = generator(noise).detach().cpu()
counter = 0
for i in fake:
    if(counter%1==0):
        
        plt.axis('off')
        #directory to save the generated images to.
        plt.imsave('celeb/epoch20/celeb' + str(counter+1)+'.png',np.transpose((i.numpy()+1)/2,(1,2,0)))
    
    counter+=1
