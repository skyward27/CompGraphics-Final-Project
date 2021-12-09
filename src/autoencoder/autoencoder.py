import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import os
import cv2
import helper
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


ims_path = ''

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(ims_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

images, labels = next(iter(dataloader))

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 9),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 9),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, 7),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 9, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 9, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 8, stride=2)
        )
    
    def forward(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded

model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay = 1e-5)
def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

device = get_device()
model.to(device)
print(device)
criterion = nn.MSELoss().to(device)
num_epochs = 1500
outputs = []
losses = []
for epoch in range(num_epochs):
    for (img, index) in dataloader:
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss on epoch {epoch + 1}: {loss.item()}")
    losses.append(loss.item())

plt.plot(range(num_epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

test = model(images.to(device))
test_reg = test.to('cpu')

plt.imshow(test_reg.detach()[0].permute(1,2,0))
plt.imshow(images.detach()[0].permute(1,2,0))

test_enc = model.encoder(images.to(device))
print("Size of encoded images: " + test_enc[0].size())
torch.save(model.state_dict(), '')
