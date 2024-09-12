'''
By Xiangyu

1D convolutional neural network for end-to-end classification
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as T
from models.MTL import General, General_withIOS

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.pretrain = True
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(3),
            nn.Flatten(),
            nn.Linear(16*7, 32))
        self.decoder = nn.Sequential(
            nn.Linear(32, 16*7),
            nn.Unflatten(1, torch.Size([16, 7])),
            nn.Upsample(scale_factor=3),
            nn.ConvTranspose1d(16, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(16, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(8, 1, 3, padding=1))
        self.output = General(32)
        
    def pre_train(self):
        self.pretrain = True
#         for param in self.encoder.parameters():
#             param.requires_grad = True
        
    def fine_tune(self):
        self.pretrain = False
#         for param in self.encoder.parameters():
#             param.requires_grad = False

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        z = self.encoder(x1)
        if self.pretrain:
            y = self.decoder(z)
            y = y.squeeze(1)
            return y, z
        else:
            y = self.output(z, x2)
            return y