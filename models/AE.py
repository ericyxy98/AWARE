import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as T
from models.MTL import General

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrain = True
        self.encoder = nn.Sequential(
            nn.Linear(84+4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32))
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 84))
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
        x = torch.cat((x1,x2),1)
        z = self.encoder(x)
        if self.pretrain:
            y = self.decoder(z)
        else:
            y = self.output(z, x2)
        return y