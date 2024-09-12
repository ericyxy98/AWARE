import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as T
from models.MTL import General, General_withIOS

class CNN1D(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
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
        self.output = nn.Sequential(
            nn.Linear(32 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels))

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x1 = self.encoder(x1)
        y = self.output(torch.cat((x1, x2), dim=-1))
        return y