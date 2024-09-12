import torch
import torch.nn as nn
import torch.nn.functional as F

class General(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(num_features + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4))
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 4 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2))

    def forward(self, x1, x2):
        x = torch.cat((x1,x2), dim=1)
        y1 = self.regressor(x)
        x = torch.cat((x,y1), dim=1)
        y2 = self.classifier(x)
        return y1, y2
    
class General_withIOS(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(num_features + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4 + 6))
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 4 + 4 + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1))

    def forward(self, x1, x2):
        x = torch.cat((x1,x2), dim=1)
        y1 = self.regressor(x)
        x = torch.cat((x,y1), dim=1)
        y2 = self.classifier(x)
        return y1, y2