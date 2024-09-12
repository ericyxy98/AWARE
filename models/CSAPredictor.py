import torch
import torch.nn as nn
import torch.nn.functional as F

class CSAPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
            nn.Linear(16*7, 32)
        )
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
            nn.ConvTranspose1d(8, 1, 3, padding=1)
        )
        self.embedding_demogr = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        self.embedding_spiro = nn.Sequential(
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

    def forward(self, csa_ref, demogr, spiro):
        x = csa_ref.expand(demogr.size(0), -1).unsqueeze(1)
        z = self.encoder(x) + self.embedding_demogr(demogr) + self.embedding_spiro(spiro)
        y = self.decoder(z)
        y = y.squeeze(1)
        return y
        
        