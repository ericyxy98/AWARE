import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.csa_reduce=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=True),
            
            
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm1d(6),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.embedding =nn.Sequential(
            #nn.Embedding(2, 21),
            nn.Linear(1, 21),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.spirometry_dense=nn.Sequential(
            nn.Linear(2, 21), # 7 is ini-num-channel
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.demographic_dense=nn.Sequential(
            nn.Linear(1, 21), # 7 is ini-num-channel
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            )
        
    
    def forward(self, csa,diagnosis,spirometry_label,demographic):
        csa=torch.unsqueeze(csa,1)
        reduced_csa=self.csa_reduce(csa)
        
        embedded_diagnosis=self.embedding(diagnosis)
        #bath_size=embedded_diagnosis.shape[0]
        embedded_diagnosis=torch.unsqueeze(embedded_diagnosis,1)
        
        expaned_spirometry=self.spirometry_dense(spirometry_label)
        expaned_spirometry=torch.unsqueeze(expaned_spirometry,1)
        
        expaned_demographic=self.demographic_dense(demographic)
        expaned_demographic=torch.unsqueeze(expaned_demographic,1)
        
        classifier_input=torch.cat([reduced_csa,embedded_diagnosis,expaned_spirometry,expaned_demographic ], axis=1)
        #classifier_input=torch.cat([reduced_csa ], axis=1)
        
        latent=self.classifier(classifier_input)

        return latent
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.expand_latent=nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose1d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.csa_regressor=nn.Sequential(
            nn.Flatten(),
            nn.Linear(84, 84), 
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(16, 1), 
            nn.Sigmoid()
            )
        self.diagnosis_regressor=nn.Sequential(
            nn.Flatten(),
            nn.Linear(84, 16), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1), 
            nn.Sigmoid()
            )
        self.spirometry_regressor=nn.Sequential(
            nn.Flatten(),
            nn.Linear(84, 16), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 2), 
            nn.Sigmoid()
            )
        self.demographic_regressor=nn.Sequential(
            nn.Flatten(),
            nn.Linear(84, 16), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1), 
            nn.Sigmoid()
            )
    def forward(self, latent):
    
        expanded_latent=self.expand_latent(latent)
        
        csa=expanded_latent[:,0,:]
        diagnosis =self.diagnosis_regressor(expanded_latent[:,1,:])
        spirometry_label =self.spirometry_regressor(expanded_latent[:,2,:])
        demographic =self.demographic_regressor(expanded_latent[:,3,:])
        return torch.squeeze(csa,1),diagnosis,spirometry_label,demographic
