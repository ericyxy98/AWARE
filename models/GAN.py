import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn

import torchvision
import torchvision.transforms as transforms

import pandas as pd
from sklearn.model_selection import ShuffleSplit#pip install scikit-learn

import matplotlib.pyplot as plt
import numpy as np
import os
from time import sleep

from tensorboardX import SummaryWriter
from IPython.display import clear_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

BATCH_SIZE = 32
RANDOM_SEED = 123

# Build dataset from csv files
class AwareDataset(torch.utils.data.Dataset):
    def __init__(self, csv_data, csv_outcome, csv_verbose, root_dir, train=True, target_classes=None, transform=None):
        self.data_raw = pd.read_csv(csv_data, header=None)
        self.data_out = pd.read_csv(csv_outcome)
        self.data_verb = pd.read_csv(csv_verbose)
        self.root_dir = root_dir
        self.transform = transform
       
        if target_classes != None:
            idx = False
            for i in target_classes:
                idx |= (self.data_out.Diagnosis==i)
                for col in self.data_out.columns[0:6]:
                    idx &= (~self.data_out[col].isna())
            self.data_raw = self.data_raw[idx]
            self.data_out = self.data_out[idx]
            self.data_verb = self.data_verb[idx]
       
        if len(self.data_raw) != len(self.data_out) or len(self.data_raw) != len(self.data_verb):
            raise Exception("Inconsistent data length")
           
        idx = list(range(0,len(self.data_raw)))
        rs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
        train_idx, test_idx = next(rs.split(idx))
        if train:
            self.data_raw = self.data_raw.iloc[train_idx]
            self.data_out = self.data_out.iloc[train_idx]
            self.data_verb = self.data_verb.iloc[train_idx]
        else:
            self.data_raw = self.data_raw.iloc[test_idx]
            self.data_out = self.data_out.iloc[test_idx]
            self.data_verb = self.data_verb.iloc[test_idx]

    def __len__(self):
        return len(self.data_raw)

   
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        data = self.data_raw.iloc[idx, 34:118].values.astype('float32')
        data = (data-np.mean(data))/np.std(data)
        target = self.data_out.iloc[idx, :].values.astype('float32')
        verbose = self.data_verb.iloc[idx, :].values.astype('float32')

        return data, target, verbose


trainset = AwareDataset(csv_data = 'data/exhale_data_v8_ave.csv',
                        csv_outcome = 'data/exhale_outcome_v8_ave.csv',
                        csv_verbose = 'data/exhale_verbose_v8_ave.csv',
                        root_dir = 'data/',
                        train = True,
                        target_classes = [0,1])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

testset = AwareDataset(csv_data = 'data/exhale_data_v8_ave.csv',
                       csv_outcome = 'data/exhale_outcome_v8_ave.csv',
                       csv_verbose = 'data/exhale_verbose_v8_ave.csv',
                       root_dir = 'data/',
                       train = False,
                       target_classes = [0,1])
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=0)

#load pretrained decoder
from AE_model import Encoder,Decoder
decoder=Decoder().to(device)
decoder.load_state_dict(torch.load('./pre_trained_model/ckpt_0.022.pth')['decoder'])

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense_g=nn.Sequential(
            nn.Linear(10, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 21))
    def forward(self,noise):
        latent=self.dense_g(noise)
        return torch.unsqueeze(latent,1)
G=Generator().to(device)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
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
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.ReLU(),
            #nn.Sigmoid()
            )
        
    
    def forward(self, csa,diagnosis,spirometry_label,demographic):
        csa=torch.unsqueeze(csa,1)
        reduced_csa=self.csa_reduce(csa)
        
        embedded_diagnosis=self.embedding(diagnosis)
        embedded_diagnosis=torch.unsqueeze(embedded_diagnosis,1)
        
        expaned_spirometry=self.spirometry_dense(spirometry_label)
        expaned_spirometry=torch.unsqueeze(expaned_spirometry,1)
        
        expaned_demographic=self.demographic_dense(demographic)
        expaned_demographic=torch.unsqueeze(expaned_demographic,1)
        
        classifier_input=torch.cat([reduced_csa,embedded_diagnosis,expaned_spirometry,expaned_demographic ], axis=1)
        
        latent=self.classifier(classifier_input)
        latent=torch.clamp(latent,0,1)
        return latent
    
D=Discriminator().to(device)

g_optimizer = torch.optim.Adam(G.parameters(), lr=0.06)
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
criterion1 = nn.BCELoss().to(device)
epochs=500
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=epochs)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=epochs)
l_g_loss=[]
l_d_loss=[]

from torch.autograd import Variable

for epoch in range(epochs):
    print(epoch)
    epoch_g_loss=0
    epoch_d_loss=0
    for batch_idx, (inputs, labels, info) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size=labels[:,0:1].shape[0]
        G.train()
        D.train()
        d_optimizer.zero_grad()
        
        spirometry=((labels[:,4:6]-100)/200).to(device)
        diagnosis=labels[:,0:1].to(device)
        demographic=(info[:,2:3]/80).to(device)
        

        real_validity = D(inputs,diagnosis,spirometry,demographic)
        real_loss = criterion1(real_validity, Variable(torch.ones(batch_size,1)).to(device))

        z = Variable(torch.rand(labels[:,0:1].shape[0], 10)).to(device)
        latent=G(z)
        g_csa,g_diagnosis,g_spirometry_label,g_demographic=decoder(latent)
        fake_validity = D(g_csa,g_diagnosis,g_spirometry_label,g_demographic)
        fake_loss = criterion1(fake_validity, Variable(torch.zeros(batch_size,1)).to(device))
        
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        d_optimizer.step()
        
        
        
        g_optimizer.zero_grad()
        z = Variable(torch.rand(labels[:,0:1].shape[0], 10)).to(device)
        latent=G(z)
        g_csa,g_diagnosis,g_spirometry_label,g_demographic=decoder(latent)
        validity = D(g_csa,g_diagnosis,g_spirometry_label,g_demographic)
        g_loss = criterion1(validity, Variable(torch.ones(batch_size,1)).to(device))
        g_loss.backward()
        g_optimizer.step()
        
        
        
        epoch_g_loss+=g_loss
        epoch_d_loss+=d_loss
    l_g_loss.append(epoch_g_loss.cpu().detach().numpy() /batch_idx)
    l_d_loss.append(epoch_d_loss.cpu().detach().numpy() /batch_idx)
    print(epoch_g_loss/batch_idx,epoch_d_loss/batch_idx)
    scheduler1.step()
    scheduler2.step()
    state = {
        'G': G.state_dict(),
        'D': D.state_dict(),

    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './pre_trained_model/gan_1.pth')

plt.plot(l_g_loss)
plt.plot(l_d_loss)



















