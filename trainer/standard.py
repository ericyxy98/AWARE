import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import random
import os
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, lr, T_max, device, summarywriter):
        self.model = model
        self.device = device
        self.writer = summarywriter
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        self.train_loss = 1e8
        self.best_loss = 1e8
        
        self.ref = torch.Tensor(pd.read_csv('data/reference.csv', header=None).to_numpy())
        self.ref = self.ref.permute(1,0).to(device)
        
    def train(self, epoch, data_loader):
        self.model.train()
        self.iteration(epoch, data_loader)
        self.scheduler.step()

    def validate(self, epoch, data_loader):
        self.model.eval()
        with torch.no_grad():
            self.iteration(epoch, data_loader, train=False, validate=True)
            
    def test(self, data_loader, no_print=False):
        self.model.load_state_dict(torch.load('./checkpoint/ckpt.pth')['net'])
        self.model.eval()
        with torch.no_grad():
            self.iteration(None, data_loader, train=False, validate=False)
        if not no_print:
            print('Best training loss: %.4f' % (self.best_loss))
            print("On which epoch reach the lowest loss:",
                  torch.load('./checkpoint/ckpt.pth')['epoch'])
        
    def save(self, loss, epoch):
        state = {
            'net': self.model.state_dict(),
            'loss': loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        
    def iteration(self, epoch, data_loader, train=True, validate=False):
        total_size = 0
        avg_loss = 0
        inputs_all = torch.Tensor([])
        labels_all = torch.Tensor([])
        info_all = torch.Tensor([])
        outputs_all = torch.Tensor([])
        for batch_idx, (inputs, masks, labels, info) in enumerate(data_loader):
            # Step 1: put data to gpu
            inputs, labels, info = inputs.to(self.device), labels.to(self.device), info.to(self.device)
            
            # Step 2: forward model
            outputs = self.model(self.ref, info[:,2:6], labels[:,1:7]) 
            
            # Step 3: calculate loss
            loss = self.criterion(outputs, inputs) 
            total_size += labels.size(0)
            avg_loss += loss.item()*labels.size(0)
            if loss.isnan().sum()>0:
                print("Loss NaN at epoch:", epoch)
            
            # Step 4: back propagation
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Step 5: visualize results
            if not train and not validate:
                # Record details
                inputs_all = torch.cat((inputs_all, inputs.cpu()), dim=0)
                labels_all = torch.cat((labels_all, labels.cpu()), dim=0)
                info_all = torch.cat((info_all, info.cpu()), dim=0)
                outputs_all = torch.cat((outputs_all, outputs.cpu()), dim=0)
                for i in range(4):
                    plt.subplot(2,2,i+1)
                    plt.plot(inputs[i,:].cpu())
                    plt.plot(outputs[i,:].cpu())
                    plt.plot(self.ref[0,:].cpu(),'--')
                plt.show()
                print("Loss: ", self.criterion(outputs, inputs))
        
        avg_loss /= total_size
        if train:
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.train_loss = avg_loss
        elif validate:
            self.writer.add_scalar('Loss/validate', avg_loss, epoch)
            # Save checkpoint.
            if avg_loss < self.best_loss:
                self.save(avg_loss, epoch)
                self.best_loss = avg_loss
        else:
            self.inputs = inputs_all
            self.labels = labels_all
            self.info = info_all
            self.outputs = outputs_all

            self.results = {
                'Loss': avg_loss,
            }