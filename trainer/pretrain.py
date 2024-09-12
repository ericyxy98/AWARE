import torch
import torch.nn as nn
import torch.optim as optim

import random
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import seaborn as sns

from utils.loss import SupConLoss

class PreTrainer:
    def __init__(self, model, lr, T_max, device, summarywriter):
        self.model = model
        self.device = device
        self.writer = summarywriter
        
        self.criterion = nn.MSELoss()
        self.criterion_2 = SupConLoss(temperature=0.8)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        self.train_loss = 1e8
        self.best_loss = 1e8
        
    def train(self, epoch, data_loader):
        self.model.train()
        self.iteration(epoch, data_loader)
        self.scheduler.step()

    def validate(self, epoch, data_loader):
        self.model.eval()
        with torch.no_grad():
            self.iteration(epoch, data_loader, train=False, validate=True)
            
    def test(self, data_loader, no_print=False):
        self.model.load_state_dict(torch.load('./checkpoint/ckpt_pretrain.pth')['net'])
        self.model.eval()
        with torch.no_grad():
            self.iteration(None, data_loader, train=False, validate=False)
        if not no_print:
            print('Best pre-training loss: %.4f' % (self.best_loss))
            print("On which epoch reach the the lowest loss:",
                  torch.load('./checkpoint/ckpt_pretrain.pth')['epoch'])
        
    def save(self, loss, epoch):
        state = {
            'net': self.model.state_dict(),
            'loss': loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_pretrain.pth')
        
    def iteration(self, epoch, data_loader, train=True, validate=False):
        total_size = 0
        avg_loss = 0
        X = torch.Tensor([])
        y = torch.Tensor([])
        for batch_idx, (inputs, masks, labels, info) in enumerate(data_loader):
            # Step 1: put data to gpu
            inputs, labels, info = inputs.to(self.device), labels.to(self.device), info.to(self.device)
            masks = masks.to(self.device)
            
            # Step 2: forward model
            outputs, intermediates = self.model(inputs*masks, info[:,2:6]) 
#             outputs, intermediates = self.model(inputs, info[:,2:6]) 
#             outputs = self.model(inputs, info[:,2:6])
            
            # Step 3: calculate loss
#             loss = self.criterion(outputs*(1-masks), inputs*(1-masks))
            loss = self.criterion(outputs, inputs) #+ self.criterion_2(intermediates.unsqueeze(1), labels[:,0].long())
#             loss = self.criterion_2(intermediates.unsqueeze(1), labels[:,0].long())
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
#             if not train and not validate:
#                 for i in range(4):
#                     plt.subplot(2,2,i+1)
#                     plt.plot(inputs[i,:].cpu())
#                     plt.plot(masks[i,:].cpu())
#                     plt.plot(outputs[i,:].cpu())
#                 plt.show()
#                 print("Loss_1: ", self.criterion(outputs, inputs))
#                 print("Loss_2: ", self.criterion_2(intermediates.unsqueeze(1), labels[:,0].long()))
                
#                 X = intermediates.detach().cpu()
#                 y = labels[:,0].long().detach().cpu()
#                 print(pairwise_distances(X[y==0,:]).mean())
#                 print(pairwise_distances(X[y==1,:]).mean())
#                 print(pairwise_distances(X[y==0,:], X[y==1,:]).mean())
#                 tsne = TSNE(n_components=2, verbose=1, n_iter=300)
#                 tsne_results = tsne.fit_transform(X)

#                 results = {'tsne-2d-one': tsne_results[:,0],
#                            'tsne-2d-two': tsne_results[:,1],
#                            'y': y}

#                 plt.figure(figsize=(8,6))
#                 sns.scatterplot(
#                     x="tsne-2d-one", y="tsne-2d-two",
#                     hue="y",
#                     data=results,
#                     palette=sns.color_palette("Set2"),
#                     legend="full",
#                 )
#                 plt.legend(["Healthy","Asthma"])
#                 plt.title("Labels")
#                 plt.show()
                
#             if train and epoch==199:
#                 X = torch.cat((X, intermediates.detach().cpu()), dim=0)
#                 y = torch.cat((y, labels[:,0].long().detach().cpu()), dim=0)
            
#         if train and epoch==199:
#             print(pairwise_distances(X[y==0,:]).mean())
#             print(pairwise_distances(X[y==1,:]).mean())
#             print(pairwise_distances(X[y==0,:], X[y==1,:]).mean())
#             tsne = TSNE(n_components=2, verbose=1, n_iter=300)
#             tsne_results = tsne.fit_transform(X)

#             results = {'tsne-2d-one': tsne_results[:,0],
#                        'tsne-2d-two': tsne_results[:,1],
#                        'y': y}

#             plt.figure(figsize=(8,6))
#             sns.scatterplot(
#                 x="tsne-2d-one", y="tsne-2d-two",
#                 hue="y",
#                 data=results,
#                 palette=sns.color_palette("Set2"),
#                 legend="full",
#             )
#             plt.legend(["Healthy","Asthma"])
#             plt.title("Labels")
#             plt.show()
        
        avg_loss /= total_size
        if train:
            self.writer.add_scalar('Loss/pre-train', avg_loss, epoch)
            self.train_loss = avg_loss
        elif validate:
            self.writer.add_scalar('Loss/pre-validate', avg_loss, epoch)
            # Save checkpoint.
            if avg_loss < self.best_loss:
                self.save(avg_loss, epoch)
                self.best_loss = avg_loss