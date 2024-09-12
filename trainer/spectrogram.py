import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import os
import matplotlib.pyplot as plt

import sklearn.metrics as M
from captum.attr import IntegratedGradients

class Trainer:
    def __init__(self, model, lr, T_max, device, summarywriter, class_weights=None):
        self.model = model
        self.device = device
        self.writer = summarywriter
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max) #, eta_min=lr/10)
        self.train_acc = 0
        self.best_acc = 0
        
    def train(self, epoch, data_loader):
        self.model.train()
        self.iteration(epoch, data_loader)
        self.scheduler.step()

    def validate(self, epoch, data_loader):
        self.model.eval()
        with torch.no_grad():
            self.iteration(epoch, data_loader, train=False, validate=True)
            
    def test(self, data_loader, no_print=False, calculate_ig=False):
        self.model.load_state_dict(torch.load('./checkpoint/ckpt.pth')['net'])
        self.model.eval()
        with torch.no_grad():
            self.iteration(None, data_loader, train=False, validate=False, calculate_ig=calculate_ig)
        if not no_print:
            print('Best accuracy (balanced): %.2f%%' % (100*self.balanced_accuracy))
            print("On which epoch reach the highest accuracy:",
                  torch.load('./checkpoint/ckpt.pth')['epoch'])
        
    def save(self, acc, epoch):
        state = {
            'net': self.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        
        
    def iteration(self, epoch, data_loader, train=True, validate=False, calculate_ig=False):
        total_size = 0
        avg_loss = 0
        labels_all = torch.Tensor([])
        # demogr_all = torch.Tensor([])
        outputs_all = torch.Tensor([])
        
        # Initiate IG algorithm
        if calculate_ig:
            ig = IntegratedGradients(self.model)
            attr_ig_all = torch.Tensor([])
        
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            # Step 1: put data to gpu
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Step 2: forward model
            # outputs = self.model(pixel_values=inputs)
            outputs = self.model(inputs)
            
            # Step 3: calculate loss and accuracy
            loss = self.criterion(outputs, labels)
            total_size += labels.size(0)
            avg_loss += loss.item()*labels.size(0)
            
            # Step 4: back propagation
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Step 5: record details
            labels_all = torch.cat((labels_all, labels.cpu()), dim=0)
            # demogr_all = torch.cat((demogr_all, demogr.cpu()), dim=0)
            outputs_all = torch.cat((outputs_all, outputs.cpu()), dim=0)
            
            # Others: Calculate IG
            if calculate_ig:
                attr_ig = ig.attribute(inputs, 
                                       target=torch.argmax(outputs, dim=1),
                                       n_steps=100)
                attr_ig = torch.cat((attr_ig[0], attr_ig[1]), dim=1)
                attr_ig_all = torch.cat((attr_ig_all, attr_ig.cpu()), dim=0)
        
        avg_loss /= total_size
        preds_all = torch.argmax(outputs_all, dim=1)
        acc = M.accuracy_score(labels_all, preds_all)
        bal_acc = M.balanced_accuracy_score(labels_all, preds_all)
        if train:
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('Acc/train', acc, epoch)
            self.writer.add_scalar('BalAcc/train', bal_acc, epoch)
            self.train_acc = bal_acc
        elif validate:
            self.writer.add_scalar('Loss/validate', avg_loss, epoch)
            self.writer.add_scalar('Acc/validate', acc, epoch)
            self.writer.add_scalar('BalAcc/validate', bal_acc, epoch)
            # Save checkpoint.
            if bal_acc >= self.best_acc:
                self.save(bal_acc, epoch)
                self.best_acc = bal_acc
        else:
            self.labels = labels_all
            # self.demogr = demogr_all
            self.outputs = outputs_all
            if calculate_ig:
                self.attr_ig = attr_ig_all
 
            cm = M.confusion_matrix(labels_all, preds_all)
            tn, fp, fn, tp = cm.ravel()
            self.accuracy = acc
            self.balanced_accuracy = bal_acc
            self.sensitivity = tp/(tp+fn)
            self.specificity = tn/(tn+fp)
            self.auroc = M.roc_auc_score(labels_all, outputs_all[:,1])

            self.results = {
                'Loss': avg_loss,
                'Acc': acc,
                'BalAcc': bal_acc,
                'Sensitivity': tp/(tp+fn),
                'Specificity': tn/(tn+fp),
                'AUROC': M.roc_auc_score(labels_all, outputs_all[:,1])
            }
               