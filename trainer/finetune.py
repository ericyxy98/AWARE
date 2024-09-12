import torch
import torch.nn as nn
import torch.optim as optim

import random
import os
import matplotlib.pyplot as plt

import sklearn.metrics as M
from captum.attr import IntegratedGradients

class FineTuner:
    def __init__(self, model, lr, T_max, device, summarywriter, class_weights=None):
        self.model = model
        self.device = device
        self.writer = summarywriter
        
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
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
        self.model.load_state_dict(torch.load('./checkpoint/ckpt_finetune.pth')['net'])
        self.model.eval()
        with torch.no_grad():
            self.iteration(None, data_loader, train=False, validate=False, calculate_ig=calculate_ig)
        if not no_print:
            print('Best fine-tuning accuracy (balanced): %.2f%%' % (100*self.balanced_accuracy))
            print("On which epoch reach the highest accuracy:",
                  torch.load('./checkpoint/ckpt_finetune.pth')['epoch'])
        
    def save(self, acc, epoch):
        state = {
            'net': self.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_finetune.pth')
        
        
    def iteration(self, epoch, data_loader, train=True, validate=False, calculate_ig=False):
        total_size = 0
        avg_loss1 = 0
        avg_loss2 = 0
        labels_all = torch.Tensor([])
        info_all = torch.Tensor([])
        outputs_reg_all = torch.Tensor([])
        outputs_cls_all = torch.Tensor([])
        
        # Initiate IG algorithm
        if calculate_ig:
            def wrapped_model(inp_1, inp_2):
                return self.model(inp_1, inp_2)[1]
            ig = IntegratedGradients(wrapped_model)
            attr_ig_all = torch.Tensor([])
        
        for batch_idx, (inputs, masks, labels, info) in enumerate(data_loader):
            # Step 1: put data to gpu
            inputs, labels, info = inputs.to(self.device), labels.to(self.device), info.to(self.device)
            
            # Step 2: forward model
            outputs_reg, outputs_cls = self.model(inputs, info[:,2:6])
            
            # Step 3: calculate loss and accuracy
#             target_reg = torch.cat((labels[:,4:6],labels[:,7:13]),dim=1)
#             target_reg[target_reg.isnan()] = outputs_reg[target_reg.isnan()]
            loss1 = self.criterion1(outputs_reg, labels[:,4:6])*1e-3
            loss2 = self.criterion2(outputs_cls, labels[:,0].long())
            loss = loss1 + loss2
            total_size += labels.size(0)
            avg_loss1 += loss1.item()*labels.size(0)
            avg_loss2 += loss2.item()*labels.size(0)
            
            # Step 4: back propagation
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Step 5: record details
            labels_all = torch.cat((labels_all, labels.cpu()), dim=0)
            info_all = torch.cat((info_all, info.cpu()), dim=0)
            outputs_reg_all = torch.cat((outputs_reg_all, outputs_reg.cpu()), dim=0)
            outputs_cls_all = torch.cat((outputs_cls_all, outputs_cls.cpu()), dim=0)
            
            # Others: Calculate IG
            if calculate_ig:
                attr_ig = ig.attribute((inputs, info[:,2:6]), 
                                       target=torch.argmax(outputs_reg, dim=1),
                                       n_steps=100)
                attr_ig = torch.cat((attr_ig[0], attr_ig[1]), dim=1)
                attr_ig_all = torch.cat((attr_ig_all, attr_ig.cpu()), dim=0)
        
        avg_loss1 /= total_size
        avg_loss2 /= total_size
        preds_all = torch.argmax(outputs_cls_all, dim=1)
        acc = M.accuracy_score(labels_all[:,0], preds_all)
        bal_acc = M.balanced_accuracy_score(labels_all[:,0], preds_all)
        if train:
            self.writer.add_scalar('Loss/train_1', avg_loss1, epoch)
            self.writer.add_scalar('Loss/train_2', avg_loss2, epoch)
            self.writer.add_scalar('Acc/train', acc, epoch)
            self.writer.add_scalar('BalAcc/train', bal_acc, epoch)
            self.train_acc = bal_acc
        elif validate:
            self.writer.add_scalar('Loss/validate_1', avg_loss1, epoch)
            self.writer.add_scalar('Loss/validate_2', avg_loss2, epoch)
            self.writer.add_scalar('Acc/validate', acc, epoch)
            self.writer.add_scalar('BalAcc/validate', bal_acc, epoch)
            # Save checkpoint.
            if bal_acc >= self.best_acc:
                self.save(bal_acc, epoch)
                self.best_acc = bal_acc
        else:
            self.labels = labels_all
            self.info = info_all
            self.outputs_reg = outputs_reg_all
            self.outputs_cls = outputs_cls_all
            if calculate_ig:
                self.attr_ig = attr_ig_all
                
            cm = M.confusion_matrix(labels_all[:,0], preds_all)
            tn, fp, fn, tp = cm.ravel()
            self.accuracy = acc
            self.balanced_accuracy = bal_acc
            self.sensitivity = tp/(tp+fn)
            self.specificity = tn/(tn+fp)
            self.auroc = M.roc_auc_score(labels_all[:,0], outputs_cls_all[:,1])
            self.fev1_mape = M.mean_absolute_percentage_error(outputs_reg_all[:,0], labels_all[:,4])
            self.ratio_mape = M.mean_absolute_percentage_error(outputs_reg_all[:,1], labels_all[:,5])