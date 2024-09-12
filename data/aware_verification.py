import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from utils.masking import generate_mask
import matplotlib.pyplot as plt

# Build dataset from csv files
class AwareDataset(torch.utils.data.Dataset):
    def __init__(self, csv_data, csv_outcome, csv_info, root_dir, target_classes=None):
        self.data_in = pd.read_csv(csv_data, header=None)
        self.data_out = pd.read_csv(csv_outcome)
        self.data_info = pd.read_csv(csv_info)
        self.root_dir = root_dir
        
        print('Outcome Columns:', self.data_out.columns)
        print('Info Columns:', self.data_info.columns)
        print('# of total CSA samples:', len(self.data_in))
        print('# of total subjects:', len(list(set(self.data_info.ID))))
        
        if target_classes != None:
            idx = False
            for i in target_classes:
                idx |= (self.data_out['Diagnosis']==i)
        else:
            idx = True

        for col in ['FEV1_pred', 'FEV1/FVC_pred']:  ### ATTENTION: change the elements when using different indices 
            idx &= (~self.data_out[col].isna())
        self.data_in = self.data_in[idx].reset_index(drop=True)
        self.data_out = self.data_out[idx].reset_index(drop=True)
        self.data_info = self.data_info[idx].reset_index(drop=True)
        print('# of valid CSA samples w/o NaN:', len(self.data_in))
        print('# of subjects w/ valid CSA samples:', len(list(set(self.data_info.ID))))
        
        print('# of healthy samples:', (self.data_out['Diagnosis']==0).sum())
        print('# of asthma sampels:', (self.data_out['Diagnosis']==1).sum())
        
        self.masks = generate_mask(torch.zeros(1000,84), 0.3, 5)
            
        if len(self.data_in) != len(self.data_out) or len(self.data_in) != len(self.data_info):
            raise Exception("Inconsistent data length")
        
        A_in = self.data_in.iloc[0:1]
        A_out = self.data_out.iloc[0:1]
        A_info = self.data_info.iloc[0:1]
        
        B_in = self.data_in.iloc[500:501]
        B_out = self.data_out.iloc[500:501]
        B_info = self.data_info.iloc[500:501]
        
        plt.plot(A_in.iloc[0,34:118])
        plt.plot(B_in.iloc[0,34:118])
        plt.show()
        
        n0 = (self.data_out['Diagnosis']==0).sum()
        n1 = (self.data_out['Diagnosis']==1).sum()
        
        self.data_in =  pd.concat([A_in+1.0*np.random.randn(194), B_in+1.0*np.random.randn(194)])
        self.data_out =  pd.concat([A_out, B_out])
        self.data_info =  pd.concat([A_info, B_info])
        
        for i in range(n0-1):
            self.data_in =  pd.concat([self.data_in, A_in + 1.0*np.random.randn(194)])
            self.data_out =  pd.concat([self.data_out, A_out])
            self.data_info =  pd.concat([self.data_info, A_info])
        for i in range(n1-1):
            self.data_in =  pd.concat([self.data_in, B_in + 1.0*np.random.randn(194)])
            self.data_out =  pd.concat([self.data_out, B_out])
            self.data_info =  pd.concat([self.data_info, B_info])
        
        self.data_in = self.data_in.reset_index(drop=True)
        self.data_out = self.data_out.reset_index(drop=True)
        self.data_info = self.data_info.reset_index(drop=True)
        plt.plot(self.data_in.iloc[:,34:118].T)
        plt.show()
        
        self.data_in_mean = self.data_in.mean().values.astype('float32')
        self.data_in_std = self.data_in.std().values.astype('float32')
        self.data_in_max = self.data_in.abs().max().values.astype('float32')
        
        self.data_info_mean = self.data_info.mean().values.astype('float32')
        self.data_info_std = self.data_info.std().values.astype('float32')
        self.data_info_max = self.data_info.abs().max().values.astype('float32')
        
    def __len__(self):
        return len(self.data_in)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        inputs = self.data_in.iloc[idx, 34:118].values.astype('float32')
        inputs = (inputs - np.mean(inputs)) / np.std(inputs) # Normalization
#         inputs = (inputs-self.data_in_mean[34:118])/self.data_in_std[34:118] # Normalization
        targets = self.data_out.iloc[idx, :].values.astype('float32')
        info = self.data_info.iloc[idx, :].values.astype('float32')
        info[2:6] = (info[2:6] - self.data_info_mean[2:6]) / self.data_info_std[2:6]
        masks = self.masks[np.random.randint(1000), :]

        return inputs, masks, targets, info

class AwareSplitter:
    def __init__(self, dataset, batch_size, random_seed, k = 5):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.splitter1 = StratifiedKFold(n_splits=5, shuffle=True)  # use StratifiedGroupKFold to get a 20% hold-out test set (since there is no StratifiedGroupShuffleSplit)
        self.splitter2 = StratifiedKFold(n_splits=k, shuffle=True)  # K-fold cross-validation (training & validation set)
            
    def __len__(self):
        return self.k
    
    def __iter__(self):
        idx = list(range(0,len(self.dataset)))
        idx, test_idx = next(iter(self.splitter1.split(idx, self.dataset.data_out['Diagnosis'])))
        test_set = torch.utils.data.Subset(self.dataset, test_idx)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_idx))
        for i, (train_idx, val_idx) in enumerate(self.splitter2.split(idx, self.dataset.data_out['Diagnosis'][idx])):
            train_idx, val_idx = idx[train_idx], idx[val_idx]  # The split() method returns the split index of of the input "idx", namely the "index of index" in our case
            train_set = torch.utils.data.Subset(self.dataset, train_idx)
            val_set = torch.utils.data.Subset(self.dataset, val_idx)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_idx))
            yield train_loader, val_loader, test_loader
