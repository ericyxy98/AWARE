import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats, signal, io
from PIL import Image
import os
from tqdm.notebook import tqdm

SAMPLE_RATE = 48000

class Dataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, video=False, dim_order="BCTHW"):
        samples = []
        labels = []
        for i in range(1000):
            img = Image.open(folder_path+"/cat"+str(i+1)+".jpg")
            samples += [img]
            labels += [0]
        for i in range(1000):
            img = Image.open(folder_path+"/dog"+str(i+1)+".jpg")
            samples += [img]
            labels += [1]
        self.data = pd.DataFrame({
            'image': samples,
            'label': labels
        })
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.class_weights = compute_class_weight(
            'balanced', 
            classes=[0,1], 
            y=self.data['label']
        )
        self.video = video
        self.dim_order = dim_order
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data['image'][idx]
        sample = self.transforms(sample)
        if self.video:
            sample = sample.repeat(32, 1, 1, 1)
            if self.dim_order=="BCTHW":
                sample = sample.permute(1,0,2,3)
        label = self.data['label'][idx]
        return sample, label
    
class Splitter:
    def __init__(self, dataset, batch_size, random_seed, k = 5):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.splitter1 = StratifiedKFold(n_splits=5, shuffle=True)  
        self.splitter2 = StratifiedKFold(n_splits=k, shuffle=True)
            
    def __len__(self):
        return self.k
    
    def __iter__(self):
        idx = list(range(0,len(self.dataset)))
        idx, test_idx = next(iter(self.splitter1.split(idx, self.dataset.data['label'])))
        test_set = torch.utils.data.Subset(self.dataset, test_idx)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size)
        for i, (train_idx, val_idx) in enumerate(self.splitter2.split(idx, self.dataset.data['label'][idx])):
            train_idx, val_idx = idx[train_idx], idx[val_idx]  # The split() method returns the split index of of the input "idx", namely the "index of index" in our case
            train_set = torch.utils.data.Subset(self.dataset, train_idx)
            val_set = torch.utils.data.Subset(self.dataset, val_idx)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size)
            yield train_loader, val_loader, test_loader