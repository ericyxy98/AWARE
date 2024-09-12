import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold, LeaveOneGroupOut, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats, signal, io
import random
import pickle
import os
from tqdm.notebook import tqdm

AGE_GROUP = list(range(0, 78, 6))

class AwareCSA(torch.utils.data.Dataset):
    def __init__(self, csv_data, csv_outcome, csv_info, redcap_csv_file, id_map_file, pickle_file=None,
                target_classes=None, age_balanced=False, bronchodilator=False,
                output_demogr=False, output_spiro_raw=False, output_spiro_pred=False, output_spiro_bdr=False, output_disease_label=True,
                output_oscil_raw=False, output_oscil_zscore=False):
        self.output_demogr = output_demogr
        self.output_spiro_raw = output_spiro_raw
        self.output_spiro_pred = output_spiro_pred
        self.output_spiro_bdr = output_spiro_bdr
        self.output_oscil_raw = output_oscil_raw
        self.output_oscil_zscore = output_oscil_zscore
        self.output_disease_label = output_disease_label
        
        if pickle_file!=None:
            self.data = pd.read_pickle(pickle_file)
            return

        data_csa = pd.read_csv(csv_data, header=None)
        data_id = pd.read_csv(csv_info)[['ID', 'Trial']]
        self.data = pd.read_csv(redcap_csv_file)
        self.data = self.data[COLUMNS_EXTENDED]
        self.data = self.data.set_index('AWARE STUDY ID:', drop=False)
        self.data.index.name = None
        self.data = self.data.loc[data_id['ID']].reset_index(drop=True)
        self.data['Test Number'] = data_id['Trial']
        self.data['CSA'] = pd.Series(data_csa.values.tolist())
        
        if target_classes != None:
            idx = False
            for i in range(len(target_classes)):
                idx |= (self.data['Participant:']==target_classes[i])
            if self.output_spiro_raw:
                for col in COLUMNS_SPIROMETRY_RAW:
                    idx &= (~self.data[col].isna())
            if self.output_spiro_pred:
                for col in COLUMNS_SPIROMETRY_PRED:
                    idx &= (~self.data[col].isna())
            if self.output_spiro_bdr:
                for col in COLUMNS_SPIROMETRY_BDR:
                    idx &= (~self.data[col].isna())
            self.data = self.data[idx].reset_index(drop=True)
        
        if age_balanced:
            for k in range(len(AGE_GROUP)-1):
                random.seed(123)
                idx_0 = (self.data['Calculated age (years):']>=AGE_GROUP[k]) & (self.data['Calculated age (years):']<AGE_GROUP[k+1]) & (self.data['Participant:']==target_classes[0])
                idx_0 = self.data.index[idx_0].to_list()
                idx_1 = (self.data['Calculated age (years):']>=AGE_GROUP[k]) & (self.data['Calculated age (years):']<AGE_GROUP[k+1]) & (self.data['Participant:']==target_classes[1])
                idx_1 = self.data.index[idx_1].to_list()
                if len(idx_0) > len(idx_1):
                    idx = random.sample(idx_0, len(idx_0)-len(idx_1))
                    idx.sort()
                    self.data = self.data.drop(idx).reset_index(drop=True)
                elif len(idx_1) > len(idx_0):
                    idx = random.sample(idx_1, len(idx_1)-len(idx_0))
                    idx.sort()
                    self.data = self.data.drop(idx).reset_index(drop=True)
                    
        self.class_weights = compute_class_weight(
            'balanced', 
            classes=target_classes, 
            y=self.data['Participant:']
        )
        
    def save_to_pickle(self, filename):
        display("Dumping data to pickle file ...")
        self.data.to_pickle(filename)
        display("Complete")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = np.array(self.data['CSA'][idx]).astype('float32')
        sample = sample[34:118]
        label = LABEL_TO_NUM[self.data['Participant:'][idx]]
        
        record = (sample,)
        if self.output_demogr:
            demogr = self.data[COLUMNS_DEMOGRAPHICS].loc[idx]
            demogr['Sex:'] = SEX_TO_NUM[demogr['Sex:']]
            demogr = demogr.astype('float32').to_numpy()
            record += (demogr,)
        if self.output_spiro_raw:
            spiro = self.data[COLUMNS_SPIROMETRY_RAW].loc[idx]
            spiro['Baseline FEV1/FVC (raw ratio):'] /= 100
            spiro = spiro.astype('float32').to_numpy()
            record += (spiro,)
        if self.output_spiro_pred:
            spiro = self.data[COLUMNS_SPIROMETRY_PRED].loc[idx]
            spiro = spiro.astype('float32').to_numpy()
            spiro /= 100
            record += (spiro,)
        if self.output_spiro_bdr:
            spiro_bdr = self.data[['AWARE STUDY ID:'] + COLUMNS_META + COLUMNS_SPIROMETRY_BDR].loc[idx]
            record += (spiro_bdr,)
        if self.output_disease_label:
            record += (label,)
        
        return record
    
class AwareSplitter:
    def __init__(self, dataset, batch_size, random_seed, k = 5):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.splitter1 = StratifiedGroupKFold(n_splits=5, shuffle=True)  # use StratifiedGroupKFold to get a 20% hold-out test set (since there is no StratifiedGroupShuffleSplit)
        self.splitter2 = StratifiedGroupKFold(n_splits=k, shuffle=True)  # K-fold cross-validation (training & validation set)
            
    def __len__(self):
        return self.k
    
    def __iter__(self):
        idx = list(range(0,len(self.dataset)))
        idx, test_idx = next(iter(self.splitter1.split(idx, self.dataset.data['Participant:'], groups=self.dataset.data['AWARE STUDY ID:'][idx])))
        test_set = torch.utils.data.Subset(self.dataset, test_idx)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size)
        for i, (train_idx, val_idx) in enumerate(self.splitter2.split(idx, self.dataset.data['Participant:'][idx], groups=self.dataset.data['AWARE STUDY ID:'][idx])):
            train_idx, val_idx = idx[train_idx], idx[val_idx]  # The split() method returns the split index of of the input "idx", namely the "index of index" in our case
            train_set = torch.utils.data.Subset(self.dataset, train_idx)
            val_set = torch.utils.data.Subset(self.dataset, val_idx)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size)
            yield train_loader, val_loader, test_loader

class KFoldSplitter:
    def __init__(self, dataset, batch_size, random_seed, k = 5):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.dataset = dataset
        self.batch_size = batch_size
        self.splitter = GroupKFold(n_splits=k)
            
    def __len__(self):
        return self.k
    
    def __iter__(self):
        idx = list(range(0,len(self.dataset)))
        for i, (train_idx, test_idx) in enumerate(self.splitter.split(idx, groups=self.dataset.data['AWARE STUDY ID:'])):
            train_set = torch.utils.data.Subset(self.dataset, train_idx)
            test_set = torch.utils.data.Subset(self.dataset, test_idx)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size)
            yield train_loader, test_loader
            
            
LABEL_TO_NUM = {
    'Control / healthy / no pulmonary disease': 0,
    'Asthma': 1,
    'CF': 2,
    'COPD': 3,
    'Others': 4
}

NUM_TO_LABEL = {
    0: 'Control / healthy / no pulmonary disease',
    1: 'Asthma',
    2: 'CF',
    3: 'COPD',
    4: 'Others'
}

SEX_TO_NUM = {
    'Male': 0,
    'Female': 1
}

COLUMNS_META = [
    'Test Number',
    'Post-BD'
]

COLUMNS_SIGNALS = [
    'Calibration_1',
    'Calibration_2',
    'Calibration_3',
    'Nasal',
    'Inhale_1', 'Inhale_2', 'Inhale_3', 
    'Exhale_1', 'Exhale_2', 'Exhale_3'
]

COLUMNS_DEMOGRAPHICS = [
    'Calculated age (years):',
    'Sex:',
    'Height (cm):',
    'Weight (kg):'
]

COLUMNS_SPIROMETRY_RAW = [
    'Baseline FEV1 (liters):',
    'Baseline FVC (liters):',
    'Baseline FEV1/FVC (raw ratio):',
    'Baseline FEF2575 (liters):'
]

COLUMNS_SPIROMETRY_PRED = [
    'Baseline FEV1 (%pred):',
    'Baseline FVC (%pred):',
    'Baseline FEV1/FVC (%pred):',
    'Baseline FEF2575 (%pred):'
]

COLUMNS_SPIROMETRY_BDR = [
    'BDR (as percent of baseline FEV1)',
    'BDR (as percent of predicted FEV1)',
    'BDR (liters)',
    'BDR (difference in %preds)'
]

COLUMNS_SIMPLIFIED = [
    'AWARE STUDY ID:',
    'Calculated age (years):',
    'Sex:',
    'Height (cm):',
    'Weight (kg):',
    'Participant:'
]

COLUMNS_EXTENDED = [
    'AWARE STUDY ID:',
    'Calculated age (years):',
    'Sex:',
    'Race/ethnicity: (choice=White)',
    'Race/ethnicity: (choice=Black / African American)',
    'Race/ethnicity: (choice=Hispanic / Latino)',
    'Race/ethnicity: (choice=Asian)',
    'Race/ethnicity: (choice=Other)',
    'Height (cm):',
    'Weight (kg):',
    'Participant:',
    'R5', 'R5-20', 'R20', 'X5', 'AX', 'TV',
    'R5 z-score',
    'R5-20 z-score',
    'R20 z-score',
    'X5 z-score',
    'AX z-score',
    'Baseline FEV1 (liters):',
    'Baseline FVC (liters):',
    'Baseline FEV1/FVC (raw ratio):',
    'Baseline FEF2575 (liters):',
    'Predictive equations used:',
    'Baseline FEV1 (%pred):',
    'Baseline FVC (%pred):',
    'Baseline FEV1/FVC (%pred):',
    'Baseline FEF2575 (%pred):',
    'Pre-BD \'score\'',
    'Post-BD \'score\'',
    'Post FEV1 (liters):',
    'Post FVC (liters):',
    'Post FEV1/FVC (raw ratio):',
    'Post FEF2575 (liters):',
    'Post FEV1 (%pred):',
    'Post FVC (%pred):',
    'Post FEV1/FVC (%pred):',
    'Post FEF2575 (%pred):',
    'BDR (as percent of baseline FEV1)',
    'BDR (as percent of predicted FEV1)',
    'BDR (liters)',
    'BDR (difference in %preds)'
]