import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold, LeaveOneGroupOut, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats, signal, io
import cv2
import sqlite3
import random
import pickle
import os
from tqdm.auto import tqdm

SAMPLE_RATE = 48000
MAX_SXX_AVE_AMP = 0.2
MIN_SXX_AVE_AMP = 0.05
CONTOUR_THRESHOLD = 0.45
AGE_GROUP = list(range(0, 78, 6))

class AwareRaw(torch.utils.data.Dataset):
    def __init__(self, redcap_csv_file, id_map_file, aware_db_file, pickle_file=None):
        if pickle_file!=None:
            self.data = pd.read_pickle(pickle_file)
            return
        
        data_info = pd.read_csv(redcap_csv_file)
        data_info = data_info[COLUMNS_EXTENDED]
        data_info = data_info.set_index('AWARE STUDY ID:', drop=False)
        data_info.index.name = None
        
        id_map = pd.read_csv(id_map_file, header=None)
        id_map.index = id_map.index + 1
        id_map.columns = ['subject_id', 'post_bd']
        
        self.data = pd.DataFrame(columns = COLUMNS_EXTENDED + ['Test Number'] + COLUMNS_SIGNALS)
        
        con = sqlite3.connect(aware_db_file)
        cur = con.cursor()
        id_list = cur.execute('SELECT ID FROM aware_upload_data').fetchall()
        id_list = [x[0] for x in id_list]         # id_list by default is a list of tuples, need to be convert to pure list
        for i in id_list:
            subject_id = id_map['subject_id'][i]
            if subject_id==0:
                continue
            print(i, subject_id)
            query = (
                'SELECT FULLINFO, CALIDATA, MEASUREDATA1, MEASUREDATA2, '
                'MEASUREDATA3, MEASUREDATA4, MEASUREDATA5 '
                'FROM aware_upload_data '
                'WHERE ID=' + str(i)
            ) 
            for row in cur.execute(query):
                phone_id = row[0].split(',')[-1]
                cur_2 = con.cursor()
                query_2 = (
                    'SELECT CALIDATA '
                    'FROM aware_precali_data '
                    'WHERE PHONE_ID=\'' + phone_id + '\' '
                    'ORDER BY ID DESC LIMIT 1'
                )
                cali1 = cur_2.execute(query_2).fetchone()
                
                cali1 = np.frombuffer(cali1[0], dtype='>i2') # ">i2" stands for big-endiean int16 (integer, 2 bytes)
                
                cali2 = np.frombuffer(row[1], dtype='>i2') # ">i2" stands for big-endiean int16 (integer, 2 bytes)
                
                if len(cali1) < 4.8*SAMPLE_RATE or len(cali2) < 11.5*SAMPLE_RATE:
                    continue
                
                cali3 = cali2[6*SAMPLE_RATE:round(11.5*SAMPLE_RATE)]
                cali2 = cali2[1:round(5.5*SAMPLE_RATE)]
                
                peak = np.argmax(cali1)
                x_ref = cali1[peak-round(0.01*SAMPLE_RATE):peak+round(0.09*SAMPLE_RATE)]
                
                for k, col in enumerate(row[2:7]):
                    if col==None:
                        continue
                    if len(col) % 2 != 0:
                        continue
                    test = np.frombuffer(col, dtype='>i2') # ">i2" stands for big-endiean int16 (integer, 2 bytes)
                    if len(test) < 42*SAMPLE_RATE:
                        if len(test) < 6*5.9*SAMPLE_RATE+5.7*SAMPLE_RATE:
                            continue
                        nasal = test[1:round(5.7*SAMPLE_RATE)]
                        inhale1 = test[round(1*5.9*SAMPLE_RATE+1):round(1*5.9*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                        exhale1 = test[round(2*5.9*SAMPLE_RATE+1):round(2*5.9*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                        inhale2 = test[round(3*5.9*SAMPLE_RATE+1):round(3*5.9*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                        exhale2 = test[round(4*5.9*SAMPLE_RATE+1):round(4*5.9*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                        inhale3 = test[round(5*5.9*SAMPLE_RATE+1):round(5*5.9*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                        exhale3 = test[round(6*5.9*SAMPLE_RATE+1):round(6*5.9*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                    else:
                        if len(test) < 5*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+5.7*SAMPLE_RATE:
                            continue
                        nasal = test[1:round(10.5*SAMPLE_RATE)]
                        inhale1 = test[round(0*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+1):round(0*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                        exhale1 = test[round(1*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+1):round(1*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                        inhale2 = test[round(2*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+1):round(2*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                        exhale2 = test[round(3*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+1):round(3*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                        inhale3 = test[round(4*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+1):round(4*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                        exhale3 = test[round(5*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+1):round(5*5.9*SAMPLE_RATE+10.5*SAMPLE_RATE+5.7*SAMPLE_RATE)]
                    
                    record = pd.DataFrame({
                        'Test Number':k+1,
                        'Post-BD':id_map['post_bd'][i],
                        'Calibration_1':[align_signal(cali1, x_ref, ref_intv=round(0.4*SAMPLE_RATE), ref_num=12)],
                        'Calibration_2':[align_signal(cali2, x_ref, ref_intv=round(0.4*SAMPLE_RATE), ref_num=12)],
                        'Calibration_3':[align_signal(cali3, x_ref, ref_intv=round(0.4*SAMPLE_RATE), ref_num=12)],
                        'Nasal':[align_signal(nasal, x_ref, ref_intv=round(0.2*SAMPLE_RATE), ref_num=25)],
                        'Inhale_1':[align_signal(inhale1, x_ref, ref_intv=round(0.2*SAMPLE_RATE), ref_num=25)],
                        'Inhale_2':[align_signal(inhale2, x_ref, ref_intv=round(0.2*SAMPLE_RATE), ref_num=25)],
                        'Inhale_3':[align_signal(inhale3, x_ref, ref_intv=round(0.2*SAMPLE_RATE), ref_num=25)],
                        'Exhale_1':[align_signal(exhale1, x_ref, ref_intv=round(0.2*SAMPLE_RATE), ref_num=25)],
                        'Exhale_2':[align_signal(exhale2, x_ref, ref_intv=round(0.2*SAMPLE_RATE), ref_num=25)],
                        'Exhale_3':[align_signal(exhale3, x_ref, ref_intv=round(0.2*SAMPLE_RATE), ref_num=25)]
                    })
                    record = pd.concat([data_info.loc[[subject_id]].reset_index(drop=True), record], axis=1)                    
                    self.data = pd.concat([self.data, record]).reset_index(drop=True)
#         display(self.data)
        
    def save_to_csv(self, filename):
        arr = []
        for col_name in COLUMNS_SIGNALS:
            display(col_name)
            tmp = np.array(self.data[col_name].to_list())
            tmp = tmp.reshape(tmp.shape[0], -1)
            arr += [tmp]
        arr = np.concatenate(arr, axis=1)
        df = self.data[COLUMNS_EXTENDED+['Test Number']]
        df = pd.concat([df, pd.DataFrame(arr)], axis=1)
        
        print('Dumping data to .csv file ...')
        for i in tqdm(df.index):
            if i == 0:
                df.loc[[i]].to_csv(filename, header=None, index=None, mode='w')
            else:
                df.loc[[i]].to_csv(filename, header=None, index=None, mode='a')
        print('Complete')

    def save_to_pickle(self, filename):
        display("Dumping data to pickle file ...")
        self.data.to_pickle(filename)
        display("Complete")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.loc[idx]

class AwareSpectrogram(torch.utils.data.Dataset):
    def __init__(self, dataset_raw, pickle_file=None, target_classes=None, age_balanced=False, bronchodilator=False,
                output_demogr=False, output_spiro_raw=False, output_spiro_pred=False, output_spiro_bdr=False,
                output_disease_label=True,output_oscil_raw=False, output_oscil_zscore=False, output_inhale_exhale=False,
                relative_change=False, calibration=False, averaged=False, num_channels=1, dim_order='BCTHW',
                mel_scale=False, modality='spectrogram'):
        self.output_demogr = output_demogr
        self.output_spiro_raw = output_spiro_raw
        self.output_spiro_pred = output_spiro_pred
        self.output_spiro_bdr = output_spiro_bdr
        self.output_oscil_raw = output_oscil_raw
        self.output_oscil_zscore = output_oscil_zscore
        self.output_disease_label = output_disease_label
        self.output_inhale_exhale = output_inhale_exhale
        self.num_channels = num_channels
        self.dim_order=dim_order
        self.modality=modality
        
        if pickle_file!=None:
            self.data = pd.read_pickle(pickle_file)
            return
        
        self.data = dataset_raw.data
        
        if target_classes is not None:
            idx = False
            for i in range(len(target_classes)):
                idx |= (self.data['Participant:']==target_classes[i])
            # Filter out NA data entries
            if self.output_spiro_raw:
                for col in COLUMNS_SPIROMETRY_RAW:
                    idx &= (~self.data[col].isna())
            if self.output_spiro_pred:
                for col in COLUMNS_SPIROMETRY_PRED:
                    idx &= (~self.data[col].isna())
            if self.output_spiro_bdr:
                for col in COLUMNS_SPIROMETRY_BDR:
                    idx &= (~self.data[col].isna())
            if self.output_oscil_raw:
                for col in COLUMNS_OSCILLOMETRY_RAW:
                    idx &= (~self.data[col].isna())
            if self.output_oscil_zscore:
                for col in COLUMNS_OSCILLOMETRY_ZSCORE:
                    idx &= (~self.data[col].isna())
            self.data = self.data[idx].reset_index(drop=True)
            
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(0.5, 0.5)
        ])
        
        arr = []
        for index, row in tqdm(self.data.iterrows()):
            # if row['AWARE STUDY ID:']>8:
            #     break
            if self.output_inhale_exhale:
                cols = [['Inhale_1'],['Exhale_1'],['Inhale_2'],['Exhale_2'],['Inhale_3'],['Exhale_3']]
            else:
                cols = [['Inhale_1','Exhale_1'],['Inhale_2','Exhale_2'],['Inhale_3','Exhale_3']]
            
            for col in cols:
                tmp = np.concatenate(row[col].to_list(), axis=0)
                # tmp = tmp[9:41,:]
                if calibration:
                    cali = row['Calibration_1']
                    cali = np.mean(cali, axis=0)
#                     b = signal.firwin(101, cutoff=12000, window='hamming', fs=48000)
#                     a = [1,0]
                    # Bandpass filter: 1000-12000Hz
                    b, a = signal.butter(N=5, Wn=[1000, 12000], fs=48000, btype='band')
                    ir = np.zeros(tmp.shape)
                    for n in range(tmp.shape[0]):
                        ir[n,:] = np.fft.irfft(np.fft.rfft(tmp[n,:])/np.fft.rfft(cali))
                        ir[n,:] = signal.lfilter(b, a, ir[n,:])
                    tmp = ir[:, 0:960]              
                    
                if averaged:
                    tmp = stats.trim_mean(tmp, 0.2, axis=0)
                
                if calibration:
                    f, t, Sxx = signal.spectrogram(tmp, 48000, window='hann', nperseg=128, noverlap=124)
                    if averaged:
                        Sxx = Sxx[0:Sxx.shape[0]//2, :]
                    else:
                        Sxx = Sxx[:, 0:Sxx.shape[1]//2, :]
                    Sxx = 10*np.log10(Sxx+1e-20)
                    Sxx = np.clip(Sxx, -150, -30) / 120 + 1.25
                elif not mel_scale:
                    f, t, Sxx = signal.spectrogram(tmp/(2**15), 48000, window='hann', nperseg=222, noverlap=181)
                    Sxx = 10*np.log10(Sxx+1e-20)
                    Sxx = np.clip(Sxx, -120, -40) / 80 + 1.5
                # Sxx = Sxx[2:58, 0:56]
                
                if not averaged:
                    Sxx = np.moveaxis(Sxx, 0, -1)
                if not calibration and np.mean(Sxx, axis=(0,1)).max() > MAX_SXX_AVE_AMP:    # Skip noisy samples
                    continue
                if np.mean(Sxx, axis=(0,1)).min() < MIN_SXX_AVE_AMP:    # Skip empty samples
                    continue
                if relative_change and not averaged:
                    msk = np.mean(Sxx, axis=-1) > CONTOUR_THRESHOLD
                    Sxx = Sxx-np.expand_dims(np.mean(Sxx, axis=-1), axis=-1)
                    msk = np.expand_dims(msk,axis=-1)
                    Sxx = Sxx*msk
                    Sxx = np.clip((Sxx*5+1)/2, 0, 1)
                
                new_row = row[COLUMNS_EXTENDED + COLUMNS_META]
                new_row['Spectrogram'] = Sxx
                if calibration:
                    new_row['IR'] = tmp
                if self.output_inhale_exhale:
                    new_row['isExhale'] = (col[0][0:6]=='Exhale')
                arr.append(new_row)

        self.data = pd.DataFrame(arr).reset_index(drop=True)
        
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

        self.class_distribution = np.zeros(len(target_classes))
        for idx, cls in enumerate(target_classes):
            self.class_distribution[idx] = np.sum(self.data['Participant:']==cls)

        if self.output_disease_label:
            self.class_weights = compute_class_weight(
                'balanced', 
                classes=np.array(target_classes), 
                y=self.data['Participant:']
            )
        elif self.output_inhale_exhale:
            self.class_weights = [1,1]
        
    def save_to_pickle(self, filename):
        display("Dumping data to pickle file ...")
        self.data.to_pickle(filename)
        display("Complete")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.modality=='spectrogram':
            sample = self.transforms(self.data['Spectrogram'][idx].astype('float32'))
            if sample.size(0) == 1:
                sample = sample.repeat(self.num_channels,1,1)
            else:
                sample = sample.repeat(self.num_channels,1,1,1)
                if self.dim_order=="BTCHW":
                    sample = sample.permute(1,0,2,3)
        elif self.modality=='ir':
            sample = self.data['IR'][idx].astype('float32')
        else:
            raise Exception('Undefined modality')
            
        label = LABEL_TO_NUM[self.data['Participant:'][idx]]
        period = self.data['isExhale'][idx].astype('int')
        
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
        if self.output_oscil_raw:
            oscil = self.data[COLUMNS_OSCILLOMETRY_RAW].loc[idx]
            oscil = oscil.astype('float32').to_numpy()
            record += (oscil,)
        if self.output_oscil_zscore:
            oscil = self.data[COLUMNS_OSCILLOMETRY_ZSCORE].loc[idx]
            oscil = oscil.astype('float32').to_numpy()
            record += (oscil,)
        if self.output_disease_label:
            record += (label,)
        if self.output_inhale_exhale:
            record += (period,)
        
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
        self.k = k
            
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
            
            
def align_signal(sig, ref, ref_intv : int, ref_num : int):
    ref_len = ref.size
    ref = np.concatenate([ref if i%2==0 else np.zeros(ref_intv-ref_len, dtype=ref.dtype) for i in range(ref_num*2-1)]) # Construct repeated ref signal
    corr = signal.correlate(sig.astype('int64'), ref.astype('int64'), mode="valid")
    peak = np.argmax(corr)
    lags = signal.correlation_lags(sig.size, ref.size, mode="valid")
    lag = lags[peak]
    out = np.zeros(shape=(0, ref_len), dtype=sig.dtype)
    for i in range(ref_num):
        out = np.concatenate([out, [sig[lag+i*ref_intv:lag+i*ref_intv+ref_len]]], axis=0)
    return out

def db_to_wav(self, id_map_file, aware_db_file, path_to_save):
    id_map = pd.read_csv(id_map_file, header=None)
    id_map.index = id_map.index + 1
    id_map.columns = ['subject_id', 'post_albuterol']

    con = sqlite3.connect(aware_db_file)
    cur = con.cursor()
    id_list = cur.execute('SELECT ID FROM aware_upload_data').fetchall()
    id_list = [x[0] for x in id_list]         # id_list by default is a list of tuples, need to be convert to pure list
    for i in id_list:
        subject_id = id_map['subject_id'][i]
        if subject_id==0:
            continue
        print(i, subject_id)
        query = (
            'SELECT FULLINFO, CALIDATA, MEASUREDATA1, MEASUREDATA2, '
            'MEASUREDATA3, MEASUREDATA4, MEASUREDATA5 '
            'FROM aware_upload_data '
            'WHERE ID=' + str(i)
        )
        for j, row in enumerate(cur.execute(query)):
            newpath = path_to_save + str(subject_id) 
            if j > 0:
                newpath = newpath + '_' + str(j+1)
            if not os.path.exists(newpath):
                os.makedirs(newpath)

            phone_id = row[0].split(',')[-1]
            cur_2 = con.cursor()
            query_2 = (
                'SELECT CALIDATA '
                'FROM aware_precali_data '
                'WHERE PHONE_ID=\'' + phone_id + '\' '
                'ORDER BY ID DESC LIMIT 1'
            )
            cali1 = cur_2.execute(query_2).fetchone()

            cali1 = np.frombuffer(cali1[0], dtype='>i2') # ">i2" stands for big-endiean int16 (integer, 2 bytes)

            cali2 = np.frombuffer(row[1], dtype='>i2') # ">i2" stands for big-endiean int16 (integer, 2 bytes)

            if len(cali1) < 4.8*SAMPLE_RATE or len(cali2) < 11.5*SAMPLE_RATE:
                continue

            cali3 = cali2[6*SAMPLE_RATE:round(11.5*SAMPLE_RATE)]
            cali2 = cali2[1:round(5.5*SAMPLE_RATE)]

            io.wavfile.write(newpath+"/cali_1.wav", 48000, cali1)
            io.wavfile.write(newpath+"/cali_2.wav", 48000, cali2)
            io.wavfile.write(newpath+"/cali_3.wav", 48000, cali3)

            for k, col in enumerate(row[2:7]):
                if col==None:
                    continue
                if len(col) % 2 != 0:
                    continue
                test = np.frombuffer(col, dtype='>i2') # ">i2" stands for big-endiean int16 (integer, 2 bytes)
                io.wavfile.write(newpath+"/test_"+str(k+1)+".wav", 48000, test)

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

COLUMNS_OSCILLOMETRY_RAW = [
    'R5', 'R5-20', 'R20', 'X5', 'AX', 'TV'
]

COLUMNS_OSCILLOMETRY_ZSCORE = [
    'R5 z-score',
    'R5-20 z-score',
    'R20 z-score',
    'X5 z-score',
    'AX z-score'
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