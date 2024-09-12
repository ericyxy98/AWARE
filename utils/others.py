import torch
import pandas as pd
import numpy as np

def weight_reset(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
                
class Metrics():
    def __init__(self):
        super().__init__()
        self.metrics = {
            'Accuracy':[], 
            'Balanced_Acc':[],
            'Sensitivity':[],
            'Specificity':[],
            'AUROC':[],
            'FEV1':[],
            'FEV1/FVC':[]
        }
        
    def append_from(self, trainer):
        self.metrics['Accuracy'].append(trainer.accuracy)
        self.metrics['Balanced_Acc'].append(trainer.balanced_accuracy)
        self.metrics['Sensitivity'].append(trainer.sensitivity)
        self.metrics['Specificity'].append(trainer.specificity)
        self.metrics['AUROC'].append(trainer.auroc)
        self.metrics['FEV1'].append(trainer.fev1_mape)
        self.metrics['FEV1/FVC'].append(trainer.ratio_mape)

    def __str__(self):
        df = pd.DataFrame(self.metrics)
        out = str(df) + "\nAVERAGE:\n" + str(df.mean(axis=0).to_frame().T)
        return out
    
    def _ipython_display_(self):
        df = pd.DataFrame(self.metrics)
        df_avg = df.mean(axis=0).to_frame().T
        df_avg.index = ['Avg']
        display(pd.concat((df, df_avg)))
    
class Outputs():
    def __init__(self):
        super().__init__()
        self.outputs = pd.DataFrame({
            'Labels':[], 
            'Info':[],
            'Outputs_reg':[],
            'Outputs_cls':[]
        })
        
    def append_from(self, trainer):
        df = pd.DataFrame({
            'Labels':trainer.labels.tolist(), 
            'Info':trainer.info.tolist(),
            'Outputs_reg':trainer.outputs_reg.tolist(),
            'Outputs_cls':trainer.outputs_cls.tolist()
        })
        self.outputs = pd.concat((self.outputs, df)).reset_index(drop=True)

    def __str__(self):
        return str(self.outputs)
    
    def _ipython_display_(self):
        display(self.outputs)
        
class BasicMetrics():
    def __init__(self):
        super().__init__()
        self.metrics = {
            'Accuracy':[], 
            'Balanced_Acc':[],
            'Sensitivity':[],
            'Specificity':[],
            'AUROC':[]
        }
        
    def append_from(self, trainer):
        self.metrics['Accuracy'].append(trainer.accuracy)
        self.metrics['Balanced_Acc'].append(trainer.balanced_accuracy)
        self.metrics['Sensitivity'].append(trainer.sensitivity)
        self.metrics['Specificity'].append(trainer.specificity)
        self.metrics['AUROC'].append(trainer.auroc)

    def __str__(self):
        df = pd.DataFrame(self.metrics)
        out = str(df) + "\nAVERAGE:\n" + str(df.mean(axis=0).to_frame().T)
        return out
    
    def _ipython_display_(self):
        df = pd.DataFrame(self.metrics)
        df_avg = df.mean(axis=0).to_frame().T
        df_avg.index = ['Avg']
        display(pd.concat((df, df_avg)))

class RegressionMetrics():
    def __init__(self):
        super().__init__()
        self.metrics = []
        
    def append_from(self, trainer):
        self.metrics.append(trainer.results)

    def __str__(self):
        df = pd.DataFrame(self.metrics)
        out = str(df) + "\nAVERAGE:\n" + str(df.mean(axis=0).to_frame().T)
        return out
    
    def _ipython_display_(self):
        df = pd.DataFrame(self.metrics)
        df_avg = df.mean(axis=0).to_frame().T
        df_avg.index = ['Avg']
        display(pd.concat((df, df_avg)))
    
class BasicOutputs():
    def __init__(self):
        super().__init__()
        self.outputs = pd.DataFrame({
            'Labels':[], 
            'Outputs':[],
            # 'Info':[]
        })
        
    def append_from(self, trainer):
        df = pd.DataFrame({
            'Labels':trainer.labels.tolist(), 
            'Outputs':trainer.outputs.tolist(),
            # 'Info':trainer.info.tolist()
        })
        self.outputs = pd.concat((self.outputs, df)).reset_index(drop=True)

    def __str__(self):
        return str(self.outputs)
    
    def _ipython_display_(self):
        display(self.outputs)