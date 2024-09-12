from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from typing import Optional, Any

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = SequenceEmbedding(num_inputs=1, num_features=8)
        self.encoder = SequenceEncoder(num_features=8, num_layers=1)
        self.output_layer = nn.Linear(8, 1)
    
    def forward(self, x1, x2):
        x = self.embedding(x1)
        x = self.encoder(x)
        
        y = self.output_layer(x)
        y = y.squeeze(-1)

        return y

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cls', torch.rand(8))
        self.embedding_csa = SequenceEmbedding(num_inputs=1, num_features=8)
        self.embedding_demogr = nn.Linear(4, 8)
        self.encoder = SequenceEncoder(num_features=8, num_layers=1)
        self.output_layer = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x1, x2):
        x1 = self.embedding_csa(x1) # (seq, batch, feature)
        
        x2 = self.embedding_demogr(x2) # (batch, feature)
        x2 = x2.unsqueeze(0) # (1, batch, feature)
        c = self.cls.unsqueeze(0).unsqueeze(1) # (1, 1, feature)
        c = c.expand(1, x2.shape[1], -1) # (1, batch, feature)
        
        x = torch.cat((c,x2,x1), 0) # Concatenate <CLS>, demographic embeddings, and CSA embeddings        
        
        x = self.encoder(x)
        x = x[:,0,:] # Only take <CLS> representations
        y = self.output_layer(x)
        y = self.softmax(y)

        return y
    
class SequenceEmbedding(nn.Module):
    def __init__(self, num_inputs, num_features):
        super().__init__()
        self.embedding = nn.Conv1d(in_channels=num_inputs, out_channels=num_features, kernel_size=3, padding=1)
        self.pos_encoder = PositionalEncoding(d_model=num_features)
    
    def forward(self, x):
        x = x.unsqueeze(1) # (batch, feature, seq)
        x = self.embedding(x)
        x = x.permute(2,0,1) # (seq, batch, feature)
        x = self.pos_encoder(x)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, num_features, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_features, nhead=4, dim_feedforward=4*num_features)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        y = self.transformer_encoder(x)
        y = y.permute(1,0,2)  # (batch, seq, feature)
#         y = self.dropout(y)

        return y

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)