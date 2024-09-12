import torch
from torch import nn
from torch.nn import functional as F
from transformers import ViTModel, ViTForImageClassification

class ViT(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", output_attentions=True)
        self.vit.classifier = nn.Linear(768, num_labels)
        
    def forward(self, x):
        x = self.vit(pixel_values=x)
        self.attentions = x.attentions
        logits = x.logits
        return logits

class ViT_demogr(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_attentions=True)
        self.mlp = nn.Linear(4, 32)
        self.classifier = nn.Linear(768 + 32, num_labels)
        
    def forward(self, x, y):
        x = self.vit(pixel_values=x)
        self.attentions = x.attentions
        x = x.last_hidden_state[:, 0, :]
        y = F.relu(self.mlp(y))
        x = torch.cat((x,y), -1)
        logits = self.classifier(x)
        return logits

class ViT_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", output_attentions=True)
        
    def forward(self, x):
        x = self.vit(pixel_values=x)
        self.attentions = x.attentions
        fea = x.last_hidden_state[:, 0, :]
        return fea