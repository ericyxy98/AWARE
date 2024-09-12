import torch
from torch import nn
from transformers import ASTModel, ASTForAudioClassification

class AST(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.ast.classifier.dense = nn.Linear(768, num_labels)
        
    def forward(self, x):
        logits = self.ast(pixel_values=x).logits
        return logits