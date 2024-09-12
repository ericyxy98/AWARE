import torch
from torch import nn
from transformers import VivitModel

# image_processor = VivitImageProcessor(
#     do_resize = True,
#     do_rescale = False,
#     offset = False,
#     size = {'height':112, 'width':112},
#     crop_size = {'height':112, 'width':112},
#     image_mean = [0.5],
#     image_std = [0.5],
#     input_data_format = 'channels_last'
# )

class ViViT(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.vivit = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, x):
        outputs = self.vivit(pixel_values=x)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        return logits