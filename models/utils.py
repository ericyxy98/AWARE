from transformers import VivitConfig, VivitImageProcessor, VivitForVideoClassification
from models.vit import ViT, ViT_demogr, ViT_encoder
from models.vivit import ViViT
from models.ast import AST
from models.resnet_video import generate_model
from models.resnet import ResNet18, ResNet34, ResNet50
from models.cnn import CNN1D
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

def select_model(str):
    if str == "resnet_18":
        model = ResNet18()
    if str == "resnet_50":
        model = ResNet50()
    if str == "resnet_50_pretrained":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    if str == "mobilenet_v2":
        model = mobilenet_v2(weights=None)
    if str == "mobilenet_v2_pretrained":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    if str == "video_resnet_18":
        model = generate_model(model_depth=18, n_classes=2)
    if str == "R(2+1)D_pretrained":
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
    if str == "vivit":
        model = ViViT(num_labels=2)
    if str == "vit_binary":
        model = ViT(num_labels=2)
    if str == "vit_four":
        model = ViT(num_labels=4)
    if str == "ast":
        model = AST(num_labels=2)
    if str == "vit_demogr_reg":
        model = ViT_demogr(num_labels=4)
    if str == "vit_encoder":
        model = ViT_encoder()
    if str == "cnn1d_reg":
        model = CNN1D(num_labels=4)
        
    return model