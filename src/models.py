import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
)

def build_efficientnet() -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = tvm.efficientnet_b0(weights=weights)

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier with Linear(in_features, 3)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 3),
    )
    
    # Ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

def unfreeze_efficientnet_last_block(model: nn.Module):
    # Unfreeze 3 blok terakhir (features[-3:]) untuk adaptasi fitur yang lebih baik
    for block in model.features[-3:]:
        for param in block.parameters():
            param.requires_grad = True
    return [name for name, param in model.named_parameters() if param.requires_grad]

def build_mobilenet() -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = tvm.mobilenet_v3_small(weights=weights)

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 3),
    )
    
    # Ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

def unfreeze_mobilenet_last_block(model: nn.Module):
    # Unfreeze 3 blok terakhir (features[-3:]) untuk adaptasi fitur yang lebih baik
    for block in model.features[-3:]:
        for param in block.parameters():
            param.requires_grad = True
    return [name for name, param in model.named_parameters() if param.requires_grad]

def get_model(model_name: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "efficientnet":
        return build_efficientnet()
    elif model_name == "mobilenet":
        return build_mobilenet()
    else:
        raise ValueError(f"Model tidak dikenal: '{model_name}'")

def unfreeze_last_block(model: nn.Module, model_name: str):
    model_name = model_name.lower()
    if model_name == "efficientnet":
        return unfreeze_efficientnet_last_block(model)
    elif model_name == "mobilenet":
        return unfreeze_mobilenet_last_block(model)
