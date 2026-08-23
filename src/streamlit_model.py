import os
import torch
import torch.nn as nn
import torchvision.models as tvm
import streamlit as st

def _build_efficientnet() -> nn.Module:
    model = tvm.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 3),
    )
    return model

def _build_mobilenet() -> nn.Module:
    model = tvm.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 3),
    )
    return model

@st.cache_resource(show_spinner=False)
def load_model(model_name: str, checkpoint_path: str) -> nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file tidak ditemukan di {checkpoint_path}")

    model_name = model_name.lower()
    if "efficientnet" in model_name:
        model = _build_efficientnet()
    elif "mobilenet" in model_name:
        model = _build_mobilenet()
    else:
        raise ValueError(f"Model tidak dikenal: '{model_name}'")

    # Load state dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Gagal memuat checkpoint: {e}")

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Handle 'module.' prefix from DataParallel/DistributedDataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        raise RuntimeError(f"Arsitektur tidak cocok dengan state_dict: {e}")

    model.eval()
    return model
