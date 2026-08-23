import os
import sys
import random
import numpy as np
import torch
import torchvision

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_env_info(model_name: str, n_train: int, n_val: int, batch_size: int, epochs: int):
    device = get_device()
    print("=" * 60)
    print("  HEAD POSE ESTIMATION (REGRESSION)")
    print("=" * 60)
    print(f"  Python         : {sys.version.split()[0]}")
    print(f"  PyTorch        : {torch.__version__}")
    print(f"  torchvision    : {torchvision.__version__}")

    if torch.cuda.is_available():
        print(f"  CUDA tersedia  : Ya")
        print(f"  Versi CUDA     : {torch.version.cuda}")
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    else:
        print(f"  CUDA tersedia  : Tidak  (Peringatan: Menggunakan CPU)")

    print(f"  Device         : {device}")
    print(f"  Model          : {model_name}")
    print(f"  Data training  : {n_train:,}")
    print(f"  Data validasi  : {n_val:,}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Epochs         : {epochs}")
    print("=" * 60)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_model_info(model: torch.nn.Module, model_name: str):
    total, trainable = count_parameters(model)
    print(f"\n  [Model] {model_name}")
    print(f"    Total parameters     : {total:,}")
    print(f"    Trainable parameters : {trainable:,}")

def ensure_dirs():
    for d in ["outputs/models", "outputs/reports", "outputs/plots"]:
        os.makedirs(d, exist_ok=True)
