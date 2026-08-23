import os
import math
import glob
import json
import random
import logging
import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224

def get_train_transform() -> T.Compose:
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transform() -> T.Compose:
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def read_pose_mat(mat_path: str):
    try:
        mat = sio.loadmat(mat_path)
        if "Pose_Para" not in mat:
            return None
        pose = mat["Pose_Para"].flatten()
        if len(pose) < 3:
            return None
        pitch = float(pose[0]) * 180.0 / math.pi
        yaw   = float(pose[1]) * 180.0 / math.pi
        roll  = float(pose[2]) * 180.0 / math.pi
        return pitch, yaw, roll
    except Exception as e:
        logger.debug(f"Gagal membaca .mat '{mat_path}': {e}")
        return None

def landmarks_to_bbox(mat_path: str, padding_ratio: float = 0.20):
    try:
        mat = sio.loadmat(mat_path)
        if "pt2d" not in mat:
            return None
        pt2d = mat["pt2d"]
        if pt2d.shape[0] == 2:
            x_coords = pt2d[0]
            y_coords = pt2d[1]
        elif pt2d.shape[1] == 2:
            x_coords = pt2d[:, 0]
            y_coords = pt2d[:, 1]
        else:
            return None

        x_min, x_max = float(x_coords.min()), float(x_coords.max())
        y_min, y_max = float(y_coords.min()), float(y_coords.max())

        w = x_max - x_min
        h = y_max - y_min
        pad_x = w * padding_ratio
        pad_y = h * padding_ratio

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = x_max + pad_x
        y2 = y_max + pad_y

        return int(x1), int(y1), int(x2), int(y2)
    except Exception as e:
        logger.debug(f"Gagal membaca landmark '{mat_path}': {e}")
        return None

def scan_300wlp(root: str):
    manifest_path = os.path.join(root, "_cache_manifest.csv")
    if os.path.exists(manifest_path):
        print(f"\n  [300W-LP] Memuat dataset secara instan dari cache: {manifest_path}")
        df = pd.read_csv(manifest_path)
        initial_count = len(df)
        df = df[(df['pitch'].abs() <= 99) & (df['yaw'].abs() <= 99) & (df['roll'].abs() <= 99)]
        filtered_count = len(df)
        if filtered_count < initial_count:
            print(f"  [WARNING] Menghapus {initial_count - filtered_count} sampel 300W-LP dari cache yang di luar rentang ±99°!")
        return df.to_dict('records')

    print("\n  [300W-LP] Memindai dataset (proses ini memakan waktu beberapa menit)...")
    subsets = ["AFW", "AFW_Flip", "HELEN", "HELEN_Flip",
               "IBUG", "IBUG_Flip", "LFPW", "LFPW_Flip"]

    records = []
    no_mat_count = 0
    no_img_count = 0
    fail_mat_count = 0
    img_fail_read_count = 0

    all_images = []
    for subset in subsets:
        subset_dir = os.path.join(root, subset)
        if os.path.isdir(subset_dir):
            all_images.extend(glob.glob(os.path.join(subset_dir, "*.jpg")))
            all_images.extend(glob.glob(os.path.join(subset_dir, "*.png")))

    all_mats_paths = set()
    for subset in subsets:
        subset_dir = os.path.join(root, subset)
        if os.path.isdir(subset_dir):
            all_mats_paths.update(glob.glob(os.path.join(subset_dir, "*.mat")))

    for img_path in all_images:
        base = os.path.splitext(img_path)[0]
        mat_path = base + ".mat"

        if not os.path.isfile(mat_path):
            no_mat_count += 1
            continue

        pose = read_pose_mat(mat_path)
        if pose is None:
            fail_mat_count += 1
            continue

        # Check image can be read (Optional pre-check)
        img = cv2.imread(img_path)
        if img is None:
            img_fail_read_count += 1
            continue

        pitch, yaw, roll = pose

        # Filter outlier: sudut kepala manusia tidak mungkin melebihi ±99 derajat
        if abs(pitch) > 99 or abs(yaw) > 99 or abs(roll) > 99:
            fail_mat_count += 1
            continue

        records.append({
            "image_path": img_path,
            "annotation_path": mat_path,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
        })

    for mat_path in all_mats_paths:
        base = os.path.splitext(mat_path)[0]
        if not os.path.isfile(base + ".jpg") and not os.path.isfile(base + ".png"):
            no_img_count += 1

    print(f"  [300W-LP] Hasil Audit:")
    print(f"    Total gambar ditemukan      : {len(all_images)}")
    print(f"    Total anotasi .mat ditemukan: {len(all_mats_paths)}")
    print(f"    Total pasangan valid        : {len(records)}")
    print(f"    Gambar tanpa anotasi        : {no_mat_count}")
    print(f"    Anotasi tanpa gambar        : {no_img_count}")
    print(f"    Gambar gagal dibaca         : {img_fail_read_count}")
    print(f"    Anotasi gagal dibaca        : {fail_mat_count}")
    print(f"    Sampel masuk DataLoader     : {len(records)}")

    # Save manifest
    os.makedirs("outputs/reports", exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(manifest_path, index=False)
    # Juga simpan ke outputs/reports untuk keperluan laporan
    os.makedirs("outputs/reports", exist_ok=True)
    df.to_csv("outputs/reports/train_manifest.csv", index=False)
    
    return records

def scan_aflw2000(root: str):
    manifest_path = os.path.join(root, "_cache_manifest.csv")
    if os.path.exists(manifest_path):
        print(f"\n  [AFLW2000-3D] Memuat dataset secara instan dari cache: {manifest_path}")
        df = pd.read_csv(manifest_path)
        initial_count = len(df)
        df = df[(df['pitch'].abs() <= 99) & (df['yaw'].abs() <= 99) & (df['roll'].abs() <= 99)]
        filtered_count = len(df)
        if filtered_count < initial_count:
            print(f"  [WARNING] Menghapus {initial_count - filtered_count} sampel AFLW2000-3D dari cache yang di luar rentang ±99°!")
        return df.to_dict('records')

    print("\n  [AFLW2000-3D] Memindai dataset...")
    records = []
    no_mat_count = 0
    no_img_count = 0
    fail_mat_count = 0
    img_fail_read_count = 0

    img_files = sorted(glob.glob(os.path.join(root, "*.jpg")) +
                       glob.glob(os.path.join(root, "*.png")))
    all_mats_paths = set(glob.glob(os.path.join(root, "*.mat")))

    for img_path in img_files:
        base = os.path.splitext(img_path)[0]
        mat_path = base + ".mat"

        if not os.path.isfile(mat_path):
            no_mat_count += 1
            continue

        pose = read_pose_mat(mat_path)
        if pose is None:
            fail_mat_count += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            img_fail_read_count += 1
            continue

        pitch, yaw, roll = pose

        # Filter outlier: sudut kepala manusia tidak mungkin melebihi ±99 derajat
        if abs(pitch) > 99 or abs(yaw) > 99 or abs(roll) > 99:
            fail_mat_count += 1
            continue

        records.append({
            "image_path": img_path,
            "annotation_path": mat_path,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
        })

    for mat_path in all_mats_paths:
        base = os.path.splitext(mat_path)[0]
        if not os.path.isfile(base + ".jpg") and not os.path.isfile(base + ".png"):
            no_img_count += 1

    print(f"  [AFLW2000-3D] Hasil Audit:")
    print(f"    Total gambar ditemukan      : {len(img_files)}")
    print(f"    Total anotasi .mat ditemukan: {len(all_mats_paths)}")
    print(f"    Total pasangan valid        : {len(records)}")
    print(f"    Gambar tanpa anotasi        : {no_mat_count}")
    print(f"    Anotasi tanpa gambar        : {no_img_count}")
    print(f"    Gambar gagal dibaca         : {img_fail_read_count}")
    print(f"    Anotasi gagal dibaca        : {fail_mat_count}")
    print(f"    Sampel masuk DataLoader     : {len(records)}")

    os.makedirs("outputs/reports", exist_ok=True)
    df = pd.DataFrame(records)
    try:
        df.to_csv(manifest_path, index=False)
        df.to_csv("outputs/reports/validation_manifest.csv", index=False)
    except Exception as e:
        print(f"  [WARNING] Gagal menyimpan manifest ke CSV (mungkin file sedang dibuka di Excel?): {e}")

    return records

def inspect_samples(records: list, dataset_name: str, n: int = 5):
    print(f"\n  [Audit Visual - {dataset_name}] (menampilkan {min(n, len(records))} sampel)")
    print(f"  {'Nama file':<35} {'pitch':>8} {'yaw':>8} {'roll':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for r in records[:n]:
        name = os.path.basename(r["image_path"])
        print(f"  {name:<35} {r['pitch']:>8.2f} {r['yaw']:>8.2f} {r['roll']:>8.2f}")

class W300LP_Dataset(Dataset):
    """300W-LP dataset with explicit train/validation modes.

    Args:
        records (list): List of dicts with image_path, pitch, yaw, roll.
        augment (bool): If True, apply training augmentation (ColorJitter +
            random horizontal flip with label correction). If False, apply
            deterministic val transform (resize + normalize only).
        transform: Deprecated — ignored. Use augment flag instead.
    """
    def __init__(self, records: list, augment: bool = True, transform=None):
        self.records = records
        self.augment = augment
        # augment=True  -> training pipeline: ColorJitter + random hflip
        # augment=False -> validation pipeline: deterministic resize+normalize
        self.transform = get_train_transform() if augment else get_val_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)

        # Horizontal flip with label correction (training only).
        # - Yaw  (left-right pan):  sign flipped
        # - Roll (left-right tilt): sign flipped
        # - Pitch (up-down nod):    NOT flipped
        do_flip = (self.augment and random.random() < 0.5)
        if do_flip:
            pil = T.functional.hflip(pil)

        tensor = self.transform(pil)

        pitch = float(rec["pitch"])
        yaw   = float(rec["yaw"])
        roll  = float(rec["roll"])
        if do_flip:
            yaw  = -yaw
            roll = -roll

        label = torch.tensor([pitch, yaw, roll], dtype=torch.float32)
        return tensor, label

class AFLW2000_Dataset(Dataset):
    def __init__(self, records: list, transform=None):
        self.records = records
        self.transform = transform or get_val_transform()
        self.skipped_count = 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec["image_path"])
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bbox = landmarks_to_bbox(rec["annotation_path"], padding_ratio=0.20)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            face = img[y1:y2, x1:x2]
        else:
            face = img
            self.skipped_count += 1

        pil = Image.fromarray(face)
        tensor = self.transform(pil)
        label = torch.tensor([rec["pitch"], rec["yaw"], rec["roll"]], dtype=torch.float32)
        return tensor, label
