import os
import glob
import math
import logging
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224

def get_val_transform():
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to Euler angles (pitch, yaw, roll) in degrees.
    Convention matches BIWI corrected (R.T, specific atan2).
    Returns (pitch, yaw, roll)
    """
    R = R.T
    sy = math.sqrt(R[2, 1]**2 + R[2, 2]**2)
    roll = -math.atan2(R[1, 0], R[0, 0])
    yaw = -math.atan2(-R[2, 0], sy)
    pitch = math.atan2(R[2, 1], R[2, 2])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

def read_biwi_pose(pose_path):
    try:
        with open(pose_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) < 3:
            return None
        R = []
        for i in range(3):
            row = [float(x) for x in lines[i].split()]
            if len(row) < 3:
                return None
            R.append(row[:3])
        R = np.array(R, dtype=np.float64)
        return rotation_matrix_to_euler(R)
    except Exception as e:
        logger.debug(f"Failed reading BIWI pose '{pose_path}': {e}")
        return None

def scan_biwi(root: str, angle_limit: float = 99.0):
    """
    Read BIWI from precomputed manifest: biwi_preprocessing_manifest.csv
    Returns list of valid records and audit dict.
    """
    manifest_path = "biwi_preprocessing_manifest.csv"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"BIWI manifest not found. Run preprocess_biwi.py first: {manifest_path}")

    audit = {
        "dataset": "BIWI",
        "initial_candidates": 0,
        "valid_pairs": 0,  # All pairs that had pose + image
        "face_detection_success": 0,
        "face_detection_failed": 0,
        "removed_angle": 0,
        "final_evaluated_samples": 0,
        "pitch_min": float("inf"),
        "pitch_max": float("-inf"),
        "yaw_min": float("inf"),
        "yaw_max": float("-inf"),
        "roll_min": float("inf"),
        "roll_max": float("-inf"),
    }

    records = []
    import csv
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audit["initial_candidates"] += 1
            audit["valid_pairs"] += 1  # assuming manifest only contains valid pairs
            
            # Read pose to get angles and update min/max
            pose = read_biwi_pose(row["pose_path"])
            if pose is None:
                continue
            pitch, yaw, roll = pose

            # Check angles
            if abs(pitch) > angle_limit or abs(yaw) > angle_limit or abs(roll) > angle_limit:
                audit["removed_angle"] += 1
                continue

            if row["face_detected"] == "True":
                audit["face_detection_success"] += 1
                
                audit["pitch_min"] = min(audit["pitch_min"], pitch)
                audit["pitch_max"] = max(audit["pitch_max"], pitch)
                audit["yaw_min"] = min(audit["yaw_min"], yaw)
                audit["yaw_max"] = max(audit["yaw_max"], yaw)
                audit["roll_min"] = min(audit["roll_min"], roll)
                audit["roll_max"] = max(audit["roll_max"], roll)
                
                records.append({
                    "image_path": row["image_path"],
                    "annotation_path": row["pose_path"],
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                    "crop_x1": int(float(row["crop_x1"])),
                    "crop_y1": int(float(row["crop_y1"])),
                    "crop_x2": int(float(row["crop_x2"])),
                    "crop_y2": int(float(row["crop_y2"]))
                })
            else:
                audit["face_detection_failed"] += 1

    audit["final_evaluated_samples"] = len(records)
    for k in ["pitch_min","pitch_max","yaw_min","yaw_max","roll_min","roll_max"]:
        if audit[k] in [float("inf"), float("-inf")]:
            audit[k] = None

    return records, audit

class BIWI_Dataset(Dataset):
    """PyTorch Dataset for BIWI - uses precomputed head ROI crop."""
    def __init__(self, records: list, transform=None):
        self.records = records
        self.transform = transform or get_val_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop ROI
        x1, y1 = rec["crop_x1"], rec["crop_y1"]
        x2, y2 = rec["crop_x2"], rec["crop_y2"]
        roi = img[y1:y2, x1:x2]
        
        pil = Image.fromarray(roi)
        tensor = self.transform(pil)
        label = torch.tensor(
            [rec["pitch"], rec["yaw"], rec["roll"]], dtype=torch.float32
        )
        return tensor, label

