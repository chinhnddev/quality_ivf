import os
import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Optional

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import normalize_exp_token, normalize_icm_te_token


def parse_norm_label(label) -> int:
    if pd.isna(label):
        return -1
    if isinstance(label, str):
        txt = label.strip()
        if txt == "" or txt.upper() in {"NA", "ND"}:
            return -1
        try:
            return int(txt)
        except ValueError:
            try:
                return int(float(txt))
            except ValueError:
                return -1
    try:
        return int(label)
    except Exception:
        try:
            return int(float(label))
        except Exception:
            return -1


class GardnerDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        task: str = "exp",
        split: str = "train",
        image_col: str = "Image",
        label_col: str = "EXP",
        image_size: int = 224,
        augment: bool = True,
        augmentation_cfg: Optional[dict] = None,
        sanity_mode: bool = False,
        color_jitter: bool = False,
    ):
        self.img_dir = img_dir
        self.task = task.lower()
        self.split = split.lower()
        self.image_col = image_col
        self.label_col = label_col
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.augmentation_cfg = augmentation_cfg or {}
        self.sanity_mode = bool(sanity_mode)
        self.color_jitter = color_jitter

        if self.split not in {"train", "val", "test", "gold_test"}:
            raise ValueError(f"Unknown split: {self.split}")

        df = pd.read_csv(csv_path)

        required_cols = {"Image", "EXP", "ICM", "TE"}
        missing = required_cols - set(df.columns.tolist())
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Got columns={df.columns.tolist()}")

        if self.image_col not in df.columns:
            raise ValueError(f"Image column '{self.image_col}' not found in CSV.")
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in CSV.")

        df[self.image_col] = df[self.image_col].astype(str).str.strip()

        if self.task == "exp":
            df["norm_label"] = df[self.label_col].apply(normalize_exp_token)
        elif self.task in {"icm", "te"}:
            df["norm_label"] = df[self.label_col].apply(normalize_icm_te_token)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        df["label_raw"] = df["norm_label"]

        if self.split in {"train", "val"}:
            if self.task == "exp":
                valid = {"0", "1", "2", "3", "4"}
                df = df[df["norm_label"].isin(valid)].copy()
            elif self.task == "icm":
                df = df[df["norm_label"].isin({"0", "1", "2"})].copy()
            else:  # te
                df = df[df["norm_label"].isin({"0", "1", "2"})].copy()
        df = df.reset_index(drop=True)

        self.df = df
        self.images = df[self.image_col].tolist()
        self.labels_raw = df["label_raw"].tolist()
        if self.split in {"test", "gold_test"}:
            print(
                f"[DATASET] {self.task.upper()} {self.split} labels_raw counter: "
                f"{Counter(self.labels_raw)}"
            )
        if self.task == "exp":
            self.labels = [parse_norm_label(lbl) for lbl in self.labels_raw]
        else:
            mapping = {"0": 0, "1": 1, "2": 2, "3": 3}
            self.labels = [mapping.get(tok, -1) for tok in df["norm_label"].astype(str).tolist()]

        self.transform = self._build_transform()

    def _build_transform(self):
        if self.split == "train" and self.augment and not self.sanity_mode:
            return self._build_train_transform()
        return self._build_eval_transform()

    def _build_train_transform(self):
        cfg_get = self._cfg_get
        img = int(self.image_size)

        crop_scale = tuple(cfg_get("random_resized_crop_scale", (0.8, 1.0)))
        crop_ratio = tuple(cfg_get("random_resized_crop_ratio", (0.9, 1.1)))

        pipeline = [
            A.RandomResizedCrop(
                size=(img, img),
                scale=crop_scale,
                ratio=crop_ratio,
                interpolation=cv2.INTER_LINEAR,
                p=1.0,
            ),
            A.HorizontalFlip(p=float(cfg_get("hflip_p", 0.5))),
            A.VerticalFlip(p=float(cfg_get("vflip_p", 0.10))),

            # very mild geometry only
            A.Affine(
                rotate=(-5, 5),
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                scale=(0.97, 1.03),
                interpolation=cv2.INTER_LINEAR,
                p=float(cfg_get("affine_p", 0.25)),
            ),

            # exposure/contrast robustness (safe for EXP)
            A.RandomGamma(gamma_limit=(90, 110), p=float(cfg_get("gamma_p", 0.25))),
            A.RandomBrightnessContrast(
                brightness_limit=0.05,
                contrast_limit=0.08,
                p=float(cfg_get("bc_p", 0.25)),
            ),

            # CLAHE: optional, light
            A.CLAHE(clip_limit=1.8, tile_grid_size=(8, 8), p=float(cfg_get("clahe_p", 0.15))),

            # very light blur/noise
            A.GaussianBlur(blur_limit=(3, 3), p=float(cfg_get("blur_p", 0.08))),
            A.GaussNoise(std_range=(0.0, 0.02), mean_range=(0.0, 0.0), p=float(cfg_get("noise_p", 0.10))),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        return A.Compose(pipeline)

    def _build_eval_transform(self):
        img = int(self.image_size)
        resize_size = int(getattr(self, "resize_size", 256))

        return A.Compose([
            A.Resize(resize_size, resize_size, interpolation=cv2.INTER_LINEAR),
            A.CenterCrop(img, img),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    def _cfg_get(self, key, default):
        cfg = self.augmentation_cfg
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image_arr = np.asarray(image)
        image = self.transform(image=image_arr)["image"]

        label = self.labels[idx]
        label_raw = self.labels_raw[idx]
        return image, torch.tensor(label, dtype=torch.long), label_raw, img_name

# ==========================================
# HumanEmbryoStageDataset
# ==========================================
class HumanEmbryoStageDataset(Dataset):
    """
    Dataset for stage pretraining using HumanEmbryo2.0 dataset.
    
    Loads images and maps developmental stages:
    - cleavage -> 0
    - morula -> 1
    - blastocyst -> 2
    """
    
    def __init__(self, csv_path, img_base_dir, split="train", augment=True):
        """
        Args:
            csv_path: Path to humanembryo2.csv
            img_base_dir: Base directory for images (e.g., data/HumanEmbryo2.0/)
            split: 'train', 'val', or 'test'
            augment: If True and split=='train', apply augmentations
        """
        self.img_base_dir = img_base_dir
        self.split = split.lower()
        self.augment = bool(augment) and self.split == "train"
        
        # Load metadata
        df = pd.read_csv(csv_path)
        
        # Validate columns
        required_cols = ["image_path", "stage", "split"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV missing column: {col}. Got: {df.columns.tolist()}")
        
        # Filter by split
        df = df[df["split"].str.lower() == self.split].copy()
        df = df.reset_index(drop=True)
        
        # Stage mapping
        stage_map = {"cleavage": 0, "morula": 1, "blastocyst": 2}
        
        # Extract image paths and labels
        self.image_paths = df["image_path"].str.strip().tolist()
        self.stage_labels = [stage_map.get(s.lower(), -1) for s in df["stage"].astype(str)]
        
        # Filter out invalid stages
        valid_indices = [i for i, label in enumerate(self.stage_labels) if label >= 0]
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.stage_labels = [self.stage_labels[i] for i in valid_indices]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid samples found for split '{self.split}' in {csv_path}")
        
        # Transforms
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_base_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.stage_labels[idx], dtype=torch.long)
        
        return image, label
