import os
import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Optional

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
                df = df[df["norm_label"].isin({"0", "1", "2", "ND"})].copy()
                df.loc[df["norm_label"] == "ND", "norm_label"] = "3"
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
        crop_scale = tuple(cfg_get("random_resized_crop_scale", (0.85, 1.0)))  # Hẹp hơn để giữ phôi
        crop_ratio = tuple(cfg_get("random_resized_crop_ratio", (0.95, 1.05)))  # Gần 1:1 cho embryo
        rotation_limit = float(cfg_get("rotation_deg", 20.0))  # Giảm nhẹ nếu cần
        clahe_clip = float(cfg_get("clahe_clip_limit", 2.5))  # Giảm để tránh over-enhance noise

        pipeline = [
            A.RandomResizedCrop(
                height=self.image_size,
                width=self.image_size,
                scale=crop_scale,
                ratio=crop_ratio,
                interpolation=A.Interpolation.BILINEAR,
                antialias=True,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=rotation_limit, p=0.7, interpolation=A.Interpolation.BILINEAR),
            A.ShiftScaleRotate(  # Thêm nhẹ để simulate misalignment
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5,
                interpolation=A.Interpolation.BILINEAR,
            ),
            A.CLAHE(clip_limit=clahe_clip, p=0.7),  # Enhance ICM/TE cell contrast
            A.GaussNoise(var_limit=(5, 20), p=0.3),  # Microscope noise nhẹ
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.CoarseDropout(  # RandomErasing-like, che small regions
                max_holes=4,
                max_height=int(self.image_size * 0.08),
                max_width=int(self.image_size * 0.08),
                fill_value=0,
                p=0.4,
            ),
            # Optional: light color nếu ảnh có variation
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3) if self.color_jitter else A.NoOp(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        return A.Compose(pipeline)

    def _build_eval_transform(self):
        return A.Compose([
            A.Resize(self.image_size, self.image_size, interpolation=A.Interpolation.BILINEAR),
            A.CenterCrop(self.image_size, self.image_size),
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
        if self.split == "train" and self.augment and not self.sanity_mode:
            image = self.transform(image=np.asarray(image))["image"]
        else:
            image = self.transform(image)

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
