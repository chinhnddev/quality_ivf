import os
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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
    def __init__(self, csv_path, img_dir, task="exp", split="train", augment=True, color_jitter=False):
        self.img_dir = img_dir
        self.task = task.lower()
        self.split = split.lower()
        self.augment = bool(augment) and self.split == "train"

        df = pd.read_csv(csv_path)

        # Basic column checks
        for col in ["Image", "EXP", "ICM", "TE"]:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}. Got columns={df.columns.tolist()}")

        df["Image"] = df["Image"].astype(str).str.strip()
        df["exp_norm_label"] = df["EXP"].apply(normalize_exp_token)
        if self.task == "exp":
            df["norm_label"] = df["EXP"].apply(normalize_exp_token)
        elif self.task in {"icm", "te"}:
            df["norm_label"] = df[self.task.upper()].apply(normalize_icm_te_token)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {self.split}")

        if self.split in {"train", "val"}:
            if self.task == "exp":
                valid = {"0", "1", "2", "3", "4"}
                df = df[df["norm_label"].isin(valid)].copy()
            else:
                before_invalid = len(df)
                invalid_counts = Counter(df["norm_label"])
                df = df[df["norm_label"].isin({"0", "1", "2"})].copy()
                print(
                    f"[FILTER] {self.task.upper()} {self.split}: kept {len(df)}/{before_invalid} after removing ND/NA/invalid labels ({dict(invalid_counts)})"
                )
                if self.task in {"icm", "te"}:
                    before_exp = len(df)
                    exp_counts = Counter(df["exp_norm_label"])
                    df = df[~df["exp_norm_label"].isin({"0", "1"})].copy()
                    after_exp = len(df)
                    dropped = before_exp - after_exp
                    print(
                        f"[FILTER] {self.task.upper()} {self.split}: removed {dropped} samples with EXP in {{0,1}} (before={before_exp}, after={after_exp})"
                    )
                    print(f"  EXP counts before final filter: {dict(exp_counts)}")
                    print(f"  Post-filter label counts: {dict(Counter(df['norm_label']))}")

        df = df.reset_index(drop=True)

        self.images = df["Image"].tolist()

        self.labels_raw = df["norm_label"].tolist()
        if self.split in {"test", "gold_test"}:
            print(f"[DATASET] {self.task.upper()} {self.split} labels_raw counter: "
                  f"{Counter(self.labels_raw)}")
        if self.task == "exp":
            self.labels = [parse_norm_label(lbl) for lbl in self.labels_raw]
        else:
            mapping = {"0": 0, "1": 1, "2": 2}
            self.labels = [mapping.get(tok, -1) for tok in self.labels_raw]

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
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = self.labels[idx]
        label_raw = self.labels_raw[idx]

        # Return filename too for preds.csv
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
