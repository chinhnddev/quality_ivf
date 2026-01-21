import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def _norm_token(x) -> str:
    """Normalize to one of: '0','1','2','ND','NA',''."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "":
        return ""
    su = s.upper()
    if su in {"ND", "NA"}:
        return su
    if s in {"0", "1", "2", "3", "4"}:
        return s
    return ""


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

        # Normalize
        df["Image"] = df["Image"].astype(str).str.strip()
        df["EXP"] = pd.to_numeric(df["EXP"], errors="coerce")  # EXP should be numeric codes 0..4
        df["ICM"] = df["ICM"].apply(_norm_token)
        df["TE"]  = df["TE"].apply(_norm_token)

        # Filtering rules
        if self.split in {"train", "val"}:
            if self.task == "exp":
                df = df[df["EXP"].isin([0, 1, 2, 3, 4])].copy()
            elif self.task == "icm":
                df = df[df["ICM"].isin(["0", "1", "2"])].copy()
            elif self.task == "te":
                df = df[df["TE"].isin(["0", "1", "2"])].copy()
            else:
                raise ValueError(f"Unknown task: {self.task}")
        elif self.split == "test":
            # No filtering; metrics will mask invalid labels for ICM/TE
            pass
        else:
            raise ValueError(f"Unknown split: {self.split}")

        df = df.reset_index(drop=True)

        self.images = df["Image"].tolist()

        # Build labels
        if self.task == "exp":
            # For test we still expect EXP valid for all 300; but keep safe.
            # Invalid -> -1
            self.labels_raw = df["EXP"].tolist()
            self.labels = [int(x) if pd.notna(x) and int(x) in [0,1,2,3,4] else -1 for x in df["EXP"]]
        elif self.task == "icm":
            self.labels_raw = df["ICM"].tolist()
            mapping = {"0": 0, "1": 1, "2": 2}
            self.labels = [mapping.get(tok, -1) for tok in self.labels_raw]  # -1 for ND/NA/''
        elif self.task == "te":
            self.labels_raw = df["TE"].tolist()
            mapping = {"0": 0, "1": 1, "2": 2}
            self.labels = [mapping.get(tok, -1) for tok in self.labels_raw]
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Transforms
        base = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if self.augment:
            aug = [
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
            if color_jitter:
                aug.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
            self.transform = transforms.Compose(aug + base)
        else:
            self.transform = transforms.Compose(base)

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
        base = [
            transforms.Resize((224, 224)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        
        if self.augment:
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ]
            self.transform = transforms.Compose(aug + base)
        else:
            self.transform = transforms.Compose(base)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_base_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.stage_labels[idx], dtype=torch.long)
        
        return image, label