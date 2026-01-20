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
