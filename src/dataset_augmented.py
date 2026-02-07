"""
Augmented dataset for Gardner embryo grading with selective class-balanced augmentation.

This module provides GardnerDatasetAugmented which supports:
- Configurable augmentation factors per class for minority class oversampling
- Heavy augmentation for replicated samples
- Standard augmentation for original samples
- Proper handling of "Not Determined" (ND) labels for TE task
"""

import os
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Handle imports for both module and script usage
try:
    from .augmentations import get_train_transforms, get_val_transforms, get_heavy_augmentation
    from .utils import normalize_exp_token, normalize_icm_te_token
except ImportError:
    # When running as script, imports will be handled in __main__
    pass


def parse_norm_label(label) -> int:
    """Parse and normalize label to integer."""
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


class GardnerDatasetAugmented(Dataset):
    """
    Gardner dataset with Albumentations-based augmentation and selective class balancing.
    
    Features:
    - Task-specific augmentation pipelines (ICM/TE vs EXP)
    - Selective augmentation: configurable replication factors per class
    - Heavy augmentation for minority class replicas
    - Standard augmentation for original samples
    - Proper ND (Not Determined) handling for TE task
    
    Args:
        csv_path: Path to CSV file with image names and labels
        img_dir: Directory containing images
        task: One of 'icm', 'te', or 'exp'
        split: One of 'train', 'val', or 'test'
        selective_augmentation: If True, apply class-based replication
        augment_factors: Dict mapping class label to replication factor
                        Example: {0: 1, 1: 2, 2: 5} means class 2 gets 5× samples
        img_size: Target image size (default: 224)
    """
    
    def __init__(
        self,
        csv_path,
        img_dir,
        task="te",
        split="train",
        selective_augmentation=False,
        augment_factors=None,
        img_size=224,
    ):
        self.img_dir = img_dir
        self.task = task.lower()
        self.split = split.lower()
        self.selective_augmentation = selective_augmentation
        self.augment_factors = augment_factors or {}
        self.img_size = img_size
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Basic column checks
        for col in ["Image", "EXP", "ICM", "TE"]:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}. Got columns={df.columns.tolist()}")
        
        df["Image"] = df["Image"].astype(str).str.strip()
        
        # Normalize labels based on task
        if self.task == "exp":
            df["norm_label"] = df["EXP"].apply(normalize_exp_token)
        elif self.task in {"icm", "te"}:
            df["norm_label"] = df[self.task.upper()].apply(normalize_icm_te_token)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # Keep raw tokens for logging
        df["label_raw"] = df["norm_label"]
        
        # Filtering based on split and task
        if self.split in {"train", "val"}:
            if self.task == "exp":
                valid = {"0", "1", "2", "3", "4"}
                df = df[df["norm_label"].isin(valid)].copy()
            elif self.task == "icm":
                df = df[df["norm_label"].isin({"0", "1", "2"})].copy()
            else:  # te
                df = df[df["norm_label"].isin({"0", "1", "2", "ND"})].copy()
                # Map ND to 3 (ignore_index) for TE task
                df.loc[df["norm_label"] == "ND", "norm_label"] = "3"
        elif self.split == "test":
            pass  # Keep all for test
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        df = df.reset_index(drop=True)
        self.df = df
        
        # Parse integer labels
        if self.task == "exp":
            self.labels = [parse_norm_label(lbl) for lbl in df["norm_label"].tolist()]
        else:
            mapping = {"0": 0, "1": 1, "2": 2, "3": 3}
            self.labels = [mapping.get(tok, -1) for tok in df["norm_label"].astype(str).tolist()]
        
        # Print original class distribution
        if self.split == "train":
            original_dist = Counter(self.labels)
            print(f"\n[DATASET] Original {self.task.upper()} {self.split} class distribution:")
            for cls in sorted(original_dist.keys()):
                if cls >= 0:  # Skip invalid labels
                    print(f"  Class {cls}: {original_dist[cls]} samples")
        
        # Build sample list with replication
        self.samples = self._build_samples()
        
        # Print augmented class distribution
        if self.split == "train" and self.selective_augmentation:
            augmented_dist = Counter([s['label'] for s in self.samples])
            print(f"\n[DATASET] Augmented {self.task.upper()} {self.split} class distribution:")
            for cls in sorted(augmented_dist.keys()):
                if cls >= 0:  # Skip invalid labels
                    print(f"  Class {cls}: {augmented_dist[cls]} samples "
                          f"({augmented_dist[cls] / original_dist[cls]:.1f}× from original)")
        
        # Setup transforms
        if self.split == "train":
            self.transform = get_train_transforms(task=self.task, img_size=self.img_size)
            self.heavy_transform = get_heavy_augmentation(task=self.task, img_size=self.img_size)
            print(f"[DATASET] Using Albumentations train transforms for {self.task.upper()}")
        else:
            self.transform = get_val_transforms(img_size=self.img_size)
            self.heavy_transform = None
            print(f"[DATASET] Using Albumentations val/test transforms")
    
    def _build_samples(self):
        """Build sample list with optional class-based replication."""
        samples = []
        
        for idx, row in self.df.iterrows():
            img_name = row["Image"]
            label = self.labels[idx]
            label_raw = row["label_raw"]
            
            # Skip invalid labels
            if label < 0:
                continue
            
            # Determine replication factor
            if self.split == "train" and self.selective_augmentation:
                factor = self.augment_factors.get(label, 1)
            else:
                factor = 1
            
            # Add original + replicas
            for aug_version in range(factor):
                samples.append({
                    'img_name': img_name,
                    'label': label,
                    'label_raw': label_raw,
                    'aug_version': aug_version,  # 0 = original, >0 = replica
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = sample['img_name']
        label = sample['label']
        label_raw = sample['label_raw']
        aug_version = sample['aug_version']
        
        # Load image
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        # Apply appropriate transform based on augmentation version
        if self.split == "train" and aug_version > 0 and self.heavy_transform is not None:
            # Replica: use heavy augmentation
            transformed = self.heavy_transform(image=image_np)
            image_tensor = transformed['image']
        else:
            # Original or val/test: use standard transform
            transformed = self.transform(image=image_np)
            image_tensor = transformed['image']
        
        # Return image, label, label_raw, img_name (matching original dataset API)
        return image_tensor, torch.tensor(label, dtype=torch.long), label_raw, img_name


# Test script
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent to path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    # Import with proper path handling
    from src.augmentations import get_train_transforms, get_val_transforms, get_heavy_augmentation
    from src.utils import normalize_exp_token, normalize_icm_te_token
    
    print("Testing GardnerDatasetAugmented...")
    
    # Check if splits exist
    train_csv = parent_dir / "splits" / "train.csv"
    img_dir = parent_dir / "data" / "blastocyst_Dataset" / "Images"
    
    if not train_csv.exists():
        print(f"Warning: {train_csv} not found. Cannot test with real data.")
        print("Test script requires splits/train.csv and data/blastocyst_Dataset/Images/")
        sys.exit(0)
    
    # Test 1: Basic dataset without selective augmentation
    print("\n=== Test 1: Basic dataset (no selective augmentation) ===")
    dataset = GardnerDatasetAugmented(
        csv_path=str(train_csv),
        img_dir=str(img_dir),
        task="te",
        split="train",
        selective_augmentation=False,
    )
    print(f"Dataset length: {len(dataset)}")
    img, label, label_raw, img_name = dataset[0]
    print(f"Sample 0: img.shape={img.shape}, label={label.item()}, img_name={img_name}")
    
    # Test 2: Dataset with selective augmentation (5× for class 2)
    print("\n=== Test 2: Selective augmentation (5× for class 2) ===")
    dataset_aug = GardnerDatasetAugmented(
        csv_path=str(train_csv),
        img_dir=str(img_dir),
        task="te",
        split="train",
        selective_augmentation=True,
        augment_factors={0: 1, 1: 2, 2: 5},
    )
    print(f"Augmented dataset length: {len(dataset_aug)}")
    
    # Test 3: Validation dataset (should have no augmentation)
    val_csv = parent_dir / "splits" / "val.csv"
    if val_csv.exists():
        print("\n=== Test 3: Validation dataset ===")
        dataset_val = GardnerDatasetAugmented(
            csv_path=str(val_csv),
            img_dir=str(img_dir),
            task="te",
            split="val",
        )
        print(f"Val dataset length: {len(dataset_val)}")
        img, label, label_raw, img_name = dataset_val[0]
        print(f"Sample 0: img.shape={img.shape}, label={label.item()}, img_name={img_name}")
    
    print("\n✓ GardnerDatasetAugmented tests passed!")
