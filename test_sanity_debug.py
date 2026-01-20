#!/usr/bin/env python3
"""Quick test to debug sanity overfit test."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

# Import the functions and classes
from scripts.train_gardner_single import GardnerDataset, load_and_merge_cfg

# Load config
cfg = load_and_merge_cfg(
    "configs/gardner/base.yaml",
    "configs/gardner/tasks/exp.yaml",
    "configs/gardner/tracks/benchmark_fair.yaml"
)

# Create dataset with sanity_mode=True
train_csv = Path("splits/train.csv")
images_root = Path("data/blastocyst_Dataset/Images")
task = "exp"
label_col = "EXP"

ds_train = GardnerDataset(
    csv_path=train_csv,
    images_root=images_root,
    task=task,
    split="train",
    image_col="Image",
    label_col=label_col,
    image_size=224,
    augmentation_cfg={},
    sanity_mode=True,  # <-- Key: sanity mode
)

print(f"Dataset size: {len(ds_train)}")

# Try to load a tiny subset
tiny_indices = list(range(min(10, len(ds_train))))
tiny_train = Subset(ds_train, tiny_indices)
tiny_loader = DataLoader(tiny_train, batch_size=2, shuffle=False, num_workers=0)

print(f"Tiny subset size: {len(tiny_train)}")
print("\nTesting DataLoader:")

for batch_idx, batch in enumerate(tiny_loader):
    print(f"\nBatch {batch_idx}:")
    print(f"  Batch length: {len(batch)}")
    x, y, img_name = batch
    print(f"  x shape: {x.shape}")
    print(f"  y: {y}")
    print(f"  img_name: {img_name}")
    if batch_idx >= 1:
        break

print("\nTest completed successfully!")
