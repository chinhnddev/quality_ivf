#!/usr/bin/env python3
"""Test the actual sanity loop with tiny dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from omegaconf import OmegaConf

# Import the functions and classes
from scripts.train_gardner_single import GardnerDataset, load_and_merge_cfg, make_loss_fn
from src.model import IVF_EffiMorphPP

# Load config
cfg = load_and_merge_cfg(
    "configs/gardner/base.yaml",
    "configs/gardner/tasks/exp.yaml",
    "configs/gardner/tracks/benchmark_fair.yaml"
)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create dataset with sanity_mode=True
train_csv = Path("splits/train.csv")
images_root = Path("data/blastocyst_Dataset/Images")
task = "exp"
label_col = "EXP"
num_classes = 5

ds_train = GardnerDataset(
    csv_path=train_csv,
    images_root=images_root,
    task=task,
    split="train",
    image_col="Image",
    label_col=label_col,
    image_size=224,
    augmentation_cfg={},
    sanity_mode=True,  # <-- CRITICAL
)

print(f"Dataset size: {len(ds_train)}")

# Tiny subset
tiny_indices = list(range(min(200, len(ds_train))))
tiny_train = Subset(ds_train, tiny_indices)
tiny_loader = DataLoader(tiny_train, batch_size=32, shuffle=True, num_workers=0)

print(f"Tiny subset size: {len(tiny_train)}")

# Create model and optimizer
model = IVF_EffiMorphPP(num_classes=num_classes, dropout_p=0.0)
model.to(device)

# Loss function
train_labels = [int(ds_train.df[label_col].iloc[i]) for i in range(len(ds_train.df))]
loss_fn = make_loss_fn(track="benchmark_fair", task="exp", num_classes=num_classes,
                       use_class_weights=False, train_labels=train_labels)

# Optimizer
sanity_opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

print(f"\nTraining...")
model.train()
final_train_acc = 0.0
for sanity_epoch in range(1, 4):  # Just 3 epochs for testing
    sanity_loss = 0.0
    sanity_correct = 0
    sanity_total = 0
    
    print(f"[Epoch {sanity_epoch}] Starting training loop...")
    for batch_idx, batch in enumerate(tiny_loader):
        if len(batch) == 4:
            x, y, _, _ = batch
        else:
            x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        sanity_opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        sanity_opt.step()
        
        sanity_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        sanity_correct += (preds == y).sum().item()
        sanity_total += x.size(0)
    
    sanity_acc = sanity_correct / sanity_total
    sanity_loss /= sanity_total
    print(f"  Epoch {sanity_epoch:2d}: loss={sanity_loss:.4f} | train_acc={sanity_acc:.4f}")

print("\nTest completed successfully!")
