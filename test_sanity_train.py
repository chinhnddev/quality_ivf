#!/usr/bin/env python3
"""Quick test of sanity training loop."""

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
    sanity_mode=True,
)

# Tiny subset
tiny_indices = list(range(min(50, len(ds_train))))
tiny_train = Subset(ds_train, tiny_indices)
tiny_loader = DataLoader(tiny_train, batch_size=8, shuffle=True, num_workers=0)

# Create model and optimizer
model = IVF_EffiMorphPP(num_classes=num_classes, dropout_p=0.0)
model.to(device)

# Loss function
train_labels = [int(ds_train.df[label_col].iloc[i]) for i in range(len(ds_train.df))]
loss_fn = make_loss_fn(track="benchmark_fair", task="exp", num_classes=num_classes,
                       use_class_weights=False, train_labels=train_labels, label_smoothing=0.0)

# Optimizer
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

print(f"Model: {model.__class__.__name__}")
print(f"Loss: {loss_fn.__class__.__name__}")
print(f"Optimizer: Adam(lr=1e-3, wd=0.0)")
print(f"Tiny dataset: {len(tiny_train)} samples")

# Test single batch
print("\n[Testing single batch]")
model.train()
batch = next(iter(tiny_loader))
print(f"Batch length: {len(batch)}")

x, y, _ = batch
print(f"x shape: {x.shape}, dtype: {x.dtype}")
print(f"y shape: {y.shape}, dtype: {y.dtype}, values: {y[:3]}")

x = x.to(device)
y = y.to(device)
print(f"x on device: {x.device}")
print(f"y on device: {y.device}")

# Forward pass
print("\n[Forward pass]")
logits = model(x)
print(f"Logits shape: {logits.shape}")
print(f"Logits mean: {logits.mean():.4f}, std: {logits.std():.4f}")

# Loss
print("\n[Loss computation]")
loss = loss_fn(logits, y)
print(f"Loss value: {loss.item():.4f}")

# Backward pass
print("\n[Backward pass]")
opt.zero_grad()
try:
    loss.backward()
    print("Backward pass successful")
except Exception as e:
    print(f"Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Optimizer step
print("\n[Optimizer step]")
opt.step()
print("Optimizer step successful")

print("\n[Training one epoch]")
model.train()
total_loss = 0.0
correct = 0
total = 0
for batch_idx, batch in enumerate(tiny_loader):
    x, y, _ = batch
    x = x.to(device)
    y = y.to(device)
    
    opt.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    
    total_loss += loss.item() * x.size(0)
    preds = logits.argmax(dim=1)
    correct += (preds == y).sum().item()
    total += x.size(0)
    
    if batch_idx % 2 == 0:
        print(f"  Batch {batch_idx:2d}: loss={loss.item():.4f}, acc={correct/total:.4f}")

avg_loss = total_loss / total
avg_acc = correct / total
print(f"\nEpoch average: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

print("\nTest completed successfully!")
