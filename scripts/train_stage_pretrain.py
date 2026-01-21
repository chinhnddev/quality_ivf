#!/usr/bin/env python3
"""
Stage Pretraining Script for IVF_EffiMorphPP Backbone

Pretrains the backbone on developmental stage classification (cleavage, morula, blastocyst).
This pretrained backbone can then be used for downstream tasks (EXP, ICM, TE).

Usage:
    python scripts/train_stage_pretrain.py \
        --metadata_csv data/metadata/humanembryo2.csv \
        --img_base_dir data/HumanEmbryo2.0 \
        --out_dir outputs/pretrain_stage \
        --epochs 40 \
        --lr 3e-4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import inspect
from src.model import IVF_EffiMorphPP

# [DEBUG] Verify model location
print("[DEBUG] IVF_EffiMorphPP imported from:", inspect.getfile(IVF_EffiMorphPP))


# =============================================
# Utilities
# =============================================

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true: list, y_pred: list) -> Dict[str, float]:
    """Compute accuracy and macro-F1 score."""
    if not y_true:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class_precision": [],
            "per_class_recall": [],
            "per_class_f1": [],
            "per_class_support": [],
        }
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": precision.tolist() if precision is not None else [],
        "per_class_recall": recall.tolist() if recall is not None else [],
        "per_class_f1": f1.tolist() if f1 is not None else [],
        "per_class_support": support.tolist() if support is not None else [],
    }


# =============================================
# Dataset
# =============================================

class StageDataset(Dataset):
    """Simple dataset for stage classification."""
    
    def __init__(self, image_paths: list, stage_labels: list, img_base_dir: str, transform):
        self.image_paths = image_paths
        self.stage_labels = stage_labels
        self.img_base_dir = img_base_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_base_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.stage_labels[idx], dtype=torch.long)
        return image, label


def create_dataset_from_df(df: pd.DataFrame, img_base_dir: str, augment: bool = False):
    """Create dataset from dataframe."""
    stage_map = {"cleavage": 0, "morula": 1, "blastocyst": 2}
    
    # Extract image paths and labels
    image_paths = df["image_path"].str.strip().tolist()
    stage_labels = [stage_map.get(s.lower(), -1) for s in df["stage"].astype(str)]
    
    # Filter out invalid stages
    valid_indices = [i for i, label in enumerate(stage_labels) if label >= 0]
    image_paths = [image_paths[i] for i in valid_indices]
    stage_labels = [stage_labels[i] for i in valid_indices]
    
    if len(image_paths) == 0:
        raise ValueError("No valid samples in dataset")
    
    # Transforms
    base = [
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    if augment:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ]
        transform = transforms.Compose(aug + base)
    else:
        transform = transforms.Compose(base)
    
    return StageDataset(image_paths, stage_labels, img_base_dir, transform)


# =============================================
# Training & Validation
# =============================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(images)
        loss = loss_fn(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1).cpu().numpy().tolist()
        y_pred.extend(predictions)
        y_true.extend(labels.cpu().numpy().tolist())
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = accuracy_score(y_true, y_pred)
    
    return avg_loss, avg_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Validate. Returns average loss, accuracy, and macro-F1."""
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.cpu().numpy().tolist())
    
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(y_true, y_pred)
    
    return avg_loss, metrics["accuracy"], metrics["macro_f1"]


# =============================================
# Main Training Loop
# =============================================

def main(args):
    """Main training function."""
    
    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    
    print(f"\n{'='*80}")
    print(f"Stage Pretraining: IVF_EffiMorphPP Backbone")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Output dir: {out_dir}")
    
    # -----------------------------------------------
    # 1. Load metadata and create train/val split
    # -----------------------------------------------
    print(f"\n[1] Loading metadata and creating train/val split...")
    meta_df = pd.read_csv(args.metadata_csv)
    print(f"Total samples in metadata: {len(meta_df)}")
    
    # Filter to train set only, then split into train/val
    train_meta = meta_df[meta_df["split"] == "train"].copy()
    print(f"Train samples available: {len(train_meta)}")
    
    # Create train/val split (80/20)
    np.random.seed(args.seed)
    indices = np.arange(len(train_meta))
    np.random.shuffle(indices)
    val_size = max(1, int(0.2 * len(train_meta)))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_meta_split = train_meta.iloc[train_indices].reset_index(drop=True)
    val_meta_split = train_meta.iloc[val_indices].reset_index(drop=True)
    
    print(f"  Train split: {len(train_meta_split)}")
    print(f"  Val split: {len(val_meta_split)}")
    
    # -----------------------------------------------
    # 2. Create datasets
    # -----------------------------------------------
    print(f"\n[2] Creating datasets...")
    try:
        ds_train = create_dataset_from_df(train_meta_split, args.img_base_dir, augment=True)
        ds_val = create_dataset_from_df(val_meta_split, args.img_base_dir, augment=False)
    except Exception as e:
        print(f"ERROR creating datasets: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"  Train dataset: {len(ds_train)}")
    print(f"  Val dataset: {len(ds_val)}")
    
    # -----------------------------------------------
    # 3. Create dataloaders
    # -----------------------------------------------
    print(f"\n[3] Creating dataloaders...")
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  Train batches: {len(dl_train)} ({args.batch_size} samples/batch)")
    print(f"  Val batches: {len(dl_val)}")
    
    # -----------------------------------------------
    # 4. Create model
    # -----------------------------------------------
    print(f"\n[4] Creating model...")
    model = IVF_EffiMorphPP(
        num_classes=3,  # cleavage, morula, blastocyst
        dropout_p=args.dropout,
        task="stage",
        use_coral=False,  # No CORAL for stage classification
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model class: {model.__class__.__name__}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # [DEBUG] Verify backbone size
    if total_params < 2_000_000:
        print(f"\n[WARNING] Model seems lightweight (~{total_params/1e6:.1f}M params)")
        print(f"  Expected backbone: ~8-9M params")
        print(f"  Suggest using width_mult=2.0 or checking model architecture")
        print(f"  Continuing with current model...\n")
    
    # -----------------------------------------------
    # 5. Setup training
    # -----------------------------------------------
    print(f"\n[5] Setting up training...")
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(f"  Optimizer: AdamW")
    print(f"  LR: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Epochs: {args.epochs}")
    
    # -----------------------------------------------
    # 6. Training loop
    # -----------------------------------------------
    print(f"\n[6] Starting training...\n")
    
    best_val_metric = -1.0
    best_epoch = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, dl_train, loss_fn, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_macro_f1 = validate(
            model, dl_val, loss_fn, device
        )
        
        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}")
        
        # Save best checkpoint (using val accuracy as metric)
        if val_acc > best_val_metric:
            best_val_metric = val_acc
            best_epoch = epoch
            best_path = out_dir / "best.ckpt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
            }, best_path)
            print(f"  ✓ Saved best checkpoint: {best_path}")
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best epoch: {best_epoch} (Val Acc: {best_val_metric:.4f})")
    print(f"Best checkpoint: {out_dir / 'best.ckpt'}")
    print(f"{'='*80}\n")
    
    # -----------------------------------------------
    # 7. Save training history
    # -----------------------------------------------
    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history: {history_path}\n")
    
    # -----------------------------------------------
    # 8. Load best model and evaluate on val set
    # -----------------------------------------------
    print(f"Loading best model and evaluating on validation set...")
    best_path = out_dir / "best.ckpt"
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in dl_val:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.cpu().numpy().tolist())
    
    metrics = compute_metrics(y_true, y_pred)
    metrics_path = out_dir / "metrics_val.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Validation metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1: {metrics['weighted_f1']:.4f}")
    print(f"  Per-class F1: {metrics['per_class_f1']}")
    print(f"Saved metrics: {metrics_path}\n")


# =============================================
# CLI
# =============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage pretraining for IVF_EffiMorphPP backbone"
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="data/metadata/humanembryo2.csv",
        help="Path to HumanEmbryo2.0 metadata CSV",
    )
    parser.add_argument(
        "--img_base_dir",
        type=str,
        default="data/HumanEmbryo2.0",
        help="Base directory for HumanEmbryo2.0 images",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/pretrain_stage",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    main(args)
