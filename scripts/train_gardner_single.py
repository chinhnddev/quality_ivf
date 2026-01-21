#!/usr/bin/env python3
# scripts/train_gardner_single.py

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

try:
    from torchinfo import summary as torchinfo_summary
except ImportError:
    torchinfo_summary = None

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))


# =========================
# Utils
# =========================

def set_seed(seed: int, deterministic: bool = True) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_token(x) -> str:
    """Normalize label tokens for ICM/TE to one of: '0','1','2','ND','NA',''."""
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if float(x).is_integer():
            return str(int(x))
        return str(x)
    s = str(x).strip()
    if s == "":
        return ""
    s_up = s.upper()
    if s_up in {"ND", "NA"}:
        return s_up
    # allow numeric strings
    if s in {"0", "1", "2", "3", "4"}:
        return s
    return s  # fallback


def compute_class_weights(labels: List[int], num_classes: int, eps: float = 1e-8) -> torch.Tensor:
    """Inverse-frequency normalized weights: w_c = N_total / (K * N_c)."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for y in labels:
        if 0 <= y < num_classes:
            counts[y] += 1
    total = counts.sum()
    K = float(num_classes)
    weights = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (K * (counts[c] + eps))
        else:
            # missing class: weight 0, warn upstream
            weights[c] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def make_loss_fn(track: str, task: str, num_classes: int, use_class_weights: bool, train_labels: List[int], sanity_mode: bool = False, use_weighted_sampler: bool = False) -> nn.Module:
    weights = None
    if use_class_weights:
        weights = compute_class_weights(train_labels, num_classes)
        # Warning if any missing class
        if (weights == 0).any():
            missing = [i for i, w in enumerate(weights.tolist()) if w == 0.0]
            print(f"[WARN] Missing classes in TRAIN for task={task}: {missing}. Their weight is set to 0.")
        else:
            print(f"[OK] Class weights computed for task={task}:")
            for i, w in enumerate(weights.tolist()):
                print(f"  Class {i}: weight={w:.4f}")
    if track == "benchmark_fair":
        return nn.CrossEntropyLoss(weight=weights)
    if track == "improved":
        if task == "exp":
            # label smoothing for EXP (disable in sanity mode or when using weighted sampler)
            if use_weighted_sampler:
                label_smoothing = 0.0
            else:
                label_smoothing = 0.0 if sanity_mode else 0.1
            return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        # focal for ICM/TE
        return FocalLoss(gamma=2.0, weight=weights)
    raise ValueError(f"Unknown track: {track}")


# =========================
# Dataset
# =========================

class GardnerDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        images_root: Path,
        task: str,
        split: str,
        image_col: str = "Image",
        label_col: str = "EXP",
        image_size: int = 224,
        augmentation_cfg: Optional[dict] = None,
        sanity_mode: bool = False,
    ):
        self.csv_path = csv_path
        self.images_root = images_root
        self.task = task
        self.split = split
        self.image_col = image_col
        self.label_col = label_col
        self.image_size = image_size
        self.sanity_mode = sanity_mode

        df = pd.read_csv(csv_path)
        if image_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"CSV missing required columns. Need {image_col} and {label_col}. Got: {df.columns.tolist()}")

        # Normalize EXP to int where possible; normalize ICM/TE tokens
        if label_col == "EXP":
            df["EXP"] = pd.to_numeric(df["EXP"], errors="raise").astype(int)
        else:
            df[label_col] = df[label_col].apply(normalize_token)

        # Filtering rule:
        # - train/val: filter for ICM/TE valid labels {0,1,2}; EXP keeps all
        # - test: do NOT filter (load all); masking is for eval script
        if split in {"train", "val"}:
            if label_col == "EXP":
                df = df[df["EXP"].isin([0, 1, 2, 3, 4])].copy()
            else:
                df = df[df[label_col].isin(["0", "1", "2"])].copy()
        elif split == "test":
            pass
        else:
            raise ValueError(f"Unknown split: {split}")

        self.df = df.reset_index(drop=True)

        # Transforms
        self.transform = self._build_transform(augmentation_cfg, sanity_mode)

    def _build_transform(self, aug: Optional[dict], sanity_mode: bool = False):
        size = self.image_size
        if self.split == "train" and not sanity_mode:
            # Default augmentation per your base.yaml (ONLY if not sanity_mode)
            rotation = (aug or {}).get("rotation_deg", 10)
            hflip = (aug or {}).get("horizontal_flip", True)
            vflip = (aug or {}).get("vertical_flip", True)
            rrc = (aug or {}).get("random_resized_crop", True)
            color_jitter = (aug or {}).get("color_jitter", False)

            t: List[transforms.Transform] = []
            if rrc:
                t.append(transforms.RandomResizedCrop(size))
            else:
                t.append(transforms.Resize((size, size)))
            if rotation and rotation > 0:
                t.append(transforms.RandomRotation(degrees=rotation))
            if hflip:
                t.append(transforms.RandomHorizontalFlip())
            if vflip:
                t.append(transforms.RandomVerticalFlip())
            if color_jitter:
                # keep minimal; microscope images are sensitive
                t.append(transforms.ColorJitter(brightness=0.05, contrast=0.05))

            t.extend([
                transforms.ToTensor(),
                # ImageNet normalization (paper-style)
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            return transforms.Compose(t)

        # val/test OR sanity_mode: deterministic (no augmentation)
        # In sanity mode: Resize(224)+CenterCrop(224)+ToTensor+ImageNet Normalize
        if sanity_mode:
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row[self.image_col]
        img_path = self.images_root / img_name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        if self.label_col == "EXP":
            y = int(row["EXP"])
        else:
            # train/val already filtered to "0","1","2"
            y = int(row[self.label_col])

        return x, y, img_name


def print_label_summary(df: pd.DataFrame, task: str, label_col: str, split_name: str) -> None:
    print(f"\n[{split_name}] label summary task={task} label_col={label_col}")
    if label_col == "EXP":
        counts = df["EXP"].value_counts().sort_index()
        print("EXP distribution:", counts.to_dict())
    else:
        col = label_col
        tokens = df[col].apply(normalize_token)
        total = len(tokens)
        valid = tokens.isin(["0", "1", "2"]).sum()
        nd = (tokens == "ND").sum()
        na = (tokens == "NA").sum()
        empty = (tokens == "").sum()
        print(f"{col}: total={total}, valid(0/1/2)={valid}, ND={nd}, NA={na}, empty={empty}")
        if valid > 0:
            valid_counts = tokens[tokens.isin(["0", "1", "2"])].value_counts().sort_index()
            print(f"{col} valid distribution:", valid_counts.to_dict())


# =========================
# Training
# =========================

@torch.no_grad()
def evaluate_on_val(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        # Handle both 3 and 4 element tuples from dataset
        if len(batch) == 4:
            x, y, _, _ = batch
        else:
            x, y, _ = batch
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy().tolist()
        ys.extend(y.numpy().tolist())
        ps.extend(pred)
    acc = accuracy_score(ys, ps) if len(ys) else 0.0
    macro_f1 = f1_score(ys, ps, average="macro") if len(ys) else 0.0
    weighted_f1 = f1_score(ys, ps, average="weighted") if len(ys) else 0.0

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(ys, ps, average=None, zero_division=0)

    # y_pred distribution
    from collections import Counter
    y_pred_counts = Counter(ps)
    y_pred_ratios = {cls: count / len(ps) for cls, count in y_pred_counts.items()}

    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": precision.tolist() if precision is not None else [],
        "per_class_recall": recall.tolist() if recall is not None else [],
        "per_class_f1": f1.tolist() if f1 is not None else [],
        "per_class_support": support.tolist() if support is not None else [],
        "y_pred_counts": dict(sorted(y_pred_counts.items())),
        "y_pred_ratios": dict(sorted(y_pred_ratios.items()))
    }


def train_one_run(cfg, args) -> None:
    task = cfg.task
    track = cfg.track
    splits_dir = Path(cfg.splits_dir)
    out_dir = Path(cfg.out_dir) / track / task
    ensure_dir(out_dir)

    # Save resolved config
    OmegaConf.save(cfg, out_dir / "config.yaml")

    # Resolve CSV paths
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    # Image root: by default same as splits_dir parent, but prefer explicit cfg if you have it
    # You SHOULD set this in base.yaml if your images are elsewhere.
    images_root = Path(getattr(cfg, "images_root", "")) if getattr(cfg, "images_root", "") else splits_dir.parent / "images"
    if not images_root.exists():
        # fallback to splits_dir itself
        images_root = splits_dir

    # Determine label_col and num_classes from task config
    label_col = cfg.label_col
    num_classes = int(cfg.num_classes)

    # Load raw dfs for sanity check prints (not filtered)
    raw_train_df = pd.read_csv(train_csv)
    raw_val_df = pd.read_csv(val_csv)
    if cfg.sanity_checks.print_label_distribution:
        print_label_summary(raw_train_df, task, label_col, "train.csv (raw)")
        print_label_summary(raw_val_df, task, label_col, "val.csv (raw)")

    # Build datasets (train/val filtered properly)
    # If sanity test is requested, create datasets with sanity_mode=True (deterministic transforms)
    sanity_mode = bool(args.sanity_overfit)
    ds_train = GardnerDataset(
        csv_path=train_csv,
        images_root=images_root,
        task=task,
        split="train",
        image_col=cfg.data.image_col if "data" in cfg else "Image",
        label_col=label_col,
        image_size=int(cfg.data.image_size) if "data" in cfg else int(cfg.image_size),
        augmentation_cfg=cfg.augmentation if "augmentation" in cfg else {},
        sanity_mode=sanity_mode,
    )
    ds_val = GardnerDataset(
        csv_path=val_csv,
        images_root=images_root,
        task=task,
        split="val",
        image_col=cfg.data.image_col if "data" in cfg else "Image",
        label_col=label_col,
        image_size=int(cfg.data.image_size) if "data" in cfg else int(cfg.image_size),
        augmentation_cfg=cfg.augmentation if "augmentation" in cfg else {},
        sanity_mode=False,
    )

    print("train_size=", len(ds_train), "val_size=", len(ds_val))
    print("num_classes=", num_classes)

    # Collect train labels (after filtering) for class weights
    train_labels = [int(ds_train.df[label_col].iloc[i]) if label_col == "EXP" else int(ds_train.df[label_col].iloc[i])
                    for i in range(len(ds_train.df))]

    # Log class distribution
    from collections import Counter
    label_counts = Counter(train_labels)
    print(f"\nTrain label distribution (task={task}):")
    for cls in sorted(label_counts.keys()):
        count = label_counts[cls]
        pct = 100.0 * count / len(train_labels)
        print(f"  Class {cls}: {count} samples ({pct:.1f}%)")

    # Define flags early
    use_class_weights = bool(cfg.use_class_weights)
    use_weighted_sampler = bool(args.use_weighted_sampler)

    # Startup diagnostics
    print(f"\n[STARTUP DIAGNOSTICS] task={task}, num_classes={num_classes}")
    print(f"  train_size={len(ds_train)}, val_size={len(ds_val)}")
    print(f"  track={track}, loss_name={'CrossEntropyLoss' if task == 'exp' else 'FocalLoss'}")
    print(f"  use_class_weights={use_class_weights}, compute_class_weights_from_train={bool(cfg.compute_class_weights_from_train) if hasattr(cfg, 'compute_class_weights_from_train') else False}")
    print(f"  use_weighted_sampler={use_weighted_sampler}")
    if task == "exp":
        print(f"  label_smoothing={0.0 if sanity_mode else 0.1}")
    else:
        print(f"  focal_gamma=2.0")

    # WeightedRandomSampler setup and logging at startup
    sampler = None

    if use_weighted_sampler:
        alpha = float(getattr(cfg.train, "sampler_alpha", 0.5))  # default 0.5

        sample_weights = []
        for y in train_labels:
            c = label_counts[int(y)]
            w = (1.0 / c) ** alpha
            sample_weights.append(w)

        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )

        print("Using WeightedRandomSampler")
        print(f"Class counts: {dict(sorted(label_counts.items()))}")
        print(f"Class weights: {dict(sorted({cls: 1.0 / count for cls, count in label_counts.items()}.items()))}")
        print(f"Sample weights (first 10): {sample_weights[:10]}")

    # Warning if both are enabled
    if use_weighted_sampler and use_class_weights:
        print("[WARN] Both --use_weighted_sampler=1 and --use_class_weights=1 are enabled. This may lead to double weighting.")

    # Model import and device setup first for pin_memory
    try:
        from src.model import IVF_EffiMorphPP  # <-- change if your path differs
    except Exception as e:
        raise ImportError(
            "Cannot import IVF_EffiMorphPP. Please update the import path in scripts/train_gardner_single.py.\n"
            f"Original error: {e}"
        )

    # In sanity_overfit mode, disable dropout for true overfit test
    dropout_p_value = 0.0 if args.sanity_overfit else float(cfg.model.dropout)

    # Sanity model scale preset
    if args.sanity_overfit:
        scale_mapping = {"small": 1.0, "base": 1.25, "large": 1.5}
        width_mult = scale_mapping[args.sanity_model_scale]
        # Auto-bump to 2.0 if params < 3M
        temp_model = IVF_EffiMorphPP(num_classes=num_classes, dropout_p=dropout_p_value, width_mult=width_mult)
        total_params = sum(p.numel() for p in temp_model.parameters())
        if total_params < 3_000_000 and args.sanity_model_scale == "large":
            width_mult = 2.0
        model = IVF_EffiMorphPP(num_classes=num_classes, dropout_p=dropout_p_value, width_mult=width_mult)
    else:
        model = IVF_EffiMorphPP(num_classes=num_classes, dropout_p=dropout_p_value)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")
    if args.sanity_overfit:
        print(f"[SANITY MODE] Using dropout_p=0.0 for overfitting test")

    # Dataloaders
    batch_size = int(cfg.train.batch_size)
    num_workers = int(cfg.data.num_workers) if "data" in cfg else 4

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        sampler=sampler if use_weighted_sampler else None,
        shuffle=False if use_weighted_sampler else True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # Print model summary
    if torchinfo_summary is not None:
        print("\n" + "="*80)
        print("Model Summary")
        print("="*80)
        try:
            torchinfo_summary(model, input_size=(1, 3, 224, 224), device=device, verbose=0)
        except Exception as e:
            print(f"Could not print model summary: {e}")
        print("="*80 + "\n")
    else:
        # Fallback: simple parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,} | Trainable params: {trainable_params:,}")

    # Loss
    use_class_weights = bool(cfg.use_class_weights)
    loss_fn = make_loss_fn(track=track, task=task, num_classes=num_classes,
                           use_class_weights=use_class_weights, train_labels=train_labels, sanity_mode=sanity_mode, use_weighted_sampler=use_weighted_sampler)

    # Optimizer / scheduler
    lr = float(cfg.optimizer.lr)
    wd = float(cfg.optimizer.weight_decay)
    opt_name = getattr(cfg.optimizer, 'name', 'adam').lower()
    
    if opt_name == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        print(f"Using AdamW optimizer: lr={lr}, weight_decay={wd}")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        print(f"Using Adam optimizer: lr={lr}, weight_decay={wd}")
    
    epochs = int(cfg.train.epochs)
    warmup_epochs = int(getattr(cfg.scheduler, 'warmup_epochs', 0))
    print(f"LR schedule: cosine with warmup_epochs={warmup_epochs}")

    best_metric = -1.0
    best_path = out_dir / "best.ckpt"
    metrics_val_path = out_dir / "metrics_val.json"

    # Sanity overfit test (optional)
    if args.sanity_overfit:
        print("\n" + "="*80)
        print("SANITY OVERFIT TEST MODE")
        print("="*80)
        print(f"Task={task} | Training on {min(args.sanity_samples, len(ds_train))} samples for {args.sanity_epochs} epochs")
        print(f"Expected: train_acc > 0.95 (indicates model can fit small dataset)")
        print("Sanity mode: ALL augmentations disabled, dropout_p=0.0, fixed LR=1e-3")
        print("="*80)
        
        # Tiny subset (smaller batch size for more frequent updates)
        tiny_indices = list(range(min(args.sanity_samples, len(ds_train))))
        tiny_train = torch.utils.data.Subset(ds_train, tiny_indices)
        tiny_loader = DataLoader(tiny_train, batch_size=16, shuffle=True, num_workers=0)
        
        # For final evaluation: use eval transform (deterministic, no augmentation)
        # Simplest: just manually apply eval transform to the tiny images
        tiny_loader_eval = DataLoader(tiny_train, batch_size=32, shuffle=False, num_workers=0)
        # Note: tiny_train was created from ds_train which has sanity_mode=True (no augmentation)
        # So tiny_loader_eval will have the same deterministic transform as tiny_loader
        
        # Print diagnostic info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters: {trainable_params:,} / {total_params:,} trainable")
        
        # Disable batch norm for pure overfitting (removes regularization)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()  # Eval mode: frozen stats, no updates
                for param in module.parameters():
                    param.requires_grad = False
        
        # Use Adam(lr=args.sanity_lr default 2e-3, wd=0) and NO scheduler
        sanity_lr = float(args.sanity_lr)
        sanity_opt = torch.optim.Adam(model.parameters(), lr=sanity_lr, weight_decay=0.0)
        print(f"Sanity optimizer: Adam(lr={sanity_lr}, weight_decay=0.0)")

        # Use sanity_epochs default 60
        sanity_epochs = int(args.sanity_epochs)
        sanity_threshold = float(args.sanity_threshold)
        print(f"Sanity epochs: {sanity_epochs}")
        print(f"Sanity threshold: {sanity_threshold}")

        # Train for sanity_epochs
        model.train()
        final_train_acc = 0.0
        for sanity_epoch in range(1, sanity_epochs + 1):
            sanity_loss = 0.0
            sanity_correct = 0
            sanity_total = 0
            
            for batch in tiny_loader:
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
            if sanity_epoch % 10 == 0 or sanity_epoch == 1 or sanity_epoch == 50:
                print(f"  Epoch {sanity_epoch:2d}/50 | loss={sanity_loss:.4f} | train_acc={sanity_acc:.4f}")
        
        # Final evaluation on the same 200 samples using DETERMINISTIC transform
        print("\n[Evaluating on deterministic transform]")
        model.eval()
        eval_correct = 0
        eval_total = 0
        first_batch_logits_mean = None
        with torch.no_grad():
            for batch_idx, batch in enumerate(tiny_loader_eval):
                x, y, _ = batch
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                if batch_idx == 0:
                    first_batch_logits_mean = logits.mean(dim=0).cpu().numpy()
                preds = logits.argmax(dim=1)
                eval_correct += (preds == y).sum().item()
                eval_total += y.size(0)
        final_train_acc = eval_correct / eval_total
        print(f"  Final train_acc (deterministic eval): {final_train_acc:.4f}")
        print(f"  First batch logits mean: {first_batch_logits_mean}")
        
        if final_train_acc < sanity_threshold:
            print(f"\n[FAILED] Sanity test FAILED: train_acc={final_train_acc:.4f} < {sanity_threshold}")
            print("[DEBUG] Possible issues:")
            print(f"  - Model: {model.__class__.__name__}")
            print(f"  - Task: {task}, num_classes: {num_classes}")
            print(f"  - Tiny dataset size: {len(tiny_train)}")
            print(f"  - Dropout enabled: {any(hasattr(m, 'p') and m.p > 0 for m in model.modules() if type(m).__name__ == 'Dropout')}")
            print(f"  - Sample labels from first batch: {list(range(min(5, eval_total)))}")
            print("="*80 + "\n")
            exit(1)  # Fail-fast: do NOT resume normal training
        else:
            print(f"\n[SUCCESS] Sanity test PASSED: train_acc={final_train_acc:.4f} >= {sanity_threshold}")
        
        print("="*80 + "\n")
        
        # Reset model to fresh weights for actual training
        from src.model import IVF_EffiMorphPP
        model = IVF_EffiMorphPP(num_classes=num_classes, dropout_p=float(cfg.model.dropout))
        model.to(device)
        # Reinit optimizer with new model
        lr = float(cfg.optimizer.lr)
        wd = float(cfg.optimizer.weight_decay)
        opt_name = getattr(cfg.optimizer, 'name', 'adam').lower()
        if opt_name == 'adamw':
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            print(f"[Resuming normal training] Fresh model, AdamW(lr={lr}, weight_decay={wd})")
        else:
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            print(f"[Resuming normal training] Fresh model, Adam(lr={lr}, weight_decay={wd})")

    # Create cosine annealing scheduler with warmup
    total_steps = epochs * len(dl_train)
    warmup_steps = warmup_epochs * len(dl_train)
    
    def get_lr_scale(step: int) -> float:
        """Warmup + cosine annealing schedule."""
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    # Create a lambda scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=get_lr_scale)

    # Training loop (simple)
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in dl_train:
            if len(batch) == 4:
                x, y, _, _ = batch
            else:
                x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            scheduler.step()  # Update LR after each batch
            running += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = running / max(n, 1)

        # val
        val_metrics = evaluate_on_val(model, dl_val, device)
        monitor = val_metrics["macro_f1"]  # align with cfg.monitor.metric if you prefer parsing it
        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
              f"val_acc={val_metrics['acc']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
              f"val_weighted_f1={val_metrics['weighted_f1']:.4f}")

        # save best
        if monitor > best_metric:
            best_metric = monitor
            torch.save({"state_dict": model.state_dict(),
                        "task": task, "track": track,
                        "num_classes": num_classes,
                        "label_col": label_col}, best_path)
            with open(metrics_val_path, "w", encoding="utf-8") as f:
                json.dump({"best_epoch": epoch, "best_val": val_metrics}, f, indent=2)
            print(f"  ✓ Saved best to {best_path} (monitor={best_metric:.4f})")

    # Write debug report
    debug_report_path = out_dir / "debug_report.txt"
    with open(debug_report_path, "w") as f:
        f.write(f"Best epoch: {best_metric}\n")
        f.write(f"Best val_macro_f1: {best_metric:.4f}\n")
        f.write(f"y_pred distribution at best epoch: {val_metrics['y_pred_counts']}\n")
        f.write(f"Confusion matrix at best epoch: {val_metrics.get('confusion_matrix', 'N/A')}\n")
        f.write(f"Class weights applied: {use_class_weights}\n")
        if use_class_weights:
            weights = compute_class_weights(train_labels, num_classes)
            f.write(f"Class weights tensor: {weights.tolist()}\n")
        f.write(f"WeightedRandomSampler used: {use_weighted_sampler}\n")
        f.write(f"Current LR value: {opt.param_groups[0]['lr']:.6f}\n")

        # Check for majority-class collapse
        max_ratio = max(val_metrics['y_pred_ratios'].values())
        if max_ratio > 0.8:
            f.write("Likely majority-class collapse due to imbalance. Recommend enabling WeightedRandomSampler or stronger class reweighting; reduce label_smoothing; reduce warmup.\n")

    print(f"\nDone. Best val macro_f1 = {best_metric:.4f}")
    print(f"Artifacts: {out_dir}")
    print(f"Debug report: {debug_report_path}")


# =========================
# Config merge
# =========================

def load_and_merge_cfg(base: str, task_cfg: str, track_cfg: str) -> OmegaConf:
    cfg_base = OmegaConf.load(base)
    cfg_task = OmegaConf.load(task_cfg)
    cfg_track = OmegaConf.load(track_cfg)
    cfg = OmegaConf.merge(cfg_base, cfg_task, cfg_track)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to base.yaml")
    parser.add_argument("--task_cfg", required=True, help="Path to task yaml (exp/icm/te)")
    parser.add_argument("--track_cfg", required=True, help="Path to track yaml (benchmark_fair/improved)")

    # Optional overrides
    parser.add_argument("--splits_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--images_root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--use_class_weights", type=int, default=None)
    parser.add_argument("--use_weighted_sampler", type=int, default=0, help="Use WeightedRandomSampler for training (0=OFF, 1=ON)")
    parser.add_argument("--sanity_overfit", action="store_true",
                        help="Run sanity overfit test on a tiny subset then exit (fail-fast).")

    parser.add_argument("--sanity_samples", type=int, default=200,
                        help="Number of training samples used in sanity overfit mode.")

    parser.add_argument("--sanity_epochs", type=int, default=60,
                        help="Epochs for sanity overfit mode (default 60).")

    parser.add_argument("--sanity_lr", type=float, default=2e-3,
                        help="Learning rate for sanity overfit optimizer (default 2e-3).")

    parser.add_argument("--sanity_threshold", type=float, default=0.95,
                        help="Required deterministic train accuracy for sanity pass.")

    parser.add_argument("--sanity_model_scale", type=str, default="large",
                        choices=["small", "base", "large"],
                        help="Model capacity preset used in sanity mode.")

    args = parser.parse_args()

    cfg = load_and_merge_cfg(args.config, args.task_cfg, args.track_cfg)

    # Apply CLI overrides if provided
    if args.splits_dir is not None:
        cfg.splits_dir = args.splits_dir
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.images_root is not None:
        cfg.images_root = args.images_root
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.use_class_weights is not None:
        cfg.use_class_weights = bool(args.use_class_weights)
    if args.use_weighted_sampler is not None:
        cfg.use_weighted_sampler = bool(args.use_weighted_sampler)

    # Sanity: task/track must exist
    if "task" not in cfg or "track" not in cfg:
        raise ValueError("Config merge failed: missing `task` or `track` fields. Check YAMLs.")

    set_seed(int(cfg.seed), bool(cfg.deterministic))
    print(f"Task={cfg.task} | Track={cfg.track}")
    print(f"Splits dir={cfg.splits_dir} | Out dir={cfg.out_dir}")

    train_one_run(cfg, args)


if __name__ == "__main__":
    main()
