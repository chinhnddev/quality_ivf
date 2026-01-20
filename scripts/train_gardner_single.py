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
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
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


def make_loss_fn(track: str, task: str, num_classes: int, use_class_weights: bool, train_labels: List[int]) -> nn.Module:
    weights = None
    if use_class_weights:
        weights = compute_class_weights(train_labels, num_classes)
        # Warning if any missing class
        if (weights == 0).any():
            missing = [i for i, w in enumerate(weights.tolist()) if w == 0.0]
            print(f"[WARN] Missing classes in TRAIN for task={task}: {missing}. Their weight is set to 0.")
    if track == "benchmark_fair":
        return nn.CrossEntropyLoss(weight=weights)
    if track == "improved":
        if task == "exp":
            # label smoothing for EXP
            return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
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
    ):
        self.csv_path = csv_path
        self.images_root = images_root
        self.task = task
        self.split = split
        self.image_col = image_col
        self.label_col = label_col
        self.image_size = image_size

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
        self.transform = self._build_transform(augmentation_cfg)

    def _build_transform(self, aug: Optional[dict]):
        size = self.image_size
        if self.split == "train":
            # Default augmentation per your base.yaml
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

        # val/test
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
    for x, y, _ in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy().tolist()
        ys.extend(y.numpy().tolist())
        ps.extend(pred)
    acc = accuracy_score(ys, ps) if len(ys) else 0.0
    macro_f1 = f1_score(ys, ps, average="macro") if len(ys) else 0.0
    weighted_f1 = f1_score(ys, ps, average="weighted") if len(ys) else 0.0
    return {"acc": float(acc), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1)}


def train_one_run(cfg) -> None:
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
    ds_train = GardnerDataset(
        csv_path=train_csv,
        images_root=images_root,
        task=task,
        split="train",
        image_col=cfg.data.image_col if "data" in cfg else "Image",
        label_col=label_col,
        image_size=int(cfg.data.image_size) if "data" in cfg else int(cfg.image_size),
        augmentation_cfg=cfg.augmentation if "augmentation" in cfg else {},
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
    )

    print("train_size=", len(ds_train), "val_size=", len(ds_val))
    print("num_classes=", num_classes)

    # Collect train labels (after filtering) for class weights
    train_labels = [int(ds_train.df[label_col].iloc[i]) if label_col == "EXP" else int(ds_train.df[label_col].iloc[i])
                    for i in range(len(ds_train.df))]

    # Dataloaders
    batch_size = int(cfg.train.batch_size)
    num_workers = int(cfg.data.num_workers) if "data" in cfg else 4
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model import (EDIT THIS LINE to match your repo)
    # Example:
    # from src.model import IVF_EffiMorphPP
    # model = IVF_EffiMorphPP(num_classes=num_classes, dropout_p=float(cfg.model.dropout))
    try:
        from src.model import IVF_EffiMorphPP  # <-- change if your path differs
    except Exception as e:
        raise ImportError(
            "Cannot import IVF_EffiMorphPP. Please update the import path in scripts/train_gardner_single.py.\n"
            f"Original error: {e}"
        )

    model = IVF_EffiMorphPP(num_classes=num_classes, dropout_p=float(cfg.model.dropout))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

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
                           use_class_weights=use_class_weights, train_labels=train_labels)

    # Optimizer / scheduler
    lr = float(cfg.optimizer.lr)
    wd = float(cfg.optimizer.weight_decay)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(cfg.train.epochs)
    best_metric = -1.0
    best_path = out_dir / "best.ckpt"
    metrics_val_path = out_dir / "metrics_val.json"

    # Training loop (simple)
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y, _ in dl_train:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = running / max(n, 1)

        # val
        val_metrics = evaluate_on_val(model, dl_val, device)
        monitor = val_metrics["macro_f1"]  # align with cfg.monitor.metric if you prefer parsing it
        print(f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f} | "
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

    print(f"\nDone. Best val macro_f1 = {best_metric:.4f}")
    print(f"Artifacts: {out_dir}")


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

    # Sanity: task/track must exist
    if "task" not in cfg or "track" not in cfg:
        raise ValueError("Config merge failed: missing `task` or `track` fields. Check YAMLs.")

    set_seed(int(cfg.seed), bool(cfg.deterministic))
    print(f"Task={cfg.task} | Track={cfg.track}")
    print(f"Splits dir={cfg.splits_dir} | Out dir={cfg.out_dir}")

    train_one_run(cfg)


if __name__ == "__main__":
    main()
