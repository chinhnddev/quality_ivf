#!/usr/bin/env python3
# scripts/train_gardner_single.py

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter, defaultdict
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torchvision import transforms
from PIL import Image

try:
    from torchinfo import summary as torchinfo_summary
except ImportError:
    torchinfo_summary = None

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CORAL functions
from src.loss_coral import coral_loss, coral_predict_class
from src.utils import normalize_exp_token, normalize_icm_te_token, print_label_distribution

IGNORE_INDEX = -100

DEFAULT_CORAL_THRESHOLDS = [0.55, 0.55, 0.55, 0.55]

DEFAULT_CORAL_THRESHOLDS = [0.5, 0.5, 0.5, 0.6]


def _extract_monitor_from_cfg(cfg: Optional[OmegaConf], task: str) -> Optional[Dict[str, Any]]:
    if cfg is None or not hasattr(cfg, "monitor"):
        return None
    monitor_cfg = getattr(cfg, "monitor")
    if monitor_cfg is None:
        return None
    if isinstance(monitor_cfg, str):
        return {"metric": str(monitor_cfg)}
    container = OmegaConf.to_container(monitor_cfg, resolve=True) if OmegaConf.is_config(monitor_cfg) else monitor_cfg
    if isinstance(container, dict):
        metric_value = None
        if task in container and container.get(task):
            metric_value = container.get(task)
        elif container.get("metric"):
            metric_value = container.get("metric")
        if metric_value:
            return {
                "metric": str(metric_value),
                "mode": str(container.get("mode", "max")),
                "patience": container.get("patience", 10),
            }
    return None


def _monitor_key_from_label(label: str) -> str:
    if label.startswith("val_"):
        return label[len("val_") :]
    return label


def _has_monitor_improvement(current: float, best: Optional[float], mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best
    return current > best


def _resolve_monitor(
    task: str, base_cfg: OmegaConf, track_cfg: OmegaConf, task_cfg: OmegaConf
) -> Tuple[str, str, int, str]:
    default_metric = "val_acc" if task == "exp" else "val_macro_f1"
    default_mode = "max"
    default_patience = 10
    for cfg, source in (
        (task_cfg, "task_cfg"),
        (track_cfg, "track_cfg"),
        (base_cfg, "base_cfg"),
    ):
        monitor_cfg = _extract_monitor_from_cfg(cfg, task)
        if monitor_cfg:
            label = monitor_cfg.get("metric", default_metric)
            mode = str(monitor_cfg.get("mode", default_mode))
            patience = monitor_cfg.get("patience", default_patience)
            return label, mode, int(patience), source
    return default_metric, default_mode, default_patience, "default"


# =========================
# Checkpoint Loading
# =========================

def load_backbone_only(model, ckpt_path: str, device="cpu"):
    """Load stage-pretrain backbone weights, excluding head layers."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)  # support both formats
    
    # drop any head weights
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[PRETRAIN LOAD] loaded backbone from: {ckpt_path}")
    print(f"[PRETRAIN LOAD] missing keys (expected head.*): {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[PRETRAIN LOAD] unexpected keys: {unexpected}")
    return model


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


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    mode: str = "inverse",
    beta: float = 0.999,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute per-class weights using inverse-frequency or effective number."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for y in labels:
        if 0 <= y < num_classes:
            counts[y] += 1
    weights = np.zeros(num_classes, dtype=np.float64)
    if mode == "effective_num":
        effective = 1.0 - np.power(beta, counts)
        for idx, (cnt, eff) in enumerate(zip(counts, effective)):
            if cnt > 0 and eff > eps:
                weights[idx] = (1.0 - beta) / (eff + eps)
            else:
                weights[idx] = 0.0
    else:
        total = counts.sum()
        K = float(num_classes)
        for c in range(num_classes):
            if counts[c] > 0:
                weights[c] = total / (K * (counts[c] + eps))
            else:
                weights[c] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def finalize_class_weights(weights: torch.Tensor, cap_multiplier: float = 10.0, eps: float = 1e-8) -> torch.Tensor:
    if weights is None or weights.numel() == 0:
        return weights
    mean = float(weights.mean())
    if mean > 0:
        weights = weights / mean
    else:
        return weights
    w0 = float(weights[0]) if weights.numel() > 0 else 0.0
    max_weight = float(weights.max()) if weights.numel() > 0 else 0.0
    cap = w0 * cap_multiplier if w0 > 0 else max_weight * cap_multiplier if max_weight > 0 else 0.0
    if cap > 0:
        weights = torch.clamp(weights, max=cap)
        mean = float(weights.mean())
        if mean > 0:
            weights = weights / mean
    return weights


def _dictify_loss_cfg(loss_cfg: Optional[Union[dict, OmegaConf]]) -> Dict[str, Any]:
    if loss_cfg is None:
        return {}
    if isinstance(loss_cfg, dict):
        return loss_cfg
    if OmegaConf.is_config(loss_cfg):
        return OmegaConf.to_container(loss_cfg, resolve=True)
    return {}


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        original_device = logits.device
        original_dtype = logits.dtype
        mask = None
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            if not mask.any():
                return torch.zeros((), device=original_device, dtype=original_dtype)
            logits = logits[mask]
            target = target[mask]
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def make_loss_fn(
    track: str,
    task: str,
    num_classes: int,
    use_class_weights: bool,
    train_labels: List[int],
    sanity_mode: bool = False,
    use_weighted_sampler: bool = False,
    use_coral: bool = False,
    label_smoothing: float = 0.0,
    loss_cfg: Optional[Union[dict, OmegaConf]] = None,
    class_weight_mode: str = "inverse",
    class_weight_beta: float = 0.999,
    class_weight_cap: float = 10.0,
) -> Tuple[nn.Module, Dict[str, Any]]:
    metadata: Dict[str, Any] = {"loss_name": "cross_entropy"}
    if use_coral and task == "exp":
        # CORAL loss for ordinal regression
        return lambda logits, targets: coral_loss(logits, targets, num_classes), metadata
    weights = None
    if use_class_weights:
        weights = compute_class_weights(
            train_labels,
            num_classes,
            mode=class_weight_mode,
            beta=class_weight_beta,
        )
        weights = finalize_class_weights(weights, cap_multiplier=class_weight_cap)
        metadata["class_weights_tensor"] = weights.clone().detach()
        metadata["class_weights"] = weights.tolist()
        if (weights == 0).any():
            missing = [i for i, w in enumerate(weights.tolist()) if w == 0.0]
            print(f"[WARN] Missing classes in TRAIN for task={task}: {missing}. Their weight is set to 0.")
        else:
            print(f"[OK] Class weights computed for task={task}:")
            for i, w in enumerate(weights.tolist()):
                print(f"  Class {i}: weight={w:.4f}")
    loss_params = _dictify_loss_cfg(loss_cfg)
    loss_name = loss_params.get("name", "cross_entropy")

    ignore_idx = IGNORE_INDEX if task in {"icm", "te"} else None
    def _cross_entropy_kwargs(smoothing: float) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"weight": weights, "label_smoothing": float(smoothing)}
        if ignore_idx is not None:
            kwargs["ignore_index"] = ignore_idx
        return kwargs
    if track == "benchmark_fair":
        smoothing = 0.0 if use_weighted_sampler or sanity_mode else 0.0
        loss = nn.CrossEntropyLoss(**_cross_entropy_kwargs(smoothing))
        metadata["loss_name"] = "cross_entropy"
        return loss, metadata
    if track == "improved":
        if task == "exp":
            if loss_name == "focal":
                gamma = float(loss_params.get("focal_gamma", 2.2))
                metadata["focal_gamma"] = gamma
                loss = FocalLoss(
                    gamma=gamma,
                    weight=weights if use_class_weights else None,
                    ignore_index=ignore_idx,
                )
                metadata["loss_name"] = "focal"
                return loss, metadata
            smoothing = 0.0 if use_weighted_sampler or sanity_mode else float(label_smoothing)
            loss = nn.CrossEntropyLoss(**_cross_entropy_kwargs(smoothing))
            metadata["loss_name"] = "cross_entropy"
            return loss, metadata
        if loss_name == "focal":
            gamma = float(loss_params.get("focal_gamma", 2.2))
            metadata["focal_gamma"] = gamma
            loss = FocalLoss(
                gamma=gamma,
                weight=weights if use_class_weights else None,
                ignore_index=ignore_idx,
            )
            metadata["loss_name"] = "focal"
            return loss, metadata
        loss = nn.CrossEntropyLoss(**_cross_entropy_kwargs(label_smoothing))
        metadata["loss_name"] = loss_name
        return loss, metadata
    raise ValueError(f"Unknown track: {track}")


# =========================
# Dataset
# =========================

def build_icm_merge_map(merge_to: int) -> Tuple[Dict[int, int], Dict[int, int]]:
    base = {0: 0, 1: 1, 2: merge_to, 3: 3}
    unique = sorted(set(base.values()))
    new_idx = {label: idx for idx, label in enumerate(unique)}
    final_map = {orig: new_idx[base[orig]] for orig in base}
    reverse_map = {idx: label for label, idx in new_idx.items()}
    return final_map, reverse_map


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
        exclude_na_nd: bool = False,
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

        df["Image"] = df["Image"].astype(str).str.strip()
        if label_col == "EXP":
            df["norm_label"] = df["EXP"].apply(normalize_exp_token)
        else:
            df["norm_label"] = df[label_col].apply(normalize_icm_te_token).astype(int)

        # Filtering rule:
        # - train/val: filter for ICM/TE valid labels {0,1,2,3}; EXP keeps all
        # - test: do NOT filter (load all); masking is for eval script
        if split in {"train", "val"}:
            if label_col == "EXP":
                df = df[df["norm_label"].isin({"0", "1", "2", "3", "4"})].copy()
            else:
                df = df[df["norm_label"].isin({0, 1, 2, 3})].copy()
                if exclude_na_nd and split == "train":
                    df = df[df["norm_label"] != 3]
        elif split == "test":
            pass
        else:
            raise ValueError(f"Unknown split: {split}")

        if label_col != "EXP" and self.task in {"icm", "te"}:
            df["target_label"] = df["norm_label"].apply(
                lambda x: IGNORE_INDEX if x == 3 else int(x)
            )
        else:
            df["target_label"] = df["norm_label"].astype(int)

        self.df = df.reset_index(drop=True)
        self.augmentation_cfg = augmentation_cfg or {}

        # Transforms
        self.transform = self._build_transform(self.augmentation_cfg, sanity_mode)
        
    def _build_transform(self, aug: Optional[dict], sanity_mode: bool = False):
        aug_cfg = {}
        if aug:
            try:
                aug_cfg.update(dict(aug))
            except Exception:
                if isinstance(aug, dict):
                    aug_cfg.update(aug)
                else:
                    aug_cfg.update({k: aug[k] for k in aug})

        base_defaults = {
            "rotation_deg": 10,
            "horizontal_flip": True,
            "vertical_flip": True,
            "random_resized_crop": True,
            "rrc_scale": [0.8, 1.0],
            "rrc_ratio": [0.9, 1.1],
            # Optional tensor-space augment (recommend OFF by default)
            "erasing_p": 0.0,
            "erasing_scale": [0.02, 0.10],
            "erasing_ratio": [0.3, 1.0],
        }

        # Conservative defaults for ICM/TE (light aug only)
        icm_te_defaults = {
            "rotation_deg": 10,
            "horizontal_flip": False,
            "vertical_flip": False,
            # Control RRC for ICM/TE via icm_train_use_rrc (default False)
            "icm_train_use_rrc": False,
            "icm_rrc_scale": [0.95, 1.00],
            "icm_rrc_ratio": [0.98, 1.02],
            "icm_rotation_deg": 10,
            "icm_hflip_p": 0.0,
            # Keep paper-style key for backward compatibility; not used to gate ICM/TE anymore
            # Keep erasing off unless explicitly enabled in config
            "erasing_p": 0.0,
        }

        is_icm_te = self.task in {"icm", "te"}
        transform_cfg = {**base_defaults, **(icm_te_defaults if is_icm_te else {}), **aug_cfg}

        def _to_tuple(value):
            if isinstance(value, (list, tuple)):
                return tuple(value)
            return tuple(value) if hasattr(value, "__iter__") else (value,)

        rotation_deg = float(transform_cfg.get("rotation_deg", 0))
        horizontal_flip_flag = bool(transform_cfg.get("horizontal_flip", True))
        vertical_flip_flag = bool(transform_cfg.get("vertical_flip", True))
        use_rrc = bool(transform_cfg.get("random_resized_crop", True))
        rrc_scale = _to_tuple(transform_cfg.get("rrc_scale", [0.8, 1.0]))
        rrc_ratio = _to_tuple(transform_cfg.get("rrc_ratio", [0.9, 1.1]))
        hflip_prob = 0.5 if horizontal_flip_flag else 0.0

        erasing_p = float(transform_cfg.get("erasing_p", 0.0))
        erasing_scale = _to_tuple(transform_cfg.get("erasing_scale", [0.02, 0.10]))
        erasing_ratio = _to_tuple(transform_cfg.get("erasing_ratio", [0.3, 1.0]))

        # Task-specific overrides for ICM/TE
        if is_icm_te:
            rotation_deg = float(transform_cfg.get("icm_rotation_deg", rotation_deg))
            hflip_prob = float(transform_cfg.get("icm_hflip_p", 0.0))
            vertical_flip_flag = False

            # For ICM/TE, RRC is controlled by icm_train_use_rrc (NOT random_resized_crop)
            use_rrc = bool(transform_cfg.get("icm_train_use_rrc", False))
            rrc_scale = _to_tuple(transform_cfg.get("icm_rrc_scale", [0.95, 1.00]))
            rrc_ratio = _to_tuple(transform_cfg.get("icm_rrc_ratio", [0.98, 1.02]))

        # Gate RRC for non-ICM/TE tasks only
        if (not is_icm_te) and (not transform_cfg.get("random_resized_crop", True)):
            use_rrc = False

        target_size = int(self.image_size)
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.split == "train" and not sanity_mode:
            pipeline = []

            # Geometric (PIL)
            if use_rrc:
                pipeline.append(
                    transforms.RandomResizedCrop(target_size, scale=rrc_scale, ratio=rrc_ratio)
                )
            else:
                pipeline.append(transforms.Resize(target_size))
                pipeline.append(transforms.CenterCrop(target_size))

            # Photometric jitter (PIL) - must be before ToTensor
            if is_icm_te:
                jitter_cfg = transform_cfg.get("icm_photometric_jitter") or transform_cfg.get("photometric_jitter")
            else:
                jitter_cfg = transform_cfg.get("photometric_jitter")
            if jitter_cfg:
                pipeline.append(transforms.ColorJitter(**dict(jitter_cfg)))

            if hflip_prob > 0.0:
                pipeline.append(transforms.RandomHorizontalFlip(p=hflip_prob))
            if vertical_flip_flag:
                pipeline.append(transforms.RandomVerticalFlip(p=0.5))
            if rotation_deg > 0:
                pipeline.append(transforms.RandomRotation(degrees=rotation_deg))

            # Tensor ops
            pipeline.append(transforms.ToTensor())

            # Optional RandomErasing (Tensor) - keep OFF unless explicitly enabled
            if is_icm_te and erasing_p > 0.0:
                pipeline.append(
                    transforms.RandomErasing(
                        p=erasing_p,
                        scale=erasing_scale,
                        ratio=erasing_ratio,
                        value="random",
                    )
                )

            pipeline.append(norm)

            transform = transforms.Compose(pipeline)
            print(f"[TRANSFORM] task={self.task} split={self.split} pipeline={transform}")
            return transform

        deterministic = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                norm,
            ]
        )
        label = "SANITY" if sanity_mode else "VAL/TEST"
        print(f"[TRANSFORM] task={self.task} split={self.split} pipeline={deterministic} ({label})")
        return deterministic

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

        y = int(row["target_label"])

        return x, y, img_name


# =========================
# Training
# =========================

@torch.no_grad()
def evaluate_on_val(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    epoch: int,
    use_coral: bool = False,
    num_classes: int = 0,
    compute_confusion_matrix: bool = False,
    coral_thresholds: Optional[List[float]] = None,
    precomputed_logits: Optional[torch.Tensor] = None,
    precomputed_labels: Optional[torch.Tensor] = None,
) -> Dict:
    model.eval()
    ys, ps = [], []
    thresholds_to_use = coral_thresholds or DEFAULT_CORAL_THRESHOLDS
    if precomputed_logits is not None and precomputed_labels is not None:
        logits_tensor = precomputed_logits.to(device)
        label_tensor = precomputed_labels.to(device)
        if use_coral:
            preds = coral_predict_class(logits_tensor, thresholds=thresholds_to_use).cpu()
        else:
            preds = logits_tensor.argmax(dim=1).cpu()
        y = label_tensor.cpu()
        mask = y != IGNORE_INDEX
        if mask.any():
            ys.extend(y[mask].tolist())
            ps.extend(preds[mask].tolist())
    else:
        for batch in loader:
            if len(batch) == 4:
                x, y, _, _ = batch
            else:
                x, y, _ = batch
            x = x.to(device)
            logits = model(x)
            if use_coral:
                preds = coral_predict_class(logits, thresholds=thresholds_to_use).cpu()
            else:
                preds = logits.argmax(dim=1).cpu()
            y = y.cpu()
            mask = y != IGNORE_INDEX
            if not mask.any():
                continue
            ys.extend(y[mask].tolist())
            ps.extend(preds[mask].tolist())
    if not ys:
        empty_metrics = {
            "acc": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "avg_precision_weighted": 0.0,
            "avg_recall_weighted": 0.0,
            "avg_f1_weighted": 0.0,
            "per_class_precision": [0.0 for _ in range(max(num_classes, 1))],
            "per_class_recall": [0.0 for _ in range(max(num_classes, 1))],
            "per_class_f1": [0.0 for _ in range(max(num_classes, 1))],
            "per_class_support": [0 for _ in range(max(num_classes, 1))],
            "y_pred_counts": {},
            "y_pred_ratios": {},
            "y_true_counts": {},
            "y_true_ratios": {},
        }
        return empty_metrics
    acc = accuracy_score(ys, ps)
    macro_f1 = f1_score(ys, ps, average="macro")
    weighted_f1 = f1_score(ys, ps, average="weighted")
    avg_precision_weighted = precision_score(ys, ps, average="weighted", zero_division=0)
    avg_recall_weighted = recall_score(ys, ps, average="weighted", zero_division=0)
    avg_f1_weighted = f1_score(ys, ps, average="weighted", zero_division=0)

    labels = list(range(num_classes if num_classes > 0 else len(set(ys))))
    precision, recall, f1, support = precision_recall_fscore_support(
        ys, ps, labels=labels, average=None, zero_division=0
    )

    y_pred_counts = Counter(ps)
    total_preds = len(ps)
    y_pred_ratios = {cls: float(count / total_preds) for cls, count in y_pred_counts.items()}
    y_true_counts = Counter(ys)
    y_true_ratios = {cls: float(count / total_preds) for cls, count in y_true_counts.items()}

    cm = None
    cm_labels: List[int] = []
    if compute_confusion_matrix and labels:
        cm_labels = labels
        try:
            cm = confusion_matrix(ys, ps, labels=labels)
        except ValueError:
            cm = confusion_matrix(ys, ps)
            cm_labels = sorted(set(ys) | set(ps))

    if task in {"icm", "te"}:
        unique_labels = sorted(set(ys))
        if epoch == 1:
            if unique_labels:
                print(
                    f"[VAL DEBUG] task={task} epoch={epoch} y_true unique={unique_labels} "
                    f"min={min(unique_labels)} max={max(unique_labels)}"
                )
            else:
                print(f"[VAL DEBUG] task={task} epoch={epoch} y_true empty")
            print(f"[VAL DEBUG] y_pred counts: {dict(sorted(y_pred_counts.items()))}")
        assert set(unique_labels).issubset({0, 1, 2}), "ICM/TE val contains invalid labels after filtering"

    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "avg_precision_weighted": float(avg_precision_weighted),
        "avg_recall_weighted": float(avg_recall_weighted),
        "avg_f1_weighted": float(avg_f1_weighted),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "y_pred_counts": dict(sorted(y_pred_counts.items())),
        "y_pred_ratios": dict(sorted(y_pred_ratios.items())),
        "y_true_counts": dict(sorted(y_true_counts.items())),
        "y_true_ratios": dict(sorted(y_true_ratios.items())),
        "confusion_matrix": cm.tolist() if cm is not None else None,
        "confusion_matrix_labels": cm_labels if cm is not None else [],
    }


def collect_coral_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_acc, label_acc = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                x, y, _, _ = batch
            else:
                x, y, _ = batch
            x = x.to(device)
            logits = model(x)
            logits_acc.append(logits.detach().cpu())
            label_acc.append(y.detach().cpu())
    if not logits_acc:
        return torch.empty((0, 4)), torch.empty((0,), dtype=torch.long)
    return torch.cat(logits_acc, dim=0), torch.cat(label_acc, dim=0)


def tune_coral_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    num_classes: int = 5,
    base_thr: float = 0.5,
    logits: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> Optional[Dict]:
    """
    Tune CORAL thresholds for EXP (ordinal 0..4) in a robust way.

    Strategy:
      - Keep thresholds for first K-2 ranks fixed at base_thr
      - Tune ONLY the last threshold (k=K-2), which controls emergence of top class (class 4)
      - Select best by macro-F1 (not weighted-F1) to avoid collapsing minority classes

    Returns:
      dict with best thresholds vector and grid results
    """
    if logits is None or labels is None:
        logits, labels = collect_coral_logits(model, loader, device)
    if logits.numel() == 0:
        print("[CORAL TUNING] No validation logits collected; skipping threshold search.")
        return None

    thr_min = float(thr_min)
    thr_max = float(thr_max)
    thr_step = float(thr_step)
    if thr_step <= 0 or thr_min > thr_max:
        print("[CORAL TUNING] Invalid threshold search range; skipping.")
        return None

    # CORAL logits dim must be K-1
    k_minus_1 = logits.shape[1]
    if k_minus_1 != (num_classes - 1):
        print(f"[CORAL TUNING] Unexpected logits dim={k_minus_1}, expected {num_classes-1}; skipping.")
        return None

    thr_values = np.arange(thr_min, thr_max + 1e-8, thr_step)
    y_true = labels.detach().cpu().numpy()

    best = None
    grid = []

    # thresholds vector: [t0, t1, ..., t_{K-2}]
    # We tune only last one t_{K-2}
    fixed = [float(base_thr)] * (num_classes - 2)

    for thr_last in thr_values:
        thr_vec = fixed + [float(thr_last)]

        preds = coral_predict_class(logits, thresholds=thr_vec).detach().cpu().numpy()

        f1_macro = f1_score(
            y_true,
            preds,
            average="macro",
            zero_division=0,
            labels=list(range(num_classes)),
        )
        f1_weighted = f1_score(
            y_true,
            preds,
            average="weighted",
            zero_division=0,
            labels=list(range(num_classes)),
        )
        acc = accuracy_score(y_true, preds)

        pred_counts = np.bincount(preds, minlength=num_classes)
        has_top_class = int(pred_counts[num_classes - 1] > 0)

        grid.append(
            {
                "thresholds": [round(x, 4) for x in thr_vec],
                "thr_last": round(float(thr_last), 4),
                "val_f1_macro": float(f1_macro),
                "val_f1_weighted": float(f1_weighted),
                "val_acc": float(acc),
                "pred_top_class_nonzero": has_top_class,
                "pred_counts": pred_counts.tolist(),
            }
        )

        if best is None or f1_macro > best["f1_macro"] or (
            abs(f1_macro - best["f1_macro"]) < 1e-8 and acc > best["acc"]
        ):
            best = {
                "thresholds": thr_vec,
                "thr_last": float(thr_last),
                "f1_macro": float(f1_macro),
                "f1_weighted": float(f1_weighted),
                "acc": float(acc),
                "pred_counts": pred_counts.tolist(),
            }

    if best is not None:
        t_str = ",".join([f"{t:.3f}" for t in best["thresholds"]])
        print(
            f"[CORAL TUNING] Best thresholds=[{t_str}] | "
            f"val_f1_macro={best['f1_macro']:.4f} | val_acc={best['acc']:.4f} | "
            f"pred_counts={best['pred_counts']}"
        )
        return {
            "best_thr": best["thresholds"],
            "best_thr_last": best["thr_last"],
            "best_f1_macro": best["f1_macro"],
            "best_f1_weighted": best["f1_weighted"],
            "best_acc": best["acc"],
            "grid": grid,
        }

    return None


def train_one_run(cfg, args) -> None:
    task = cfg.task
    track = cfg.track
    splits_dir = Path(cfg.splits_dir)
    out_dir = Path(cfg.out_dir) / track / task
    ensure_dir(out_dir)
    metrics_cfg = getattr(cfg, "metrics", {})
    compute_confusion_matrix = bool(getattr(metrics_cfg, "compute_confusion_matrix", False))

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
        print_label_distribution(raw_train_df, label_col, task, "train.csv (raw)")
        print_label_distribution(raw_val_df, label_col, task, "val.csv (raw)")

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
        exclude_na_nd=bool(getattr(cfg, "train_icmte_exclude_na_nd", False)),
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

    merge_target = None
    merge_info = {}
    if task == "icm":
        merge_target = getattr(cfg, "icm_merge_class2_to", None)
        if args.icm_merge_class2_to is not None:
            merge_target = args.icm_merge_class2_to
        if merge_target not in (None, 1, 3):
            if merge_target is not None:
                raise ValueError("icm_merge_class2_to must be 1 or 3 when set.")
        if merge_target is not None:
            final_map, reverse_map = build_icm_merge_map(merge_target)
            for ds in (ds_train, ds_val):
                ds.df["norm_label"] = (
                    ds.df["norm_label"]
                    .astype(int)
                    .map(final_map)
                    .fillna(max(final_map.values()))
                    .astype(int)
                )
            num_classes = len(reverse_map)
            merge_info = {
                "merge_target": merge_target,
                "label_map": final_map,
                "reverse_map": reverse_map,
            }
            print(f"[ICM MERGE] class2 -> {merge_target}; resulting labels: {sorted(reverse_map.values())}")
    label_counts_after = Counter(ds_train.df["norm_label"])
    print(f"[DATA FILTER] {task.upper()} train after filtering: {len(ds_train)} samples; val: {len(ds_val)} samples")
    print(f"  Train label counts: {dict(sorted(label_counts_after.items()))}")
    val_targets_full = ds_val.df["target_label"].tolist()
    val_label_counts = Counter(y for y in val_targets_full if y != IGNORE_INDEX)
    ignored_val = len(val_targets_full) - sum(1 for y in val_targets_full if y != IGNORE_INDEX)
    print(f"  Val label counts:   {dict(sorted(val_label_counts.items()))}")
    print(f"  Ignored ND/NA in val: {ignored_val}")
    print(f"num_classes=", num_classes)

    # Collect train labels (after filtering) for class weights
    train_targets_full = ds_train.df["target_label"].tolist()
    train_labels = [int(y) for y in train_targets_full if y != IGNORE_INDEX]
    valid_train_indices = [i for i, y in enumerate(train_targets_full) if y != IGNORE_INDEX]

    # Log class distribution
    label_counts = Counter(train_labels)
    print(f"\nTrain label distribution (task={task}):")
    total_train_valid = len(train_labels)
    for cls in range(num_classes):
        count = label_counts.get(cls, 0)
        pct = 100.0 * count / max(total_train_valid, 1)
        print(f"  Class {cls}: {count} samples ({pct:.1f}%)")
    ignored_train = len(train_targets_full) - total_train_valid
    print(f"  Ignored ND/NA samples (label=3): {ignored_train}")

    # Define flags early
    use_class_weights = bool(cfg.use_class_weights)
    use_weighted_sampler = bool(getattr(cfg, "use_weighted_sampler", False))
    if args.use_weighted_sampler is not None:
        use_weighted_sampler = bool(args.use_weighted_sampler)
    sampling_cfg = getattr(cfg, "sampling", {})
    sampler_use_sqrt_inv = bool(sampling_cfg.get("use_sqrt_inverse", False))
    sampler_cap_ratio = float(sampling_cfg.get("cap_ratio", 5.0))
    if args.sampler_use_sqrt_inv is not None:
        sampler_use_sqrt_inv = bool(args.sampler_use_sqrt_inv)
    if args.sampler_cap_ratio is not None:
        sampler_cap_ratio = float(args.sampler_cap_ratio)
    if use_weighted_sampler and use_class_weights:
        print("[WARN] WeightedRandomSampler and class weights both enabled; watch for double weighting.")
    use_coral = bool(args.use_coral) and task == "exp"
    loss_cfg = getattr(cfg, "loss", {})
    label_smoothing_cfg = float(getattr(loss_cfg, "label_smoothing", 0.0))
    class_weight_mode_cfg = getattr(loss_cfg, "class_weight_mode", "effective_num")
    class_weight_beta_cfg = float(getattr(loss_cfg, "class_weight_beta", 0.999))
    class_weight_cap_cfg = float(getattr(loss_cfg, "class_weight_cap", 10.0))

    if task == "exp":
        loss_name = "CORAL_BCEWithLogits" if use_coral else "CrossEntropyLoss"
    else:
        loss_name = "FocalLoss" if track == "improved" else "CrossEntropyLoss"
    applied_smoothing = 0.0
    applied_smoothing = label_smoothing_cfg
    if not use_coral and task == "exp" and (use_weighted_sampler or sanity_mode):
        applied_smoothing = 0.0

    print(f"[CONFIG] use_class_weights={use_class_weights}, use_weighted_sampler={use_weighted_sampler}")
    # Startup diagnostics
    print(f"\n[STARTUP DIAGNOSTICS] task={task}, num_classes={num_classes}")
    print(f"  train_size={len(ds_train)}, val_size={len(ds_val)}")
    print(f"  track={track}, loss_name={loss_name}")
    print(f"  compute_class_weights_from_train={bool(getattr(cfg, 'compute_class_weights_from_train', False))}")
    if task == "exp":
        print(f"  label_smoothing={applied_smoothing}")
    else:
        print(f"  label_smoothing={applied_smoothing} (CrossEntropyLoss)")

    # WeightedRandomSampler setup and logging at startup
    sampler = None
    train_dataset = ds_train

    if use_weighted_sampler:
        alpha = float(getattr(cfg.train, "sampler_alpha", 0.5))  # default 0.5
        class_sampling_weights = {}
        for cls in range(num_classes):
            count = label_counts.get(cls, 0)
            if count > 0:
                base = 1.0 / (math.sqrt(count) if sampler_use_sqrt_inv else count)
            else:
                base = 0.0
            class_sampling_weights[int(cls)] = float(base ** alpha if alpha != 1.0 else base)
        valid_weights = [w for w in class_sampling_weights.values() if w > 0]
        if valid_weights:
            min_w = min(valid_weights)
            max_allowed = sampler_cap_ratio * min_w
            current_max = max(valid_weights)
            if current_max > max_allowed and min_w > 0.0:
                scale = max_allowed / current_max
                for cls in class_sampling_weights:
                    class_sampling_weights[cls] *= scale
        normalized = {}
        total_weight = sum(class_sampling_weights.values())
        for cls, weight in class_sampling_weights.items():
            normalized[cls] = float(weight / total_weight) if total_weight > 0 else 0.0

        sample_weights = [class_sampling_weights[int(y)] for y in train_labels]
        weight_tensor = torch.tensor(sample_weights, dtype=torch.double)
        stats = {
            "min": float(weight_tensor.min()) if len(weight_tensor) else 0.0,
            "max": float(weight_tensor.max()) if len(weight_tensor) else 0.0,
            "mean": float(weight_tensor.mean()) if len(weight_tensor) else 0.0,
        }

        sampler = WeightedRandomSampler(
            weights=weight_tensor,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_dataset = Subset(ds_train, valid_train_indices)

        print("Using WeightedRandomSampler")
        print(f"  sampler_use_sqrt_inv={sampler_use_sqrt_inv}, cap_ratio={sampler_cap_ratio}")
        print(f"  Class counts: {dict(sorted(label_counts.items()))}")
        print(f"  Effective train samples={len(train_dataset)} (ignored={len(ds_train)-len(train_dataset)})")
        print(f"  Sampling weights stats: {stats}")
        print(f"  Per-class sampling weights (normalized): {normalized}")
        print(f"  Sample weights (first 10): {weight_tensor[:10].tolist()}")

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

    model_cfg = getattr(cfg, "model", {})
    width_mult_cfg = float(getattr(model_cfg, "width_mult", 1.0))
    depth_mult_cfg = float(getattr(model_cfg, "depth_mult", 1.0))
    use_xception_mid_cfg = bool(getattr(model_cfg, "use_xception_mid", False))
    use_late_mhsa_cfg = bool(getattr(model_cfg, "use_late_mhsa", False))
    mhsa_layers_cfg = int(getattr(model_cfg, "mhsa_layers", 1))
    mhsa_heads_cfg = int(getattr(model_cfg, "mhsa_heads", 4))
    use_gem_cfg = bool(getattr(model_cfg, "use_gem", False))
    head_mlp_cfg = bool(getattr(model_cfg, "head_mlp", False))
    head_hidden_dim_cfg = getattr(model_cfg, "head_hidden_dim", None)
    head_dropout_cfg = float(getattr(model_cfg, "head_dropout", 0.0))

    def _build_model_instance():
        return IVF_EffiMorphPP(
            num_classes=num_classes,
            dropout_p=dropout_p_value,
            width_mult=width_mult_cfg,
            depth_mult=depth_mult_cfg,
            task=task,
            use_coral=use_coral,
            use_xception_mid=use_xception_mid_cfg,
            use_late_mhsa=use_late_mhsa_cfg,
            mhsa_layers=mhsa_layers_cfg,
            mhsa_heads=mhsa_heads_cfg,
            use_gem=use_gem_cfg,
            head_mlp=head_mlp_cfg,
            head_hidden_dim=head_hidden_dim_cfg,
            head_dropout=head_dropout_cfg,
        )

    # Sanity model scale preset
    if args.sanity_overfit:
        scale_mapping = {"small": 1.0, "base": 1.25, "large": 1.5}
        width_mult = scale_mapping[args.sanity_model_scale]
        # Auto-bump to 2.0 if params < 3M
        temp_model = IVF_EffiMorphPP(
            num_classes=num_classes,
            dropout_p=dropout_p_value,
            width_mult=width_mult,
            depth_mult=depth_mult_cfg,
            task=task,
            use_coral=use_coral,
            use_xception_mid=use_xception_mid_cfg,
            use_late_mhsa=use_late_mhsa_cfg,
            mhsa_layers=mhsa_layers_cfg,
            mhsa_heads=mhsa_heads_cfg,
            use_gem=use_gem_cfg,
            head_mlp=head_mlp_cfg,
            head_hidden_dim=head_hidden_dim_cfg,
            head_dropout=head_dropout_cfg,
        )
        total_params = sum(p.numel() for p in temp_model.parameters())
        if total_params < 3_000_000 and args.sanity_model_scale == "large":
            width_mult = 2.0
        model = IVF_EffiMorphPP(num_classes=num_classes, dropout_p=dropout_p_value, width_mult=width_mult, task=task, use_coral=use_coral)
    else:
        model = _build_model_instance()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load pretrained backbone if provided
    if args.pretrain_ckpt:
        model = load_backbone_only(model, args.pretrain_ckpt, device=device)
    
    print(f"Device: {device}")
    if args.sanity_overfit:
        print(f"[SANITY MODE] Using dropout_p=0.0 for overfitting test")

    # Safety check for CORAL
    if use_coral and task == "exp":
        # Test forward pass to check output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_output = model(dummy_input)
            assert dummy_output.shape[1] == 4, f"Expected 4 CORAL logits for EXP, got {dummy_output.shape[1]}"
        print(f"[CORAL] Model outputs {dummy_output.shape[1]} logits for EXP ordinal regression")

    # Dataloaders
    batch_size = int(cfg.train.batch_size)
    num_workers = int(cfg.data.num_workers) if "data" in cfg else 4

    dl_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler if use_weighted_sampler else None,
        shuffle=False if use_weighted_sampler else True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # Print model summary
    print("\n" + "="*80)
    print("Model Summary")
    print("="*80)
    
    # Always show parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable params: {trainable_params:,}")
    
    # Try to compute FLOPs
    try:
        from fvcore.nn import FlopCounterMode
        with FlopCounterMode(model) as fcm:
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            _ = model(dummy_input)
        flops = fcm.flop_counts.get("Global", 0)
        print(f"FLOPs (per sample): {flops:,}")
    except ImportError:
        # Estimate FLOPs: roughly 2 * params for inference
        flops_estimate = total_params * 2
        print(f"FLOPs (estimated): {flops_estimate:,}")
    except Exception as e:
        pass
    
    # Try torchinfo if available
    if torchinfo_summary is not None:
        try:
            print("\nDetailed summary:")
            torchinfo_summary(model, input_size=(1, 3, 224, 224), device=device, verbose=0)
        except Exception as e:
            pass
    
    print("="*80 + "\n")

    # Loss
    loss_fn, loss_meta = make_loss_fn(
        track=track,
        task=task,
        num_classes=num_classes,
        use_class_weights=use_class_weights,
        train_labels=train_labels,
        sanity_mode=sanity_mode,
        use_weighted_sampler=use_weighted_sampler,
        use_coral=use_coral,
        label_smoothing=label_smoothing_cfg,
        loss_cfg=loss_cfg,
        class_weight_mode=class_weight_mode_cfg,
        class_weight_beta=class_weight_beta_cfg,
        class_weight_cap=class_weight_cap_cfg,
    )
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn.to(device)
    class_weights_tensor = loss_meta.get("class_weights_tensor")
    monitor_metric_label = getattr(cfg, "monitor_label", None)
    monitor_mode = getattr(cfg, "monitor_mode", "max") or "max"
    monitor_patience = getattr(cfg, "monitor_patience", 10) or 10
    monitor_source = getattr(cfg, "monitor_source", "default") or "default"
    if monitor_metric_label is None:
        monitor_metric_label = "val_macro_f1" if task in {"icm", "te"} else "val_acc"
    monitor_metric_key = _monitor_key_from_label(monitor_metric_label)
    monitor_mode = str(monitor_mode).lower()
    if monitor_mode not in {"min", "max"}:
        monitor_mode = "max"
    monitor_patience = max(0, int(monitor_patience))
    print(
        f"[MONITOR] task={task} resolved monitor={monitor_metric_label} "
        f"key={monitor_metric_key} mode={monitor_mode} patience={monitor_patience} "
        f"source={monitor_source}"
    )
    if task in {"icm", "te"}:
        print(
            f"[ICM/TE] monitor={monitor_metric_label} (key={monitor_metric_key}) "
            f"source={monitor_source} sampler={use_weighted_sampler} "
            f"(sqrt_inv={sampler_use_sqrt_inv}, cap_ratio={sampler_cap_ratio}) "
            f"use_class_weights={use_class_weights} "
            f"loss={loss_meta.get('loss_name','cross_entropy')} "
            f"focal_gamma={loss_meta.get('focal_gamma')} "
            f"class_weights={loss_meta.get('class_weights')}"
        )

    # Optimizer / scheduler
    optimizer_cfg = cfg.optimizer
    scheduler_cfg = getattr(cfg, "scheduler", {})
    swa_cfg = getattr(cfg, "swa", {})
    coral_cfg = getattr(cfg, "coral_tuning", {})
    lr = float(optimizer_cfg.lr)
    wd = float(optimizer_cfg.weight_decay)
    opt_name = getattr(optimizer_cfg, 'name', 'adam').lower()
    
    if opt_name == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        print(f"Using AdamW optimizer: lr={lr}, weight_decay={wd}")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        print(f"Using Adam optimizer: lr={lr}, weight_decay={wd}")
    if use_coral and task == "exp":
        print("Optimizer=Adam wd=0.0 | label_smoothing=0.0 | loss=CORAL")
    
    epochs = int(cfg.train.epochs)
    warmup_epochs = int(getattr(scheduler_cfg, 'warmup_epochs', 0))
    warmup_lr = float(getattr(scheduler_cfg, 'warmup_lr', 0.0))
    if task in {"icm", "te"}:
        warmup_epochs = 0
        warmup_lr = 0.0
    warmup_scale = warmup_lr / lr if lr > 0 else 0.0
    warmup_scale = min(max(warmup_scale, 0.0), 1.0)
    use_swa = bool(getattr(swa_cfg, "use", False))
    if task in {"icm", "te"} and use_swa:
        print("[INFO] Disabling SWA for ICM/TE training.")
        use_swa = False
    swa_start_epoch = int(getattr(swa_cfg, "start_epoch", epochs + 1))
    swa_start_epoch = max(1, min(swa_start_epoch, epochs))
    swa_lr = float(getattr(swa_cfg, "lr", lr))
    tune_coral_thr = bool(getattr(coral_cfg, "tune", False))
    thr_min = float(getattr(coral_cfg, "thr_min", 0.30))
    thr_max = float(getattr(coral_cfg, "thr_max", 0.70))
    thr_step = float(getattr(coral_cfg, "thr_step", 0.01))
    print(f"LR schedule: cosine with warmup_epochs={warmup_epochs} warmup_lr={warmup_lr:.6f}")
    print(f"SWA: use={use_swa} start_epoch={swa_start_epoch} swa_lr={swa_lr}")
    if task == "exp" and use_coral:
        print(f"CORAL tuning: enabled={tune_coral_thr} grid=[{thr_min},{thr_max}] step={thr_step}")

    best_metric: Optional[float] = None
    best_epoch = 0
    best_path = out_dir / "best.ckpt"
    metrics_val_path = out_dir / "metrics_val.json"
    current_coral_thresholds = list(DEFAULT_CORAL_THRESHOLDS)
    epochs_without_improvement = 0

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
        model = _build_model_instance()
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
        if warmup_steps > 0 and step < warmup_steps:
            progress = float(step) / float(max(1, warmup_steps))
            return warmup_scale + (1.0 - warmup_scale) * progress
        denom = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    # Create a lambda scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=get_lr_scale)
    swa_model = AveragedModel(model) if use_swa else None
    swa_scheduler = SWALR(opt, swa_lr=swa_lr) if use_swa else None
    if use_swa:
        swa_model.to(device)
        print(f"[SWA] AveragedModel ready | swa_lr={swa_lr} | start_epoch={swa_start_epoch}")

    collapse_dominant_class = None
    collapse_run_len = 0
    zero_val_recall_runs = defaultdict(int)
    # Training loop (simple)
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        scheduler_active = not (use_swa and epoch >= swa_start_epoch)
        train_pred_counts = Counter()
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
            if epoch <= 5:
                preds = logits.argmax(dim=1)
                valid_mask = y != IGNORE_INDEX
                if valid_mask.any():
                    valid_preds = preds[valid_mask].detach().cpu().tolist()
                    train_pred_counts.update(valid_preds)
            if scheduler_active:
                scheduler.step()  # Update LR after each batch
            running += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = running / max(n, 1)
        if epoch <= 5:
            print(f"  Train y_pred counts: {dict(sorted(train_pred_counts.items()))}")

        # val
        precomputed_logits = None
        precomputed_labels = None
        if use_coral and tune_coral_thr:
            precomputed_logits, precomputed_labels = collect_coral_logits(model, dl_val, device)
            epoch_tune = tune_coral_threshold(
                model,
                dl_val,
                device,
                thr_min,
                thr_max,
                thr_step,
                num_classes=num_classes,
                logits=precomputed_logits,
                labels=precomputed_labels,
            )
            if epoch_tune is not None:
                current_coral_thresholds = epoch_tune["best_thr"]

        val_metrics = evaluate_on_val(
            model,
            dl_val,
            device,
            task,
            epoch,
            use_coral,
            num_classes,
            compute_confusion_matrix=compute_confusion_matrix,
            coral_thresholds=current_coral_thresholds if use_coral else None,
            precomputed_logits=precomputed_logits,
            precomputed_labels=precomputed_labels,
        )
        monitor = val_metrics[monitor_metric_key]
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"val_avg_prec={val_metrics['avg_precision_weighted']:.4f} "
            f"val_avg_rec={val_metrics['avg_recall_weighted']:.4f} "
            f"val_avg_f1={val_metrics['avg_f1_weighted']:.4f}"
        )
        if use_coral:
            print(f"  CORAL thresholds in use: {current_coral_thresholds}")
        per_class_recall = val_metrics.get("per_class_recall", [])
        per_class_support = val_metrics.get("per_class_support", [])
        if per_class_recall:
            labels_for_metrics = list(range(len(per_class_recall)))
            recall_summary = ", ".join(
                f"{label}:{rec:.3f}(sup={per_class_support[idx] if idx < len(per_class_support) else 0})"
                for idx, (label, rec) in enumerate(zip(labels_for_metrics, per_class_recall))
            )
            print(f"  Val per-class recall: {recall_summary}")
            if task not in {"icm", "te"}:
                y_pred_counts_full = {
                    cls: val_metrics["y_pred_counts"].get(cls, 0) for cls in labels_for_metrics
                }
                y_true_counts_full = {
                    cls: val_metrics["y_true_counts"].get(cls, 0) for cls in labels_for_metrics
                }
                print(f"  Val y_pred counts: {y_pred_counts_full}")
                print(f"  Val y_true counts: {y_true_counts_full}")
        if compute_confusion_matrix:
            cm = val_metrics.get("confusion_matrix")
            cm_labels = val_metrics.get("confusion_matrix_labels", [])
            if cm is not None:
                label_order = cm_labels if cm_labels else list(range(len(cm)))
                print(f"  Val confusion matrix (labels={label_order}):")
                for lbl, row in zip(label_order, cm):
                    print(f"    {lbl}: {row}")
        if task == "icm":
            per_class_recall_icm = val_metrics.get("per_class_recall", [])
            if per_class_recall_icm:
                per_class_support = val_metrics.get("per_class_support", [])
                for cls_idx, rec in enumerate(per_class_recall_icm):
                    support = per_class_support[cls_idx] if cls_idx < len(per_class_support) else 0
                    skip_zero = cls_idx == 2 and support < 5
                    if rec == 0 and not skip_zero:
                        zero_val_recall_runs[cls_idx] += 1
                        if zero_val_recall_runs[cls_idx] >= 3:
                            print(f"[WARN] Val recall for class {cls_idx} stayed at zero for {zero_val_recall_runs[cls_idx]} epochs.")
                    else:
                        zero_val_recall_runs[cls_idx] = 0
            y_pred_counts = val_metrics.get("y_pred_counts", {})
            total_preds = sum(y_pred_counts.values())
            if total_preds:
                dominant_cls = max(y_pred_counts, key=y_pred_counts.get)
                dominant_ratio = y_pred_counts[dominant_cls] / total_preds
                if dominant_ratio > 0.9:
                    if dominant_cls == collapse_dominant_class:
                        collapse_run_len += 1
                    else:
                        collapse_dominant_class = dominant_cls
                        collapse_run_len = 1
                else:
                    collapse_dominant_class = None
                    collapse_run_len = 0
                if collapse_run_len >= 3:
                    print(
                        "[WARN] Val predictions dominated by class "
                        f"{collapse_dominant_class} ({dominant_ratio:.0%}) for "
                        f"{collapse_run_len} consecutive epochs."
                    )
            else:
                collapse_dominant_class = None
                collapse_run_len = 0
        if task in {"icm", "te"}:
            class_range = range(num_classes)
            y_pred_counts_full = {
                cls: val_metrics["y_pred_counts"].get(cls, 0) for cls in class_range
            }
            y_true_counts_full = {
                cls: val_metrics["y_true_counts"].get(cls, 0) for cls in class_range
            }
            print(f"  Val y_pred counts: {y_pred_counts_full}")
            print(f"  Val y_true counts: {y_true_counts_full}")

        # save best
        improved = _has_monitor_improvement(monitor, best_metric, monitor_mode)
        if improved:
            best_metric = monitor
            best_epoch = epoch
            epochs_without_improvement = 0
            ckpt_payload = {
                "state_dict": model.state_dict(),
                "task": task,
                "track": track,
                "num_classes": num_classes,
                "label_col": label_col,
            }
            ckpt_payload.update(merge_info)
            torch.save(ckpt_payload, best_path)
            with open(metrics_val_path, "w", encoding="utf-8") as f:
                json.dump({"best_epoch": epoch, "best_val": val_metrics}, f, indent=2)
            print(
                f"  [BEST] Saved best to {best_path} "
                f"(monitor={monitor_metric_label}={best_metric:.4f})"
            )
        else:
            epochs_without_improvement += 1
            if monitor_patience > 0 and epochs_without_improvement >= monitor_patience:
                print(
                    f"[EARLY STOP] No improvement on {monitor_metric_label} "
                    f"for {epochs_without_improvement} epochs (patience={monitor_patience})."
                )
                break
        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()

    # SWA finalization / CORAL tuning
    swa_ckpt_path = out_dir / "best_swa.ckpt"
    threshold_json_path = out_dir / "best_coral_threshold.json"
    swa_val_metrics = None
    eval_model = swa_model if use_swa else model

    if use_swa:
        print("[SWA] Updating BatchNorm statistics for averaged weights...")
        update_bn(dl_train, swa_model, device=device)
        torch.save({
            "state_dict": swa_model.state_dict(),
            "task": task,
            "track": track,
            "num_classes": num_classes,
            "label_col": label_col,
            "swa": True
        }, swa_ckpt_path)
        print(f"[SWA] Saved averaged checkpoint to {swa_ckpt_path}")
        swa_val_metrics = evaluate_on_val(
            swa_model,
            dl_val,
            device,
            task,
            epoch,
            use_coral,
            num_classes,
            compute_confusion_matrix=compute_confusion_matrix,
            coral_thresholds=current_coral_thresholds if use_coral else None,
        )
        print(f"[SWA VAL] acc={swa_val_metrics['acc']:.4f} macro_f1={swa_val_metrics['macro_f1']:.4f} weighted_f1={swa_val_metrics['weighted_f1']:.4f}")

    threshold_result = None
    if task == "exp" and use_coral and tune_coral_thr:
        threshold_result = tune_coral_threshold(eval_model, dl_val, device, thr_min, thr_max, thr_step)
        if threshold_result:
            current_coral_thresholds = threshold_result["best_thr"]
            threshold_payload = {
                "best_coral_thr": threshold_result["best_thr"],
                "best_thr_last": threshold_result.get("best_thr_last"),
                "metric": {
                    "val_f1_macro": threshold_result.get("best_f1_macro"),
                    "val_f1_weighted": threshold_result.get("best_f1_weighted"),
                    "val_acc": threshold_result.get("best_acc"),
                },
                "grid": threshold_result["grid"],
            }
            with open(threshold_json_path, "w", encoding="utf-8") as f:
                json.dump(threshold_payload, f, indent=2)
            print(f"[CORAL TUNING] Saved tuned threshold to {threshold_json_path}")
            print(f"[CORAL TUNING] Using new thresholds: {current_coral_thresholds}")

    # Write debug report
    debug_report_path = out_dir / "debug_report.txt"
    with open(debug_report_path, "w") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(
            f"Best monitor {monitor_metric_label} "
            f"({monitor_metric_key}) = {best_metric:.4f}\n"
        )
        f.write(f"Monitor mode/patience: {monitor_mode}/{monitor_patience}\n")
        f.write(f"Epochs without improvement at stop: {epochs_without_improvement}\n")
        f.write(f"y_pred distribution at best epoch: {val_metrics['y_pred_counts']}\n")
        f.write(f"Confusion matrix at best epoch: {val_metrics.get('confusion_matrix', 'N/A')}\n")
        f.write(f"Class weights applied: {use_class_weights}\n")
        if use_class_weights and class_weights_tensor is not None:
            f.write(f"Class weights tensor: {class_weights_tensor.tolist()}\n")
        f.write(f"WeightedRandomSampler used: {use_weighted_sampler}\n")
        f.write(f"Current LR value: {opt.param_groups[0]['lr']:.6f}\n")
        if swa_val_metrics is not None:
            f.write(f"SWA val metrics: acc={swa_val_metrics['acc']:.4f} macro_f1={swa_val_metrics['macro_f1']:.4f} weighted_f1={swa_val_metrics['weighted_f1']:.4f}\n")
        if threshold_result is not None:
            f.write(f"Best tuned coral threshold: {threshold_result['best_thr']:.4f}\n")
            f.write(f"Threshold f1/acc: {threshold_result['best_f1']:.4f}/{threshold_result['best_acc']:.4f}\n")

        # Check for majority-class collapse
        max_ratio = max(val_metrics['y_pred_ratios'].values())
        if max_ratio > 0.8:
            f.write("Likely majority-class collapse due to imbalance. Recommend enabling WeightedRandomSampler or stronger class reweighting; reduce label_smoothing; reduce warmup.\n")

    print(
        f"\nDone. Best monitor '{monitor_metric_label}' "
        f"({monitor_metric_key}) = {best_metric:.4f}"
    )
    print(f"Artifacts: {out_dir}")
    print(f"Debug report: {debug_report_path}")
    if use_swa:
        print(f"SWA checkpoint: {swa_ckpt_path}")


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
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--use_class_weights", type=int, default=None)
    parser.add_argument("--use_weighted_sampler", type=int, default=None, help="Use WeightedRandomSampler for training (0=OFF, 1=ON)")
    parser.add_argument("--use_coral", type=int, default=0, help="Use CORAL ordinal regression for EXP task (0=OFF, 1=ON)")
    parser.add_argument("--icm_use_focal", type=int, choices=[0, 1], default=0,
                        help="Use focal loss for ICM/TE runs when track=improved (default 0).")
    parser.add_argument("--sampler_use_sqrt_inv", type=int, choices=[0, 1], default=None,
                        help="Override sampling weights to use sqrt inverse frequency (1=ON, 0=OFF).")
    parser.add_argument("--sampler_cap_ratio", type=float, default=None,
                        help="Maximum ratio between largest and smallest sampling weight.")
    parser.add_argument("--icm_merge_class2_to", type=int, choices=[1, 3], default=None,
                        help="Merge ICM class2 into class 1 or 3 during training (reduces num_classes).")
    parser.add_argument("--train_icmte_exclude_na_nd", type=int, choices=[0, 1], default=0,
                        help="Drop class 3 (NA/ND) from the ICM/TE training split when enabled.")
    parser.add_argument("--monitor_metric", type=str, default=None,
                        help="Metric key to monitor for best checkpoint (default weighted F1 for ICM/TE, macro F1 for EXP).")

    parser.add_argument("--lr", type=float, default=None, help="Base learning rate (paper default 5e-4).")
    parser.add_argument("--warmup_lr", type=float, default=None, help="Warmup learning rate (paper default 1e-6).")
    parser.add_argument("--warmup_epochs", type=int, default=None, help="Warmup epochs (paper default 5).")
    parser.add_argument("--use_swa", type=int, choices=[0, 1], default=None, help="Enable SWA training (paper default 1).")
    parser.add_argument("--swa_start_epoch", type=int, default=None, help="Epoch to start SWA averaging (paper default 30).")
    parser.add_argument("--swa_lr", type=float, default=None, help="SWA learning rate (paper default 2.5e-4).")
    parser.add_argument("--tune_coral_thr", type=int, choices=[0, 1], default=None, help="Tune CORAL threshold on val (paper default 1).")
    parser.add_argument("--thr_min", type=float, default=None, help="Minimum CORAL threshold for tuning (paper default 0.30).")
    parser.add_argument("--thr_max", type=float, default=None, help="Maximum CORAL threshold for tuning (paper default 0.70).")
    parser.add_argument("--thr_step", type=float, default=None, help="CORAL threshold tuning step size (paper default 0.01).")
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

    parser.add_argument("--pretrain_ckpt", type=str, default="",
                        help="Path to stage-pretrain best.ckpt for backbone initialization.")

    args = parser.parse_args()

    cfg_base = OmegaConf.load(args.config)
    cfg_task = OmegaConf.load(args.task_cfg)
    cfg_track = OmegaConf.load(args.track_cfg)
    cfg = OmegaConf.merge(cfg_base, cfg_task, cfg_track)

    if "scheduler" not in cfg:
        cfg.scheduler = OmegaConf.create({})
    if "swa" not in cfg:
        cfg.swa = OmegaConf.create({})
    if "coral_tuning" not in cfg:
        cfg.coral_tuning = OmegaConf.create({})

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
    if args.num_workers is not None:
        if "data" not in cfg:
            cfg.data = {}
        cfg.data.num_workers = args.num_workers
    if args.use_class_weights is not None:
        cfg.use_class_weights = bool(args.use_class_weights)
    if args.use_weighted_sampler is not None:
        cfg.use_weighted_sampler = bool(args.use_weighted_sampler)
    monitor_label = args.monitor_metric
    monitor_mode = None
    monitor_patience = None
    monitor_source = "cli" if monitor_label else None
    if monitor_label is None:
        monitor_label, monitor_mode, monitor_patience, monitor_source = _resolve_monitor(
            cfg.task, cfg_base, cfg_track, cfg_task
        )
    else:
        monitor_mode = "max"
        monitor_patience = 10
    cfg.monitor_label = monitor_label
    cfg.monitor_mode = monitor_mode
    cfg.monitor_patience = monitor_patience
    cfg.monitor_source = monitor_source or "default"
    cfg.train_icmte_exclude_na_nd = bool(getattr(cfg, "train_icmte_exclude_na_nd", False))
    if args.train_icmte_exclude_na_nd is not None:
        cfg.train_icmte_exclude_na_nd = bool(args.train_icmte_exclude_na_nd)
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    if args.warmup_lr is not None:
        cfg.scheduler.warmup_lr = args.warmup_lr
    if args.warmup_epochs is not None:
        cfg.scheduler.warmup_epochs = args.warmup_epochs
    if args.use_swa is not None:
        cfg.swa.use = bool(args.use_swa)
    if args.swa_start_epoch is not None:
        cfg.swa.start_epoch = args.swa_start_epoch
    if args.swa_lr is not None:
        cfg.swa.lr = args.swa_lr
    if args.tune_coral_thr is not None:
        cfg.coral_tuning.tune = bool(args.tune_coral_thr)
    if args.thr_min is not None:
        cfg.coral_tuning.thr_min = args.thr_min
    if args.thr_max is not None:
        cfg.coral_tuning.thr_max = args.thr_max
    if args.thr_step is not None:
        cfg.coral_tuning.thr_step = args.thr_step

    # Sanity: task/track must exist
    if "task" not in cfg or "track" not in cfg:
        raise ValueError("Config merge failed: missing `task` or `track` fields. Check YAMLs.")

    set_seed(int(cfg.seed), bool(cfg.deterministic))
    print(f"Task={cfg.task} | Track={cfg.track}")
    print(f"Splits dir={cfg.splits_dir} | Out dir={cfg.out_dir}")

    train_one_run(cfg, args)


if __name__ == "__main__":
    main()
