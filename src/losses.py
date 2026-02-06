"""
Loss utilities for Gardner EXP / ICM / TE experiments.

Supported:
- CrossEntropyLoss (optionally with class weights)
- CrossEntropyLoss with label smoothing (EXP improved)
- Focal Loss with alpha (ICM/TE improved)
- Effective Number of Samples weighting (better for extreme imbalance)

IMPORTANT RULES
- Class weights MUST be computed from TRAIN distribution only.
- For ICM/TE, compute weights only over valid labels {0,1,2}.
- No hard-coded class weights.
"""

from typing import Iterable, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Class weight utilities - V1 (Original for EXP)
# ------------------------------------------------------------
def compute_class_weights_v1(
    labels: Iterable[int],
    num_classes: int,
    eps: float = 1e-8,
    beta: float = 0.9999
) -> torch.Tensor:
    """
    Original version for EXP task (with normalization).
    
    Compute class weights using Effective Number of Samples (Cui et al. 2019):
        effective_num = 1 - beta^n_c
        w_c = (1 - beta) / (effective_num + eps)
    Normalize to sum ~ num_classes, boost missing classes lightly if needed.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for y in labels:
        if 0 <= int(y) < num_classes:
            counts[int(y)] += 1

    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / (effective_num + eps)

    # Normalize weights to sum to num_classes (KEEP FOR EXP)
    sum_w = weights.sum()
    if sum_w > 0:
        weights = weights / sum_w * num_classes
    else:
        weights = np.ones(num_classes)  # fallback (unlikely)

    # Boost missing classes lightly to avoid complete ignore
    missing = counts == 0
    if missing.any():
        valid_weights = weights[~missing]
        if len(valid_weights) > 0:
            mean_valid = valid_weights.mean()
            weights[missing] = mean_valid * 1.5
        else:
            weights[missing] = 1.0
        print(f"[INFO] Boosted weights for missing classes: {weights[missing].tolist()}")

    return torch.tensor(weights, dtype=torch.float32)


# ------------------------------------------------------------
# Class weight utilities - V2 (New for ICM/TE)
# ------------------------------------------------------------
def compute_class_weights_v2(
    labels: Iterable[int],
    num_classes: int,
    eps: float = 1e-8,
    beta: float = 0.9999,
    max_ratio: float = 10.0
) -> torch.Tensor:
    """
    New version for ICM/TE tasks (no normalization, with clipping).
    
    Compute class weights using Effective Number of Samples (Cui et al. 2019):
        effective_num = 1 - beta^n_c
        w_c = (1 - beta) / (effective_num + eps)
    
    Key differences from V1:
    - NO normalization (keep raw Effective Number weights)
    - WITH clipping (max_ratio to prevent extreme weights)
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for y in labels:
        if 0 <= int(y) < num_classes:
            counts[int(y)] += 1

    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / (effective_num + eps)

    # NO NORMALIZATION for ICM/TE (keep raw weights)
    print(f"[INFO] Raw Effective Number weights (before clipping): {weights.tolist()}")

    # Boost missing classes lightly to avoid complete ignore
    missing = counts == 0
    if missing.any():
        valid_weights = weights[~missing]
        if len(valid_weights) > 0:
            mean_valid = valid_weights.mean()
            weights[missing] = mean_valid * 1.5
        else:
            weights[missing] = 1.0
        print(f"[INFO] Boosted weights for missing classes: {weights[missing].tolist()}")
    
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # Clip weights to prevent extreme ratios
    if max_ratio > 0:
        valid_mask = weights > 0
        if valid_mask.sum() > 0:
            min_w = weights[valid_mask].min()
            max_w = min_w * max_ratio
            weights_before = weights.clone()
            weights = torch.clamp(weights, max=max_w)
            
            # Log if clipping occurred
            if not torch.allclose(weights, weights_before):
                print(f"[INFO] Clipped weights with max_ratio={max_ratio}:")
                for i in range(len(weights)):
                    if weights[i] != weights_before[i]:
                        print(f"  Class {i}: {weights_before[i].item():.4f} → {weights[i].item():.4f}")
    
    return weights


# Backward compatibility alias (will use v1 by default)
def compute_class_weights(
    labels: Iterable[int],
    num_classes: int,
    eps: float = 1e-8,
    beta: float = 0.9999
) -> torch.Tensor:
    """
    Backward compatibility function - defaults to V1 (original behavior).
    Use compute_class_weights_v1() or compute_class_weights_v2() explicitly.
    """
    return compute_class_weights_v1(labels, num_classes, eps, beta)


# ------------------------------------------------------------
# Focal loss with alpha
# ------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Multi-class focal loss with optional alpha class weights and ignore index.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = 3
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        if alpha is not None and alpha.device != logits.device:
            alpha = alpha.to(logits.device)
            self.alpha = alpha

        ce = F.cross_entropy(
            logits,
            targets,
            weight=alpha,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        pt = torch.exp(-ce)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            mask = targets != self.ignore_index
            if mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device)
            return focal_loss[mask].mean()
        elif self.reduction == "sum":
            mask = targets != self.ignore_index
            return focal_loss[mask].sum()
        else:
            return focal_loss


# ------------------------------------------------------------
# Loss factory
# ------------------------------------------------------------
def get_loss_fn(
    *,
    track: str,
    task: str,
    num_classes: int,
    train_labels: List[int],
    use_class_weights: bool,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0
) -> nn.Module:
    """
    Factory function to build the correct loss for a given task/track.
    
    EXP task uses V1 weights (normalized, stable).
    ICM/TE tasks use V2 weights (raw, clipped).
    """
    weight = None
    if use_class_weights:
        if task == "exp":
            # EXP: Use V1 (original with normalization)
            weight = compute_class_weights_v1(train_labels, num_classes, beta=0.9999)
            print(f"[INFO] EXP: Using V1 weights (normalized, beta=0.9999):")
        else:
            # ICM/TE: Use V2 (no normalization, with clipping)
            weight = compute_class_weights_v2(
                train_labels, 
                num_classes, 
                beta=0.9999, 
                max_ratio=10.0
            )
            print(f"[INFO] {task.upper()}: Using V2 weights (raw, clipped, beta=0.9999):")
        
        # Print weights for all tasks
        for i, w in enumerate(weight.tolist()):
            print(f"  Class {i}: weight={w:.4f}")

        missing = [i for i, w in enumerate(weight.tolist()) if w == 0.0 or np.isnan(w)]
        if missing:
            print(f"[WARN] Missing or NaN classes in TRAIN for task={task}: {missing}")

    # -------------------------
    # Benchmark-fair (same as paper baseline)
    # -------------------------
    if track == "benchmark_fair":
        return nn.CrossEntropyLoss(weight=weight)

    # -------------------------
    # Improved track
    # -------------------------
    if track == "improved":
        if task == "exp":
            # EXP: CE + label smoothing (giá trị từ config, default 0.0)
            return nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=label_smoothing
            )
        else:
            # ICM / TE: Focal + alpha (class-balanced)
            return FocalLoss(
                gamma=focal_gamma,
                alpha=weight,
                reduction="mean",
                ignore_index=3
            )

    raise ValueError(f"Unknown track: {track}")