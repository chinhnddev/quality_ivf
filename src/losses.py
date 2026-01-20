# src/losses.py
"""
Loss utilities for Gardner EXP / ICM / TE experiments.

Supported:
- CrossEntropy (optionally with class weights)
- CrossEntropy with label smoothing (EXP improved)
- Focal Loss (ICM/TE improved)
- Class weight computation from TRAIN split only

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
# Class weight utilities
# ------------------------------------------------------------

def compute_class_weights(
    labels: Iterable[int],
    num_classes: int,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute inverse-frequency normalized class weights:
        w_c = N_total / (K * N_c)

    Args:
        labels: iterable of integer labels AFTER filtering valid samples
        num_classes: number of classes (EXP=5, ICM/TE=3)
        eps: numerical stability

    Returns:
        torch.Tensor of shape (num_classes,)
    """
    counts = np.zeros(num_classes, dtype=np.float64)

    for y in labels:
        if 0 <= int(y) < num_classes:
            counts[int(y)] += 1

    total = counts.sum()
    K = float(num_classes)

    weights = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (K * (counts[c] + eps))
        else:
            # Missing class in training set
            # Set weight to 0 and warn upstream
            weights[c] = 0.0

    return torch.tensor(weights, dtype=torch.float32)


# ------------------------------------------------------------
# Focal loss
# ------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Multi-class focal loss:
        FL = (1 - p_t)^gamma * CE

    Works on logits directly (no softmax needed).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C)
            targets: (N,)
        """
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none"
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


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

    Args:
        track: "benchmark_fair" or "improved"
        task: "exp", "icm", or "te"
        num_classes: 5 for EXP, 3 for ICM/TE
        train_labels: labels from TRAIN split AFTER filtering valid samples
        use_class_weights: whether to compute and use class weights
        label_smoothing: used for EXP in improved track
        focal_gamma: used for ICM/TE in improved track

    Returns:
        nn.Module loss function
    """

    weight = None
    if use_class_weights:
        weight = compute_class_weights(train_labels, num_classes)
        if (weight == 0).any():
            missing = [i for i, w in enumerate(weight.tolist()) if w == 0.0]
            print(
                f"[WARN][losses] Missing classes in TRAIN for task={task}: {missing}. "
                f"Corresponding class weights set to 0."
            )

    # -------------------------
    # Benchmark-fair
    # -------------------------
    if track == "benchmark_fair":
        # Same CE for all tasks, no tricks
        return nn.CrossEntropyLoss(weight=weight)

    # -------------------------
    # Improved
    # -------------------------
    if track == "improved":
        if task == "exp":
            # EXP: CE + label smoothing
            return nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=label_smoothing
            )
        else:
            # ICM / TE: Focal loss (+ optional weights)
            return FocalLoss(
                gamma=focal_gamma,
                weight=weight
            )

    raise ValueError(f"Unknown track: {track}")
