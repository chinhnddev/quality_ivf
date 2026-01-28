"""
CORAL (Consistent Rank Logits) implementation for ordinal regression.
Used for EXP task ordinal classification (0,1,2,3,4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional


def coral_encode_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Encode ordinal targets for CORAL loss.
    For ordinal regression with K classes (0 to K-1), we use K-1 thresholds.

    Args:
        y: (B,) tensor of class labels in [0, K-1]
        num_classes: K (total number of classes)

    Returns:
        targets: (B, K-1) binary tensor where targets[b,k] = 1 if y[b] > k else 0
    """
    assert num_classes >= 2, "num_classes must be >= 2"
    device = y.device
    targets = torch.zeros(y.shape[0], num_classes - 1, device=device, dtype=torch.float32)

    for k in range(num_classes - 1):
        targets[:, k] = (y > k).float()

    return targets


def coral_loss(logits: torch.Tensor, y: torch.Tensor, num_classes: int, reduction: str = "mean") -> torch.Tensor:
    """
    CORAL loss for ordinal regression.

    Args:
        logits: (B, K-1) CORAL logits for K-1 thresholds
        y: (B,) class labels in [0, K-1]
        num_classes: K (total number of classes)
        reduction: "mean" or "sum"

    Returns:
        loss: scalar tensor
    """
    assert logits.shape[1] == num_classes - 1, f"Expected {num_classes-1} logits, got {logits.shape[1]}"

    targets = coral_encode_targets(y, num_classes)

    # BCE loss over all thresholds
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # Average over thresholds and batch
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def coral_predict_class(
    logits: torch.Tensor,
    thresholds: Union[float, List[float]] = 0.5
) -> torch.Tensor:
    """
    Convert CORAL logits to class predictions.

    Args:
        logits: Tensor of shape (B, K-1) containing raw logits for K-1 cumulative thresholds
        thresholds: Either a single threshold or a list of K-1 thresholds

    Returns:
        Tensor of shape (B,) containing predicted class indices (0 to K-1)
    """
    probs = torch.sigmoid(logits)  # (B, K-1)

    if isinstance(thresholds, (int, float)):
        # Uniform threshold
        preds = (probs > thresholds).sum(dim=1)  # Count how many thresholds exceeded
    else:
        # Per-threshold values
        thresholds_tensor = torch.tensor(thresholds, device=logits.device, dtype=logits.dtype)
        preds = (probs > thresholds_tensor.unsqueeze(0)).sum(dim=1)

    return preds.long()


class CoralLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) loss for ordinal regression.

    Paper: "Rank consistent ordinal regression for neural networks with application to age estimation"
    https://arxiv.org/abs/1901.07884

    For K classes (0, 1, ..., K-1), the model outputs K-1 logits.
    Each logit l_k represents P(Y > k | X) for k = 0, 1, ..., K-2
    """

    def __init__(self, num_classes: int, weight: Optional[torch.Tensor] = None):
        """
        Args:
            num_classes: Number of ordinal classes K
            weight: Optional class weights of shape (K,) for weighted loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.register_buffer('weight', weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predictions of shape (B, K-1)
            targets: Ground truth labels of shape (B,) with values in {0, 1, ..., K-1}

        Returns:
            Scalar loss value
        """
        batch_size = logits.size(0)

        # Create binary labels for each threshold
        # For target y and threshold k: label is 1 if y > k, else 0
        # Shape: (B, K-1)
        levels = torch.arange(self.num_thresholds, device=logits.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) > levels).float()

        # Binary cross entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction='none'
        )  # (B, K-1)

        # Apply class weights if provided
        if self.weight is not None:
            # Weight by the target class
            sample_weights = self.weight[targets]  # (B,)
            loss = loss * sample_weights.unsqueeze(1)

        return loss.mean()


class CoralLossWithImportance(nn.Module):
    """
    CORAL loss with importance weighting for different thresholds.
    Allows giving more weight to certain ordinal boundaries.
    """

    def __init__(
        self,
        num_classes: int,
        threshold_weights: Optional[List[float]] = None,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        if threshold_weights is None:
            threshold_weights = [1.0] * self.num_thresholds

        self.register_buffer(
            'threshold_weights',
            torch.tensor(threshold_weights, dtype=torch.float32)
        )
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)

        levels = torch.arange(self.num_thresholds, device=logits.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) > levels).float()

        loss = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction='none'
        )  # (B, K-1)

        # Apply threshold importance weights
        loss = loss * self.threshold_weights.unsqueeze(0)

        # Apply class weights if provided
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]
            loss = loss * sample_weights.unsqueeze(1)

        return loss.mean()
