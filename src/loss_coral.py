"""
CORAL (Consistent Rank Logits) implementation for ordinal regression.
Used for EXP task ordinal classification (0,1,2,3,4).

Paper: https://arxiv.org/abs/1901.07884
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
    assert num_classes >= 2, f"num_classes must be >= 2, got {num_classes}"
    assert y.min() >= 0 and y.max() < num_classes, \
        f"Labels must be in [0, {num_classes-1}], got range [{y.min()}, {y.max()}]"
    
    device = y.device
    num_thresholds = num_classes - 1
    
    # Vectorized version (faster than loop)
    levels = torch.arange(num_thresholds, device=device).unsqueeze(0)  # (1, K-1)
    targets = (y.unsqueeze(1) > levels).float()  # (B, K-1)
    
    return targets


def coral_predict_class(
    logits: torch.Tensor,
    thresholds: Union[float, List[float]] = 0.5
) -> torch.Tensor:
    """
    Convert CORAL logits to class predictions.

    Args:
        logits: (B, K-1) raw logits for K-1 cumulative thresholds
        thresholds: Either:
            - Single float: uniform threshold for all K-1 logits (default 0.5)
            - List of K-1 floats: per-threshold values

    Returns:
        (B,) tensor of predicted class indices in [0, K-1]
    """
    probs = torch.sigmoid(logits)  # (B, K-1)

    if isinstance(thresholds, (int, float)):
        # Uniform threshold
        exceeds = (probs > thresholds).sum(dim=1)
    else:
        # Per-threshold values
        thresholds_tensor = torch.as_tensor(
            thresholds, device=logits.device, dtype=logits.dtype
        )
        assert thresholds_tensor.shape[0] == logits.shape[1], \
            f"Threshold count ({len(thresholds)}) must match logits dim ({logits.shape[1]})"
        exceeds = (probs > thresholds_tensor.unsqueeze(0)).sum(dim=1)

    return exceeds.long()


class CoralLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) loss for ordinal regression.

    For K classes (0, 1, ..., K-1), the model outputs K-1 logits.
    Each logit l_k represents P(Y > k | X) for k = 0, 1, ..., K-2
    
    Paper: https://arxiv.org/abs/1901.07884
    """

    def __init__(
        self, 
        num_classes: int, 
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        """
        Args:
            num_classes: Number of ordinal classes K
            class_weights: Optional tensor of shape (K,) for per-sample weighting
            reduction: "mean" | "sum" | "none"
        """
        super().__init__()
        assert num_classes >= 2, f"num_classes must be >= 2, got {num_classes}"
        
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.reduction = reduction
        
        if class_weights is not None:
            assert class_weights.shape[0] == num_classes, \
                f"class_weights shape {class_weights.shape} != num_classes {num_classes}"
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, K-1) predictions
            targets: (B,) ground truth labels in {0, 1, ..., K-1}

        Returns:
            Scalar loss value (if reduction="mean"|"sum") or (B, K-1) if "none"
        """
        # Validate shapes
        assert logits.dim() == 2, f"logits must be 2D, got {logits.dim()}D"
        assert logits.shape[1] == self.num_thresholds, \
            f"Expected {self.num_thresholds} logits, got {logits.shape[1]}"
        assert targets.dim() == 1, f"targets must be 1D, got {targets.dim()}D"
        assert logits.shape[0] == targets.shape[0], \
            f"Batch size mismatch: logits {logits.shape[0]} vs targets {targets.shape[0]}"

        # Create binary labels for each threshold
        binary_targets = coral_encode_targets(targets, self.num_classes)  # (B, K-1)

        # Binary cross entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction='none'
        )  # (B, K-1)

        # Apply class weights if provided
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]  # (B,)
            loss = loss * sample_weights.unsqueeze(1)  # (B, K-1)

        # Reduce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}, reduction={self.reduction}"


def coral_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    num_classes: int, 
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Functional interface for CORAL loss.
    
    Args:
        logits: (B, K-1) CORAL logits for K-1 thresholds
        targets: (B,) class labels in [0, K-1]
        num_classes: K (total number of classes)
        reduction: "mean" | "sum" | "none"

    Returns:
        loss: scalar tensor (if reduction != "none")
    """
    loss_fn = CoralLoss(num_classes=num_classes, reduction=reduction)
    return loss_fn(logits, targets)


def coral_loss_masked(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    num_classes: int, 
    ignore_index: int = -1
) -> torch.Tensor:
    """
    CORAL loss with masking for ignored labels.

    Args:
        logits: (B, K-1) CORAL logits
        targets: (B,) class labels, with ignore_index for masked positions
        num_classes: K
        ignore_index: value to ignore (default -1)

    Returns:
        loss: scalar tensor (mean over valid positions)
    """
    valid_mask = (targets != ignore_index)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits_valid = logits[valid_mask]
    targets_valid = targets[valid_mask]

    return coral_loss(logits_valid, targets_valid, num_classes, reduction="mean")