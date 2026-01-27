"""
CORAL (Consistent Rank Logits) implementation for ordinal regression.
Used for EXP task ordinal classification (0,1,2,3,4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    # k: (1, K-1)
    k = torch.arange(num_classes - 1, device=y.device).unsqueeze(0)

    # y: (B,) -> (B, 1), targets: (B, K-1)
    targets = (y.unsqueeze(1) > k).float()

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


def coral_predict_class(logits: torch.Tensor, thresholds = 0.5) -> torch.Tensor:
    # Convert logits to probabilities
    probs = torch.sigmoid(logits)  # (B, K-1)

    # Handle thresholds: scalar or list
    if isinstance(thresholds, (int, float)):
        # Scalar: broadcast to all K-1 thresholds
        exceeds = (probs >= thresholds).float()
    else:
        # List/tensor of thresholds
        thresholds_tensor = torch.as_tensor(thresholds, device=logits.device, dtype=logits.dtype)
        if thresholds_tensor.dim() == 0:
            # Scalar tensor -> broadcast
            exceeds = (probs > thresholds_tensor).float()
        else:
            # 1D tensor of length K-1
            assert thresholds_tensor.shape[0] == logits.shape[1], \
                f"thresholds length ({thresholds_tensor.shape[0]}) must match logits dim 1 ({logits.shape[1]})"
            exceeds = (probs > thresholds_tensor.unsqueeze(0)).float()

    # Count how many thresholds are exceeded
    # y_pred = sum over k where prob_k > threshold_k
    y_pred = exceeds.sum(dim=1).long()
    y_pred = y_pred.clamp(min=0, max=logits.shape[1])


    return y_pred


def coral_loss_masked(logits: torch.Tensor, y: torch.Tensor, num_classes: int, ignore_index: int = -1) -> torch.Tensor:
    """
    CORAL loss with masking for ignored labels (optional helper).

    Args:
        logits: (B, K-1) CORAL logits
        y: (B,) class labels, with ignore_index for masked positions
        num_classes: K
        ignore_index: value to ignore

    Returns:
        loss: scalar tensor (mean over valid positions)
    """
    valid_mask = (y != ignore_index)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    logits_valid = logits[valid_mask]
    y_valid = y[valid_mask]

    return coral_loss(logits_valid, y_valid, num_classes, reduction="mean")
