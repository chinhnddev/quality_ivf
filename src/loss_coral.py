"""
CORAL (Consistent Rank Logits) implementation for ordinal regression.
Used for EXP task ordinal classification (0,1,2,3,4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Target encoding
# =========================
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
    y = y.long()

    assert y.min() >= 0 and y.max() <= num_classes - 1, \
        f"y must be in [0, {num_classes-1}], got min={y.min()}, max={y.max()}"

    device = y.device
    B = y.shape[0]

    targets = torch.zeros(B, num_classes - 1, device=device, dtype=torch.float32)
    for k in range(num_classes - 1):
        targets[:, k] = (y > k).float()

    return targets


# =========================
# CORAL loss
# =========================
def coral_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    CORAL loss for ordinal regression.

    Args:
        logits: (B, K-1) CORAL logits
        y: (B,) class labels in [0, K-1]
        num_classes: K
        reduction: "mean" or "sum"

    Returns:
        loss: scalar tensor
    """
    assert logits.dim() == 2, f"logits must be 2D, got {logits.shape}"
    assert logits.shape[1] == num_classes - 1, \
        f"Expected {num_classes-1} logits, got {logits.shape[1]}"
    assert logits.shape[0] == y.shape[0], "Batch size mismatch"

    targets = coral_encode_targets(y, num_classes)

    # BCE over thresholds
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# =========================
# Prediction / decoding
# =========================
def coral_predict_class(
    logits: torch.Tensor,
    thresholds=0.5,
) -> torch.Tensor:
    """
    Decode CORAL logits to class predictions.

    Args:
        logits: (B, K-1) CORAL logits
        thresholds: float or list/tensor of length K-1 (probability space)

    Returns:
        y_pred: (B,) predicted class labels in [0, K-1]
    """
    assert logits.dim() == 2, f"logits must be 2D, got {logits.shape}"

    probs = torch.sigmoid(logits)  # (B, K-1)

    if isinstance(thresholds, (int, float)):
        thr = torch.full_like(probs, float(thresholds))
        exceeds = (probs >= thr).float()
    else:
        thresholds = torch.as_tensor(
            thresholds, device=logits.device, dtype=logits.dtype
        )
        if thresholds.dim() == 0:
            exceeds = (probs >= thresholds).float()
        else:
            assert thresholds.shape[0] == logits.shape[1], \
                f"thresholds length {thresholds.shape[0]} != K-1 {logits.shape[1]}"
            exceeds = (probs >= thresholds.unsqueeze(0)).float()

    # Class = number of exceeded thresholds
    y_pred = exceeds.sum(dim=1).long()
    return y_pred


# =========================
# Masked CORAL loss (optional)
# =========================
def coral_loss_masked(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    CORAL loss with masking for ignored labels.

    Args:
        logits: (B, K-1)
        y: (B,)
        num_classes: K
        ignore_index: label value to ignore

    Returns:
        loss: scalar tensor
    """
    valid_mask = (y != ignore_index)

    if valid_mask.sum() == 0:
        # keep graph & dtype safe
        return logits.sum() * 0.0

    logits_valid = logits[valid_mask]
    y_valid = y[valid_mask]

    return coral_loss(
        logits_valid,
        y_valid,
        num_classes=num_classes,
        reduction="mean",
    )
