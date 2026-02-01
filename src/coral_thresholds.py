import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_CORAL_THRESHOLD = 0.5


def _grid_values(grid_min: float, grid_max: float, grid_step: float) -> List[float]:
    if grid_step <= 0 or grid_min > grid_max:
        return []
    return np.arange(grid_min, grid_max + 1e-9, grid_step).tolist()


def prepare_threshold_vector(
    thresholds: Union[float, Sequence[float], Dict[str, Any]],
    num_boundaries: int,
) -> List[float]:
    if isinstance(thresholds, dict):
        if "thresholds" in thresholds:
            value = thresholds["thresholds"]
        elif "threshold" in thresholds:
            value = thresholds["threshold"]
        else:
            raise ValueError("Threshold payload missing expected key.")
    else:
        value = thresholds

    if isinstance(value, (int, float)):
        return [float(value)] * num_boundaries
    value_list = [float(x) for x in value]
    if len(value_list) != num_boundaries:
        raise ValueError(
            f"Expected {num_boundaries} thresholds, got {len(value_list)}."
        )
    return value_list


def decode_coral_predictions(
    logits: torch.Tensor,
    thresholds: Union[float, Sequence[float]],
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num_boundaries = logits.shape[1]
    thr_vector = prepare_threshold_vector(thresholds, num_boundaries)
    thr_tensor = torch.as_tensor(
        thr_vector, device=logits.device, dtype=logits.dtype
    )
    exceeds = (probs > thr_tensor.unsqueeze(0)).float()
    return exceeds.sum(dim=1).long()


def evaluate_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: Sequence[float],
) -> Tuple[float, float]:
    preds = decode_coral_predictions(logits, thresholds).cpu().numpy()
    labels_np = labels.cpu().numpy()
    acc = accuracy_score(labels_np, preds)
    weighted_f1 = f1_score(labels_np, preds, average="weighted", zero_division=0)
    return acc, weighted_f1


def tune_vector_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    grid_min: float,
    grid_max: float,
    grid_step: float,
    num_classes: int,
    iterations: int = 3,
    ckpt_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    if logits.numel() == 0:
        return None
    grid = _grid_values(grid_min, grid_max, grid_step)
    if not grid:
        print("[CORAL TUNING] Invalid threshold grid.")
        return None

    num_boundaries = num_classes - 1
    labels_tensor = labels

    best_uniform_thr = DEFAULT_CORAL_THRESHOLD
    best_acc = -1.0
    best_wf1 = -1.0
    for thr in grid:
        acc, wf1 = evaluate_thresholds(logits, labels_tensor, [thr] * num_boundaries)
        if acc > best_acc or (acc == best_acc and wf1 > best_wf1):
            best_acc = acc
            best_wf1 = wf1
            best_uniform_thr = float(thr)

    best_thresholds = [best_uniform_thr] * num_boundaries
    best_metrics = (best_acc, best_wf1)

    for _ in range(iterations):
        improved = False
        for idx in range(num_boundaries):
            local_best_thr = best_thresholds[idx]
            for candidate in grid:
                trial = best_thresholds.copy()
                trial[idx] = float(candidate)
                acc, wf1 = evaluate_thresholds(logits, labels_tensor, trial)
                if acc > best_metrics[0] or (
                    acc == best_metrics[0] and wf1 > best_metrics[1]
                ):
                    best_metrics = (acc, wf1)
                    best_thresholds = trial
                    improved = True
        if not improved:
            break

    payload = {
        "type": "vector",
        "thresholds": [float(t) for t in best_thresholds],
        "metric": "val_acc",
        "val_acc": float(best_metrics[0]),
        "val_weighted_f1": float(best_metrics[1]),
        "grid": {"min": grid_min, "max": grid_max, "step": grid_step},
        "uniform_init": {"threshold": best_uniform_thr},
        "ckpt": str(ckpt_path) if ckpt_path is not None else None,
    }
    return payload


def load_threshold_payload(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data
