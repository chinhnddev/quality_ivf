#!/usr/bin/env python3
import argparse
import os
import sys
import json
import random
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.dataset import GardnerDataset
from src.model import IVF_EffiMorphPP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # reproducibility (may reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_state_dict_robust(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)

    # lightning-style: {"state_dict": ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # strip common prefixes
    new_state = {}
    for k, v in state.items():
        k2 = k
        if k2.startswith("model."):
            k2 = k2[len("model.") :]
        if k2.startswith("module."):
            k2 = k2[len("module.") :]
        new_state[k2] = v
    return new_state


def normalize_labels_in_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize to string then strip; keep EXP also numeric-friendly later
    for col in ["EXP", "ICM", "TE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def build_valid_mask_and_ytrue(df: pd.DataFrame, task: str):
    """
    Returns:
        valid_mask_np: np.ndarray bool of shape [N] aligned with df row order
        y_true: list[int] aligned with valid_mask positions
    """
    task_u = task.upper()

    if task == "exp":
        exp_num = pd.to_numeric(df["EXP"], errors="coerce")
        valid_mask = exp_num.isin([0, 1, 2, 3, 4]).to_numpy()
        y_true = exp_num[valid_mask].astype(int).tolist()
        return valid_mask, y_true

    # icm/te: expect 0/1/2 are valid classes; ND/NA/empty invalid
    # ensure string
    col = task_u
    valid_mask = df[col].isin(["0", "1", "2"]).to_numpy()
    y_true = df.loc[valid_mask, col].astype(int).tolist()
    return valid_mask, y_true


def main():
    parser = argparse.ArgumentParser(description="Evaluate IVF-EffiMorphPP on Gardner test set")
    parser.add_argument("--task", type=str, required=True, choices=["exp", "icm", "te"],
                        help="Task to evaluate: exp, icm, or te")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--splits_dir", type=str, required=True, help="Directory containing test.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for metrics and preds")
    parser.add_argument(
        "--img_dir",
        type=str,
        default="Images",
        help="Directory containing blastocyst images",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_strict",
        action="store_true",
        help="Disable strict load_state_dict (allow missing/extra keys)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    test_csv = os.path.join(args.splits_dir, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"test.csv not found at: {test_csv}")

    # Load raw test for saving preds + masking ground-truth
    test_df = pd.read_csv(test_csv)
    test_df = normalize_labels_in_df(test_df)

    # Dataset + loader
    # Assumption: GardnerDataset returns samples in the same order as test.csv rows.
    # If your GardnerDataset filters/drops rows (missing image/label), you MUST modify dataset
    # to also return image_name so we can merge predictions by key.
    test_dataset = GardnerDataset(test_csv, args.img_dir, task=args.task, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    num_classes = 5 if args.task == "exp" else 3
    model = IVF_EffiMorphPP(num_classes).to(device)

    state = load_state_dict_robust(args.checkpoint, device)
    # strict=True by default; use --no_strict to disable
    strict = not args.no_strict
    model.load_state_dict(state, strict=strict)
    model.eval()

    # Predict
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            # support datasets returning (images, labels) or (images, labels, meta)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images = batch[0]
            else:
                raise ValueError("Unexpected batch structure from DataLoader. Expected (images, labels, ...).")

            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    # If lengths mismatch, your dataset is not 1-1 with test_df rows
    if len(all_preds) != len(test_df):
        raise RuntimeError(
            f"Length mismatch: preds={len(all_preds)} vs test_df={len(test_df)}. "
            f"Your GardnerDataset likely filters/drops rows. "
            f"Fix by returning image_name from dataset and join preds by Image key."
        )

    # Build preds_df aligned with test_df row order
    preds_df = test_df[["Image"]].copy() if "Image" in test_df.columns else pd.DataFrame({"Image": list(range(len(test_df)))})
    gt_col = args.task.upper()
    if gt_col not in test_df.columns:
        raise KeyError(f"Ground-truth column '{gt_col}' not found in test.csv")

    preds_df["y_true_raw"] = test_df[gt_col]
    preds_df["y_pred"] = all_preds
    for i in range(num_classes):
        preds_df[f"prob_{i}"] = [p[i] for p in all_probs]

    # Mask invalid labels for metrics
    valid_mask_np, y_true = build_valid_mask_and_ytrue(test_df, args.task)
    valid_pos = np.nonzero(valid_mask_np)[0]
    y_pred = [all_preds[i] for i in valid_pos]

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "task": args.task,
        "num_classes": num_classes,
        "checkpoint": str(args.checkpoint),
        "splits_dir": str(args.splits_dir),
        "img_dir": str(args.img_dir),
        "seed": args.seed,
        "device": str(device),
        "n_total_test_rows": int(len(test_df)),
        "n_eval_used": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    preds_df.to_csv(os.path.join(args.out_dir, "preds_test.csv"), index=False)

    print(
        f"Evaluation complete | task={args.task} | n_eval_used={len(y_true)} | "
        f"Acc={acc:.4f} | MacroF1={macro_f1:.4f} | WeightedF1={weighted_f1:.4f}"
    )


if __name__ == "__main__":
    main()
