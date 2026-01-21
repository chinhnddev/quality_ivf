#!/usr/bin/env python3
import argparse
import os
import sys
import json
import random
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, List

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
from src.loss_coral import coral_predict_class


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
    for col in ["EXP", "ICM", "TE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def _map_icm_te_to_012(series: pd.Series) -> pd.Series:
    """
    Accepts: "0/1/2", "A/B/C", "1/2/3"  -> returns normalized strings "0/1/2" (invalid stays as-is)
    """
    s = series.astype(str).str.strip()

    # A/B/C -> 0/1/2
    s = s.replace({"A": "0", "B": "1", "C": "2"})

    # 1/2/3 -> 0/1/2
    s = s.replace({"1": "0", "2": "1", "3": "2"})

    return s


def build_valid_mask_and_ytrue(df: pd.DataFrame, task: str) -> Tuple[np.ndarray, List[int]]:
    """
    Returns:
        valid_mask_np: np.ndarray bool [N] aligned with df row order
        y_true_list: list[int] aligned with valid_mask positions (same order as df[valid_mask])
    """
    task_u = task.upper()

    if task == "exp":
        exp_num = pd.to_numeric(df["EXP"], errors="coerce")
        valid_mask = exp_num.isin([0, 1, 2, 3, 4]).to_numpy()
        y_true = exp_num[valid_mask].astype(int).tolist()
        return valid_mask, y_true

    # icm/te
    col = task_u
    s = _map_icm_te_to_012(df[col])
    valid_mask = s.isin(["0", "1", "2"]).to_numpy()
    y_true = s[valid_mask].astype(int).tolist()
    return valid_mask, y_true


def build_encoded_ytrue_column(df: pd.DataFrame, task: str) -> np.ndarray:
    """
    y_true encoded aligned to df rows; invalid label -> -1
    """
    y_true_encoded = np.full(len(df), -1, dtype=int)
    valid_mask, y_true_list = build_valid_mask_and_ytrue(df, task)
    if valid_mask.any():
        y_true_encoded[valid_mask] = np.array(y_true_list, dtype=int)
    return y_true_encoded


def main():
    parser = argparse.ArgumentParser(description="Evaluate IVF-EffiMorphPP on Gardner split (val/test)")
    parser.add_argument("--task", type=str, required=True, choices=["exp", "icm", "te"],
                        help="Task to evaluate: exp, icm, or te")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--splits_dir", type=str, required=True, help="Directory containing {val,test}.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for metrics and preds")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"],
                        help="Split to evaluate: test or val (default: test)")
    parser.add_argument("--img_dir", type=str, default="data/blastocyst_Dataset/Images",
                        help="Directory containing blastocyst images")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_strict", action="store_true",
                        help="Disable strict load_state_dict (allow missing/extra keys)")
    parser.add_argument("--use_coral", type=int, default=0,
                        help="Use CORAL ordinal regression for EXP task (0=OFF, 1=ON)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    split_csv = os.path.join(args.splits_dir, f"{args.split}.csv")
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"{args.split}.csv not found at: {split_csv}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    if not os.path.exists(args.img_dir):
        raise FileNotFoundError(f"Image directory not found at: {args.img_dir}")

    # Load split df
    split_df = pd.read_csv(split_csv)
    split_df = normalize_labels_in_df(split_df)

    # Dataset + loader
    dataset = GardnerDataset(split_csv, args.img_dir, task=args.task, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    num_classes = 5 if args.task == "exp" else 3
    model = IVF_EffiMorphPP(num_classes, task=args.task, use_coral=bool(args.use_coral)).to(device)

    state = load_state_dict_robust(args.checkpoint, device)
    strict = not args.no_strict
    model.load_state_dict(state, strict=strict)
    model.eval()

    # Safety check for CORAL
    if args.use_coral and args.task == "exp":
        # Test forward pass to check output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_output = model(dummy_input)
            assert dummy_output.shape[1] == 4, f"Expected 4 CORAL logits for EXP, got {dummy_output.shape[1]}"
        print(f"[CORAL] Model outputs {dummy_output.shape[1]} logits for EXP ordinal regression")

    # Predict
    all_preds: List[int] = []
    all_probs: List[List[float]] = []
    all_img_names: List[str] = []

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Unexpected batch structure. Expected (images, labels, ...)")

            images = batch[0]

            # If dataset returns image names as the last element
            img_names = None
            if len(batch) >= 4:
                img_names = batch[-1]

            images = images.to(device, non_blocking=True)
            outputs = model(images)
            if args.use_coral and args.task == "exp":
                preds = coral_predict_class(outputs)
                probs = torch.sigmoid(outputs)  # For CORAL, use sigmoid for probabilities
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

            if img_names is not None:
                # ensure python list of strings
                if isinstance(img_names, (list, tuple)):
                    all_img_names.extend([str(x) for x in img_names])
                else:
                    # sometimes tensor/object array
                    try:
                        all_img_names.extend([str(x) for x in img_names])
                    except Exception:
                        # fallback: no names
                        all_img_names = []

    # Build predictions dataframe
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_file = os.path.join(args.out_dir, f"metrics_{args.split}.json")
    preds_file = os.path.join(args.out_dir, f"preds_{args.split}.csv")

    gt_col = args.task.upper()
    if gt_col not in split_df.columns:
        raise KeyError(f"Ground-truth column '{gt_col}' not found in {split_csv}")

    # If we have image names, merge by key; otherwise assume order matches CSV
    if len(all_img_names) == len(all_preds) and "Image" in split_df.columns:
        pred_rows = {"Image": all_img_names, "y_pred": all_preds}
        for i in range(num_classes):
            pred_rows[f"prob_{i}"] = [p[i] for p in all_probs]
        pred_by_key = pd.DataFrame(pred_rows)

        # aggregate in case duplicates exist
        pred_by_key = pred_by_key.groupby("Image", as_index=False).mean(numeric_only=True)

        preds_df = split_df.copy()
        preds_df["y_true_raw"] = preds_df[gt_col]
        preds_df["y_true"] = build_encoded_ytrue_column(preds_df, args.task)

        preds_df = preds_df.merge(pred_by_key, on="Image", how="left")

        # Safety: ensure we have predictions for all rows (if not, dataset/df mismatch)
        missing = preds_df["y_pred"].isna().sum()
        if missing > 0:
            raise RuntimeError(
                f"Missing predictions for {missing} rows after merge by Image key. "
                f"Likely mismatch between split CSV and dataset (filtered/dropped images)."
            )

        preds_df["y_pred"] = preds_df["y_pred"].astype(int)

    else:
        # Fallback: order-aligned
        if len(all_preds) != len(split_df):
            raise RuntimeError(
                f"Length mismatch: preds={len(all_preds)} vs split_df={len(split_df)}. "
                f"Your GardnerDataset likely filters/drops rows. "
                f"Fix by returning image_name from dataset and merge by Image key."
            )

        preds_df = split_df.copy()
        preds_df["y_true_raw"] = preds_df[gt_col]
        preds_df["y_true"] = build_encoded_ytrue_column(preds_df, args.task)
        preds_df["y_pred"] = all_preds
        for i in range(num_classes):
            preds_df[f"prob_{i}"] = [p[i] for p in all_probs]

    # Metrics (mask invalid labels)
    valid_mask_np, y_true = build_valid_mask_and_ytrue(split_df, args.task)
    valid_pos = np.nonzero(valid_mask_np)[0]
    y_pred = [int(preds_df.loc[i, "y_pred"]) for i in valid_pos]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    # y_pred distribution full (0..K-1)
    counts = Counter(y_pred)
    counts_full = {i: int(counts.get(i, 0)) for i in range(num_classes)}
    ratios_full = {i: float(counts_full[i] / max(1, len(y_pred))) for i in range(num_classes)}
    print(f"y_pred counts: {counts_full}")
    print(f"y_pred ratio:  {ratios_full}")

    print("Confusion Matrix:")
    print(cm)
    print("Per-class recall:", recall.tolist())

    metrics = {
        "task": args.task,
        "split": args.split,
        "num_classes": num_classes,
        "checkpoint": str(args.checkpoint),
        "splits_dir": str(args.splits_dir),
        "img_dir": str(args.img_dir),
        "seed": args.seed,
        "device": str(device),
        "n_total_split_rows": int(len(split_df)),
        "n_eval_used": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "y_pred_distribution": {"counts": counts_full, "ratios": ratios_full},
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    preds_df.to_csv(preds_file, index=False)

    print(f"Saved metrics to: {os.path.abspath(metrics_file)}")
    print(f"Saved predictions to: {os.path.abspath(preds_file)}")
    print(
        f"Evaluation complete | task={args.task} | split={args.split} | n_eval_used={len(y_true)} | "
        f"Acc={acc:.4f} | MacroF1={macro_f1:.4f} | WeightedF1={weighted_f1:.4f}"
    )


if __name__ == "__main__":
    main()
