#!/usr/bin/env python3
import argparse
import os
import sys
import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.dataset import GardnerDataset
from src.loss_coral import coral_predict_class
from src.model import IVF_EffiMorphPP
from src.utils import normalize_exp_token, normalize_icm_te_token


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_state_dict_robust(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    new_state = {}
    for k, v in state.items():
        k2 = k
        if k2.startswith("model."):
            k2 = k2[len("model.") :]
        if k2.startswith("module."):
            k2 = k2[len("module.") :]
        new_state[k2] = v
    if "n_averaged" in new_state:
        new_state.pop("n_averaged", None)
        print("[SWA] Dropped 'n_averaged' key from checkpoint before loading.")
    return new_state


def normalize_labels_in_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["EXP", "ICM", "TE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


MODEL_CONFIG_MAPPING = {
    "dropout": "dropout_p",
    "width_mult": "width_mult",
    "depth_mult": "depth_mult",
    "use_xception_mid": "use_xception_mid",
    "use_late_mhsa": "use_late_mhsa",
    "mhsa_layers": "mhsa_layers",
    "mhsa_heads": "mhsa_heads",
    "use_gem": "use_gem",
    "head_mlp": "head_mlp",
    "head_hidden_dim": "head_hidden_dim",
    "head_dropout": "head_dropout",
}


def load_model_kwargs_from_config(config_path: Optional[Path]) -> dict:
    if config_path is None or not config_path.exists():
        return {}
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.get("model")
    if model_cfg is None:
        return {}
    if OmegaConf.is_config(model_cfg):
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    kwargs = {}
    for field, arg_name in MODEL_CONFIG_MAPPING.items():
        if field in model_cfg and model_cfg[field] is not None:
            kwargs[arg_name] = model_cfg[field]
    return kwargs


# --------------------------
# Match THEIR evaluation mapping
# --------------------------
def _to_int_or_none(x: str) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return int(s)


def map_exp_gold_value(v: str) -> int:
    token = normalize_exp_token(v)
    return int(token) if token in {"0", "1", "2", "3", "4"} else 5


def map_icm_te_gold_value(v: str) -> int:
    token = normalize_icm_te_token(v)
    return int(token) if token in {"0", "1", "2"} else 3


def map_pred_icm_te(v: int) -> int:
    # If prediction is not 0/1/2 -> map to 3
    return v if v in [0, 1, 2] else 3


def map_pred_exp(v: int) -> int:
    # If prediction is not 0..4 -> map to 5
    return v if v in [0, 1, 2, 3, 4] else 5


def _normalize_exp_prediction(value) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        candidate = int(float(value))
    except Exception:
        return None
    return map_pred_exp(candidate)


def extract_predicted_exp_series(preds_df: pd.DataFrame) -> Optional[pd.Series]:
    candidates = [
        "EXP_pred",
        "exp_pred",
        "pred_EXP",
        "pred_exp",
        "y_pred_exp",
        "y_pred_EXP",
        "EXP_pred_raw",
        "exp_pred_raw",
    ]
    for col in candidates:
        if col in preds_df.columns:
            series = preds_df[col].apply(_normalize_exp_prediction)
            series.name = col
            return series
    return None


DEFAULT_CORAL_THR = 0.5


def find_automatic_coral_threshold(checkpoint: Path, out_dir: Path) -> Optional[float]:
    candidates = []
    if checkpoint:
        candidates.append(checkpoint.parent)
    if out_dir:
        candidates.append(out_dir)
    seen = set()
    for root in candidates:
        if root in seen or not root:
            continue
        seen.add(root)
        thr_path = root / "best_coral_threshold.json"
        if not thr_path.exists():
            continue
        try:
            with thr_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            thr_value = payload.get("best_coral_thr")
            if thr_value is None:
                continue
            thr_value = float(thr_value)
            print(f"[AUTO CORAL] Loaded tuned threshold {thr_value:.4f} from {thr_path}")
            return thr_value
        except Exception as exc:
            print(f"[AUTO CORAL] Failed to read {thr_path}: {exc}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IVF-EffiMorphPP on Gardner split, using the SAME evaluation logic as authors."
    )
    parser.add_argument("--task", type=str, required=True, choices=["exp", "icm", "te"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"],
                        help="Split to evaluate: test or val (default: test)")
    parser.add_argument("--img_dir", type=str, default="data/blastocyst_Dataset/Images")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)  # safer on colab
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config.yaml to recreate training model settings (defaults to <checkpoint>/config.yaml).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_strict", action="store_true")
    parser.add_argument("--use_coral", type=int, default=0)
    parser.add_argument("--coral_thr", type=float, default=None,
                        help=f"Uniform CORAL threshold for EXP (default {DEFAULT_CORAL_THR}).")
    parser.add_argument("--auto_thr", type=int, choices=[0, 1], default=1,
                        help="Automatically load tuned CORAL threshold when --coral_thr is not provided.")
    parser.add_argument("--authors_filter_exp01_for_icm_te", type=int, choices=[0, 1], default=1,
                        help="When evaluating ICM/TE, skip samples where either EXP_gt or EXP_pred is in {0,1}.")
    parser.add_argument("--coral_thr_last", type=float, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_csv = os.path.join(args.splits_dir, f"{args.split}.csv")
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"{args.split}.csv not found at: {split_csv}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
    if not os.path.exists(args.img_dir):
        raise FileNotFoundError(f"Image directory not found at: {args.img_dir}")

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

    # Model outputs (training) classes:
    train_num_classes = 5 if args.task == "exp" else 3

    # CORAL auto-detect for EXP
    use_coral_flag = bool(args.use_coral)
    state_dict = load_state_dict_robust(args.checkpoint, device)
    if args.task == "exp" and "head.4.weight" in state_dict:
        out_dim = state_dict["head.4.weight"].shape[0]
        inferred_coral = (out_dim == 4)
        if not args.use_coral and inferred_coral:
            print("[INFO] Checkpoint has 4 outputs; auto-enabling CORAL")
            use_coral_flag = True

    user_provided_thr = args.coral_thr is not None
    coral_thr_value = args.coral_thr if user_provided_thr else DEFAULT_CORAL_THR
    auto_thr_enabled = bool(args.auto_thr)
    if use_coral_flag and args.task == "exp" and auto_thr_enabled and not user_provided_thr:
        auto_thr = find_automatic_coral_threshold(Path(args.checkpoint), Path(args.out_dir))
        if auto_thr is not None:
            coral_thr_value = auto_thr
        else:
            print(f"[AUTO CORAL] No tuned threshold found; falling back to default {coral_thr_value:.2f}")

    config_file = Path(args.config) if args.config else None
    if config_file and not config_file.exists():
        raise FileNotFoundError(f"Config file not found at: {config_file}")
    if config_file is None:
        candidate = Path(args.checkpoint).resolve().parent / "config.yaml"
        if candidate.exists():
            config_file = candidate

    model_kwargs = load_model_kwargs_from_config(config_file)
    if config_file:
        print(f"[CONFIG] Loading model settings from {config_file}")
    else:
        print("[CONFIG] No config.yaml detected next to checkpoint; using default model settings.")

    model = IVF_EffiMorphPP(
        train_num_classes,
        task=args.task,
        use_coral=use_coral_flag,
        **model_kwargs,
    ).to(device)
    strict = not args.no_strict
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    # CORAL decoding thresholds
    coral_thresholds = None
    if use_coral_flag and args.task == "exp":
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            out = model(dummy)
            assert out.shape[1] == 4, f"Expected 4 CORAL logits, got {out.shape[1]}"
        if args.coral_thr_last is not None:
            coral_thresholds = [coral_thr_value, coral_thr_value, coral_thr_value, args.coral_thr_last]
            print(f"[CORAL] Using per-threshold decoding: {coral_thresholds}")
        else:
            coral_thresholds = coral_thr_value
            print(f"[CORAL] Using uniform threshold: {coral_thresholds}")

    # Predict
    all_preds: List[int] = []
    all_probs: List[List[float]] = []
    all_img_names: List[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch[0]
            img_names = batch[-1] if len(batch) >= 4 else None

            images = images.to(device, non_blocking=True)
            outputs = model(images)

            if use_coral_flag and args.task == "exp":
                preds = coral_predict_class(outputs, thresholds=coral_thresholds)
                probs = torch.sigmoid(outputs)
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

            if img_names is not None:
                if isinstance(img_names, (list, tuple)):
                    all_img_names.extend([str(x) for x in img_names])
                else:
                    all_img_names.extend([str(x) for x in img_names])

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_file = os.path.join(args.out_dir, f"metrics_{args.split}.json")
    preds_file = os.path.join(args.out_dir, f"preds_{args.split}.csv")

    if "Image" not in split_df.columns:
        raise KeyError("Expected 'Image' column in split CSV.")

    # Merge by Image
    pred_rows = {"Image": all_img_names, "y_pred_raw": all_preds}
    num_outputs = len(all_probs[0]) if all_probs else train_num_classes
    for i in range(num_outputs):
        pred_rows[f"prob_{i}"] = [p[i] for p in all_probs]
    pred_by_key = pd.DataFrame(pred_rows).groupby("Image", as_index=False).mean(numeric_only=True)

    preds_df = split_df.copy()
    preds_df = preds_df.merge(pred_by_key, on="Image", how="left")
    if preds_df["y_pred_raw"].isna().any():
        missing = int(preds_df["y_pred_raw"].isna().sum())
        raise RuntimeError(f"Missing predictions for {missing} rows after merge by Image.")

    preds_df["y_pred_raw"] = preds_df["y_pred_raw"].astype(int)
    exp_pred_series = extract_predicted_exp_series(preds_df)

    # --------------------------
    # Build y_true/y_pred EXACTLY like authors
    # --------------------------
    task = args.task

    y_true: List[int] = []
    y_pred: List[int] = []
    used_images: List[str] = []

    if task == "exp":
        # Evaluate EXP on ALL gold_test images, mapping NA/invalid to class 5
        for _, r in preds_df.iterrows():
            gt = map_exp_gold_value(r["EXP"])
            pr = map_pred_exp(int(r["y_pred_raw"]))
            y_true.append(gt)
            y_pred.append(pr)
            used_images.append(str(r["Image"]))
        eval_num_classes = 6  # 0..5
        labels = list(range(eval_num_classes))

    else:
        # Evaluate ICM/TE only when EXP_gt not in [0,1]
        col = task.upper()
        exp_pred_label = exp_pred_series.name if exp_pred_series is not None else None
        if exp_pred_label:
            print(f"[ICM/TE] Filtering using predicted EXP column '{exp_pred_label}'.")
        raw_series = preds_df[col].fillna("").astype(str).str.strip()
        raw_unique = sorted({val if val != "" else "<EMPTY>" for val in raw_series.unique()})
        raw_numeric = pd.to_numeric(preds_df[col], errors="coerce")
        na_raw_count = int((raw_numeric == -1).sum())
        nd_raw_count = int((raw_numeric == 3).sum())
        mapped_counts = Counter(map_icm_te_gold_value(v) for v in preds_df[col])
        print(f"[ICM/TE GOLD] Raw label values: {raw_unique}")
        if na_raw_count:
            print(f"[ICM/TE GOLD] Raw '-1' (NA) occurrences={na_raw_count} -> mapped to class 3.")
        if nd_raw_count:
            print(f"[ICM/TE GOLD] Raw '3' (ND) occurrences={nd_raw_count} -> mapped to class 3.")
        print(f"[ICM/TE GOLD] Mapped label counts: {dict(sorted(mapped_counts.items()))}")
        filter_by_exp = bool(args.authors_filter_exp01_for_icm_te)
        if filter_by_exp and exp_pred_label:
            print("[ICM/TE] Applying authors exp {0,1} filter.")
        for idx, r in preds_df.iterrows():
            exp_gt = map_exp_gold_value(r["EXP"])
            if filter_by_exp and exp_gt in [0, 1]:
                continue  # not defined for ICM/TE per authors
            exp_pred = exp_pred_series.loc[idx] if exp_pred_series is not None else None
            if filter_by_exp and exp_pred in [0, 1]:
                continue
            gt = map_icm_te_gold_value(r[col])
            pr = map_pred_icm_te(int(r["y_pred_raw"]))
            y_true.append(gt)
            y_pred.append(pr)
            used_images.append(str(r["Image"]))
        eval_num_classes = 4  # 0..3
        labels = list(range(eval_num_classes))

    # Metrics (same names as Table 2 intent)
    acc = accuracy_score(y_true, y_pred)
    avg_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    avg_rec = recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    avg_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)

    per_prec, per_rec, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Logging
    y_true_counts = Counter(y_true)
    print(f"y_true class counts (authors-style): {dict(y_true_counts)}")
    for cls in labels:
        if y_true_counts.get(cls, 0) == 0:
            print(f"WARNING: Class {cls} has 0 support in {args.split}")

    y_pred_counts = Counter(y_pred)
    counts_full = {i: int(y_pred_counts.get(i, 0)) for i in labels}
    ratios_full = {i: float(counts_full[i] / max(1, len(y_pred))) for i in labels}
    print(f"y_pred counts: {counts_full}")
    print(f"y_pred ratio:  {ratios_full}")
    print("Confusion Matrix:")
    print(cm)
    print("Per-class recall:", per_rec.tolist())

    # Save
    out = {
        "task": task,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "device": str(device),
        "n_total_split_rows": int(len(split_df)),
        "n_eval_used": int(len(y_true)),
        "eval_label_space": labels,
        "accuracy": float(acc),
        "avg-prec": float(avg_prec),
        "avg-rec": float(avg_rec),
        "avg-f1": float(avg_f1),
        "per_class_precision": per_prec.tolist(),
        "per_class_recall": per_rec.tolist(),
        "per_class_f1": per_f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "y_pred_distribution": {"counts": counts_full, "ratios": ratios_full},
    }
    if use_coral_flag and task == "exp":
        out["coral_thresholds"] = coral_thresholds

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    preds_df.to_csv(preds_file, index=False)

    print(f"Saved metrics to: {os.path.abspath(metrics_file)}")
    print(f"Saved predictions to: {os.path.abspath(preds_file)}")
    print(
        f"Evaluation complete | task={task} | split={args.split} | n_eval_used={len(y_true)} | "
        f"accuracy={acc:.4f} | avg-prec={avg_prec:.4f} | avg-rec={avg_rec:.4f} | avg-f1={avg_f1:.4f}"
    )


if __name__ == "__main__":
    main()
