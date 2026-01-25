#!/usr/bin/env python3
import argparse
import os
import sys
import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional, Dict, Iterable

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
    classification_report,
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


def load_state_dict_robust(ckpt_path: str, device: torch.device) -> Tuple[dict, dict]:
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
    metadata = {}
    if isinstance(ckpt, dict):
        metadata = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return new_state, metadata


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


def _map_icm_te_value(value) -> int:
    if pd.isna(value):
        return 3
    if isinstance(value, (int, np.integer)):
        candidate = int(value)
    else:
        text = str(value).strip()
        if text == "" or text.upper() == "NA":
            return 3
        if text.upper() == "ND":
            return 3
        try:
            candidate = int(float(text))
        except Exception:
            return 3
    return candidate if candidate in {0, 1, 2} else 3


def map_icm_te_gold_value(v: str, merge_map: Optional[dict] = None) -> int:
    base = _map_icm_te_value(v)
    if merge_map:
        return merge_map.get(base, merge_map.get(3, base))
    return base


def map_pred_icm_te(v: int) -> int:
    return _map_icm_te_value(v)


def map_pred_exp(v: int) -> int:
    # If prediction is not 0..4 -> map to 5
    return v if v in [0, 1, 2, 3, 4] else 5


def _classification_metrics(y_true: List[int], y_pred: List[int], labels: List[int]) -> dict:
    if not y_true:
        zero_cm = [[0 for _ in labels] for _ in labels]
        return {
            "confusion_matrix": zero_cm,
            "per_class_recall": [0.0 for _ in labels],
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "confusion_matrix": cm.tolist(),
        "per_class_recall": per_rec.tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def _build_task_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: List[int],
    total_matched: int,
    skipped: int,
) -> dict:
    stats = _classification_metrics(y_true, y_pred, labels)
    stats.update(
        {
            "n_total_matched": int(total_matched),
            "n_skipped_exp_rule": int(skipped),
            "n_used": int(len(y_true)),
        }
    )
    return stats


def _normalize_exp_prediction(value) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        candidate = int(float(value))
    except Exception:
        return None
    return map_pred_exp(candidate)


def _resolve_consensus_csv(consensus_csv: str, splits_dir: str) -> str:
    if consensus_csv and os.path.exists(consensus_csv):
        return consensus_csv
    fallback = os.path.join(splits_dir, "test.csv") if splits_dir else None
    if fallback and os.path.exists(fallback):
        print(f"[PAPER EVAL] Consensus CSV not found at {consensus_csv}; falling back to {fallback}")
        return fallback
    raise FileNotFoundError(f"Consensus CSV missing at {consensus_csv} and no fallback available.")


def _find_column_case_insensitive(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        normalized = candidate.lower()
        if normalized in lower_map:
            return lower_map[normalized]
    return None


def _prepare_image_df(df: pd.DataFrame, image_col: Optional[str] = None) -> pd.DataFrame:
    if image_col is None:
        image_col = _find_column_case_insensitive(df, ["Image", "image", "Filename", "filename"])
    if image_col is None or image_col not in df.columns:
        raise KeyError(f"Could not find image column in predictions DataFrame: {df.columns.tolist()}")
    df = df.rename(columns={image_col: "Image"})
    df = df.copy()
    df["Image"] = df["Image"].astype(str).str.strip()
    return df.drop_duplicates(subset=["Image"])


def _load_paper_predictions(src_df: pd.DataFrame, pred_csv: Optional[str], out_dir: str, split: str) -> Tuple[pd.DataFrame, str]:
    default_path = os.path.join(out_dir, f"preds_{split}.csv")
    if pred_csv:
        if not os.path.exists(pred_csv):
            raise FileNotFoundError(f"Predictions CSV not found at {pred_csv}")
        print(f"[PAPER EVAL] Loading predictions from provided CSV: {pred_csv}")
        preds = pd.read_csv(pred_csv)
        preds = _prepare_image_df(preds)
        return preds, pred_csv
    return _prepare_image_df(src_df.copy()), default_path


def run_paper_evaluation(
    preds_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    icm_merge_map: Optional[dict],
) -> Tuple[dict, Dict[str, str], str]:
    consensus_csv = _resolve_consensus_csv(args.consensus_csv, args.splits_dir)
    preds_source, pred_source_path = _load_paper_predictions(preds_df, args.pred_csv, args.out_dir, args.split)
    consensus_df_raw = pd.read_csv(consensus_csv)
    consensus_df = _prepare_image_df(consensus_df_raw)
    required_cols = {"EXP", "ICM", "TE"}
    if not required_cols.issubset(set(consensus_df.columns)):
        raise KeyError(f"Consensus CSV must include columns: {required_cols}. Got: {consensus_df.columns.tolist()}")
    consensus_df = consensus_df.drop_duplicates(subset=["Image"])

    merged = preds_source.merge(
        consensus_df[["Image", "EXP", "ICM", "TE"]],
        on="Image",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No matching rows between predictions and consensus CSV.")

    exp_pred_series = extract_predicted_exp_series(preds_source)
    if exp_pred_series is not None:
        exp_pred_series = exp_pred_series.reindex(merged.index)
        merged["exp_pred_mapped"] = exp_pred_series.apply(_normalize_exp_prediction)
    else:
        merged["exp_pred_mapped"] = pd.Series([None] * len(merged))

    merged["EXP_gt_mapped"] = merged["EXP"].apply(map_exp_gold_value)
    merged["ICM_gt_mapped"] = merged["ICM"].apply(lambda v: map_icm_te_gold_value(v, merge_map=icm_merge_map))
    merged["TE_gt_mapped"] = merged["TE"].apply(map_icm_te_gold_value)
    merged["icm_pred_mapped"] = merged["y_pred_raw"].apply(map_pred_icm_te)
    merged["te_pred_mapped"] = merged["y_pred_raw"].apply(map_pred_icm_te)

    skip_mask = merged["EXP_gt_mapped"].isin({0, 1})
    if exp_pred_series is not None:
        skip_mask |= merged["exp_pred_mapped"].isin({0, 1})
    skipped_filenames = merged.loc[skip_mask, "Image"].astype(str).tolist()
    filtered = merged.loc[~skip_mask]

    reports: Dict[str, str] = {}
    metrics: Dict[str, Any] = {
        "eval_protocol": "paper",
        "total_matched": int(len(merged)),
        "skipped_due_to_exp_rule": int(skip_mask.sum()),
        "skipped_filenames": skipped_filenames,
        "n_total_consensus_rows": int(len(consensus_df)),
        "n_total_pred_rows": int(len(preds_source)),
        "pred_csv": pred_source_path,
        "task": args.task,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "device": str(device),
        "consensus_csv": consensus_csv,
        "eval_protocol": "paper",
    }

    summary_parts = [
        f"total_matched={metrics['total_matched']}",
        f"skipped={metrics['skipped_due_to_exp_rule']}",
    ]
    task = args.task
    if exp_pred_series is not None:
        exp_labels = sorted(set(merged["EXP_gt_mapped"].tolist()) | set(merged["exp_pred_mapped"].dropna().tolist()))
        exp_report_str = classification_report(
            merged["EXP_gt_mapped"], merged["exp_pred_mapped"], labels=exp_labels, zero_division=0
        )
        exp_report_dict = classification_report(
            merged["EXP_gt_mapped"], merged["exp_pred_mapped"], labels=exp_labels, zero_division=0, output_dict=True
        )
        exp_cm = confusion_matrix(merged["EXP_gt_mapped"], merged["exp_pred_mapped"], labels=exp_labels)
        reports["exp"] = exp_report_str
        metrics["n_eval_used_exp"] = int(len(merged))
        metrics["exp"] = {
            "classification_report": exp_report_dict,
            "confusion_matrix": exp_cm.tolist(),
        }

    if task in {"icm", "te"}:
        filtered_true = (
            filtered["ICM_gt_mapped"].tolist() if task == "icm" else filtered["TE_gt_mapped"].tolist()
        )
        filtered_pred = (
            filtered["icm_pred_mapped"].tolist() if task == "icm" else filtered["te_pred_mapped"].tolist()
        )
        report_str = ""
        report_dict = {}
        cm = []
        if filtered_true:
            report_str = classification_report(
                filtered_true, filtered_pred, labels=[0, 1, 2, 3], zero_division=0
            )
            report_dict = classification_report(
                filtered_true, filtered_pred, labels=[0, 1, 2, 3], zero_division=0, output_dict=True
            )
            cm = confusion_matrix(filtered_true, filtered_pred, labels=[0, 1, 2, 3]).tolist()
        reports[task] = report_str or f"No {task.upper()} samples after filtering."
        metrics[f"n_eval_used_{task}"] = int(len(filtered_true))
        metrics[task] = {
            "classification_report": report_dict,
            "confusion_matrix": cm,
        }
        summary_parts.append(f"n_eval_used_{task}={len(filtered_true)}")

    summary_msg = f"Paper evaluation complete | task={args.task} | split={args.split} | " + " | ".join(summary_parts)
    return metrics, reports, summary_msg


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
    parser.add_argument("--eval_protocol", type=str, choices=["default", "paper"], default="default",
                        help="Choose evaluation logic; 'paper' mimics the Gardner paper metrics.")
    parser.add_argument("--consensus_csv", type=str, default="annotations/test_rev.csv",
                        help="Consensus CSV to use for the paper protocol (default mirrors authors).")
    parser.add_argument("--pred_csv", type=str, default=None,
                        help="Predictions CSV to evaluate for paper protocol (default uses generated preds file).")
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
    use_coral_flag = bool(args.use_coral)
    state_dict, ckpt_meta = load_state_dict_robust(args.checkpoint, device)
    default_num_classes = 5 if args.task == "exp" else 3
    train_num_classes = ckpt_meta.get("num_classes", default_num_classes)
    icm_merge_map_raw = ckpt_meta.get("label_map")
    icm_merge_map = None
    if icm_merge_map_raw:
        icm_merge_map = {int(k): int(v) for k, v in icm_merge_map_raw.items()}
    if train_num_classes is None:
        train_num_classes = default_num_classes
    print(f"[EVAL] task={args.task} | train_num_classes={train_num_classes}")
    if args.task == "exp" and "head.4.weight" in state_dict:
        out_dim = state_dict["head.4.weight"].shape[0]
        inferred_coral = (out_dim == 4)
        if not args.use_coral and inferred_coral:
            print("[INFO] Checkpoint has 4 outputs; auto-enabling CORAL")
            use_coral_flag = True
    if args.task in {"icm", "te"} and "head.4.weight" in state_dict:
        ckpt_out_dim = state_dict["head.4.weight"].shape[0]
        assert ckpt_out_dim == train_num_classes, (
            f"Checkpoint head has {ckpt_out_dim} outputs but evaluator built {train_num_classes} logits. "
            "Pass the matching config or checkpoint metadata."
        )

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

    metrics_payload = {}
    summary_msg = ""
    paper_reports: Dict[str, str] = {}
    if args.eval_protocol == "paper":
        metrics_payload, paper_reports, summary_msg = run_paper_evaluation(preds_df, args, device, icm_merge_map)
        for name in ("exp", "icm", "te"):
            report = paper_reports.get(name)
            if report:
                print(f"\n[PAPER EVAL] {name.upper()} classification report:\n{report}")
        metrics_payload["n_total_split_rows"] = int(len(split_df))
    else:
        # --------------------------
        # Build y_true/y_pred EXACTLY like authors
        # --------------------------
        task = args.task
        initial_total = None
        removed_nd_na = 0
        n_eval_total_before = 0

        if task == "exp":
            y_true: List[int] = []
            y_pred: List[int] = []
            used_images: List[str] = []
            # Evaluate EXP on ALL gold_test images, mapping NA/invalid to class 5
            for _, r in preds_df.iterrows():
                gt = map_exp_gold_value(r["EXP"])
                pr = map_pred_exp(int(r["y_pred_raw"]))
                y_true.append(gt)
                y_pred.append(pr)
                used_images.append(str(r["Image"]))
            eval_num_classes = 6  # 0..5
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

            n_eval_total_before = initial_total if initial_total is not None else len(y_true)
            filtered_ratio = removed_nd_na / n_eval_total_before if n_eval_total_before else 0.0
            out = {
                "task": task,
                "split": args.split,
                "checkpoint": str(args.checkpoint),
                "seed": args.seed,
                "device": str(device),
                "n_total_split_rows": int(len(split_df)),
                "n_eval_used": int(len(y_true)),
                "n_eval_total_before_filter": int(n_eval_total_before),
                "n_filtered_nd_na": int(removed_nd_na),
                "filtered_ratio": float(filtered_ratio),
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

            metrics_payload = out
            summary_msg = (
                f"Evaluation complete | task={task} | split={args.split} | n_eval_used={len(y_true)} | "
                f"accuracy={acc:.4f} | avg-prec={avg_prec:.4f} | avg-rec={avg_rec:.4f} | avg-f1={avg_f1:.4f}"
            )

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
            total_matched = int(len(preds_df))
            skipped_due_to_exp_rule = 0
            icm_gt_list: List[int] = []
            icm_pred_list: List[int] = []
            te_gt_list: List[int] = []
            te_pred_list: List[int] = []
            exp_skip_values = {0, 1, -1}
            for idx, r in preds_df.iterrows():
                exp_gt = map_exp_gold_value(r["EXP"])
                exp_pred = exp_pred_series.loc[idx] if exp_pred_series is not None else None
                skip_exp = False
                if filter_by_exp and exp_gt in exp_skip_values:
                    skip_exp = True
                if filter_by_exp and exp_pred in exp_skip_values:
                    skip_exp = True
                if skip_exp:
                    skipped_due_to_exp_rule += 1
                    continue
                n_eval_total_before += 1
                icm_gt_mapped = map_icm_te_gold_value(r["ICM"], merge_map=icm_merge_map)
                te_gt_mapped = map_icm_te_gold_value(r["TE"])
                pred_label = map_pred_icm_te(int(r["y_pred_raw"]))
                skip_na = False
                if task == "icm" and icm_gt_mapped == 3:
                    skip_na = True
                elif task == "te" and te_gt_mapped == 3:
                    skip_na = True
                if skip_na:
                    removed_nd_na += 1
                    continue
                icm_gt_list.append(icm_gt_mapped)
                te_gt_list.append(te_gt_mapped)
                icm_pred_list.append(pred_label)
                te_pred_list.append(pred_label)
            labels = [i for i in range(train_num_classes) if i != 3]
            if not labels:
                labels = list(range(train_num_classes))
            initial_total = total_matched
            filtered_ratio = removed_nd_na / (n_eval_total_before + 1e-9)
            if len(icm_gt_list) != len(te_gt_list):
                print("[ICM/TE] WARNING: inconsistent ICM/TE list lengths after filtering.")
            icm_metrics = None
            te_metrics = None
            print(
                f"[ICM/TE] total_matched={total_matched}, skipped_due_to_exp_rule={skipped_due_to_exp_rule}, "
                f"n_eval_used_icm={len(icm_gt_list)}, n_eval_used_te={len(te_gt_list)}"
            )
        if task in {"icm", "te"}:
            if task == "icm":
                icm_accuracy = 0.0
                icm_avg_prec = 0.0
                icm_avg_rec = 0.0
                icm_avg_f1 = 0.0
            if icm_gt_list:
                icm_accuracy = float(accuracy_score(icm_gt_list, icm_pred_list))
                icm_avg_prec = float(
                    precision_score(icm_gt_list, icm_pred_list, labels=labels, average="weighted", zero_division=0)
                )
                icm_avg_rec = float(
                    recall_score(icm_gt_list, icm_pred_list, labels=labels, average="weighted", zero_division=0)
                )
                icm_avg_f1 = float(
                    f1_score(icm_gt_list, icm_pred_list, labels=labels, average="weighted", zero_division=0)
                )
                icm_report = classification_report(icm_gt_list, icm_pred_list, labels=labels, zero_division=0)
            icm_metrics = _build_task_metrics(icm_gt_list, icm_pred_list, labels, total_matched, skipped_due_to_exp_rule)
            print(
                f"[ICM] accuracy={icm_accuracy:.4f} | avg-prec={icm_avg_prec:.4f} | "
                f"avg-rec={icm_avg_rec:.4f} | avg-f1={icm_avg_f1:.4f}"
            )
            print("Confusion Matrix (ICM):")
            print(icm_metrics["confusion_matrix"])
            print("Per-class recall (ICM):", icm_metrics["per_class_recall"])
            if icm_gt_list:
                print(f"ICM classification report:\n{icm_report}")
            elif task == "te":
                te_metrics = _build_task_metrics(te_gt_list, te_pred_list, labels, total_matched, skipped_due_to_exp_rule)
                print("Confusion Matrix (TE):")
                print(te_metrics["confusion_matrix"])
                print("Per-class recall (TE):", te_metrics["per_class_recall"])
                if te_gt_list:
                te_report = classification_report(te_gt_list, te_pred_list, labels=labels, zero_division=0)
                print(f"TE classification report:\n{te_report}")

            n_eval_used = len(icm_gt_list) if task == "icm" else len(te_gt_list)
            metrics_payload = {
                "task": task,
                "split": args.split,
                "checkpoint": str(args.checkpoint),
                "seed": args.seed,
                "device": str(device),
                "n_total_split_rows": int(len(split_df)),
                "n_eval_used": int(n_eval_used),
                "n_eval_total_before_filter": int(n_eval_total_before) if n_eval_total_before is not None else 0,
                "n_filtered_nd_na": int(removed_nd_na),
                "filtered_ratio": float(filtered_ratio),
                "eval_label_space": labels,
            }
            if task == "icm":
                metrics_payload["n_eval_used_icm"] = int(len(icm_gt_list))
                metrics_payload["icm"] = icm_metrics
                summary_msg = (
                    f"Evaluation complete | task={task} | split={args.split} | "
                    f"n_eval_used_icm={len(icm_gt_list)} | accuracy_icm={icm_metrics['accuracy']:.4f}"
                )
            else:
                metrics_payload["n_eval_used_te"] = int(len(te_gt_list))
                metrics_payload["te"] = te_metrics
                summary_msg = (
                    f"Evaluation complete | task={task} | split={args.split} | "
                    f"n_eval_used_te={len(te_gt_list)} | accuracy_te={te_metrics['accuracy']:.4f}"
                )

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    preds_df.to_csv(preds_file, index=False)

    print(f"Saved metrics to: {os.path.abspath(metrics_file)}")
    print(f"Saved predictions to: {os.path.abspath(preds_file)}")
    print(summary_msg)
    if task in {"icm", "te"} and removed_nd_na:
        print(
            f"[ICM/TE] Removed ND/NA from metrics: {removed_nd_na}/{n_eval_total_before} "
            f"({filtered_ratio:.2%}) excluded; coverage uses {len(icm_gt_list)} samples."
        )


if __name__ == "__main__":
    main()
