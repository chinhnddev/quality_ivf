# src/utils.py
"""
Utility functions for Gardner EXP / ICM / TE experiments.

Used by:
- train_gardner_single.py
- eval_gardner_single.py

Design principles:
- No hidden magic
- Explicit label handling
- Safe sanity checks (fail fast)
"""

import random
import numpy as np
import torch
import pandas as pd
from typing import Iterable, List, Dict


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for python, numpy, torch.

    Args:
        seed: random seed
        deterministic: whether to enforce deterministic CuDNN
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Label normalization / validation
# ============================================================

def normalize_exp_token(x) -> str:
    """
    Normalize EXP tokens to one of '0','1','2','3','4','NA'.
    Invalid / empty -> 'NA'.
    """
    if pd.isna(x):
        return "NA"
    try:
        if isinstance(x, (int, np.integer)):
            val = int(x)
        elif isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return "NA"
            val = int(float(x))
        else:
            val = int(float(str(x).strip()))
        if val in {0, 1, 2, 3, 4}:
            return str(val)
    except Exception:
        pass
    text = str(x).strip().upper()
    if text == "NA":
        return "NA"
    return "NA"


def normalize_icm_te_token(x) -> str:
    """
    Normalize ICM/TE to one of '0','1','2','ND','NA'.
    Numeric -1 -> 'NA', 3 -> 'ND', invalid -> 'ND'.
    """
    if pd.isna(x):
        return "ND"
    try:
        if isinstance(x, (int, np.integer)):
            val = int(x)
        elif isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return "ND"
            val = int(float(x))
        else:
            val = int(float(str(x).strip()))
        if val == -1:
            return "NA"
        if val == 3:
            return "ND"
        if val in {0, 1, 2}:
            return str(val)
    except Exception:
        pass
    text = str(x).strip().upper()
    if text in {"ND", "NA"}:
        return text
    return "ND"


def normalize_token(x) -> str:
    """Backward-compatible alias used by some helpers (maps to ICM/TE tokens)."""
    return normalize_icm_te_token(x)


def assert_valid_labels(
    labels: Iterable,
    valid_set: Iterable,
    task: str,
    split: str
) -> None:
    """
    Assert all labels belong to valid_set.
    Fail fast if dataset is corrupted.

    Args:
        labels: iterable of labels
        valid_set: allowed label values
        task: exp / icm / te
        split: train / val / test
    """
    valid_set = set(valid_set)
    uniq = set(labels)
    bad = uniq - valid_set
    if bad:
        raise ValueError(
            f"[ERROR] Invalid labels detected for task={task}, split={split}: {bad}. "
            f"Allowed set={valid_set}"
        )


# ============================================================
# Dataset statistics
# ============================================================

def print_label_distribution(
    df: pd.DataFrame,
    label_col: str,
    task: str,
    split: str
) -> None:
    """
    Print label distribution for debugging / sanity check.
    """
    print(f"\n[{split}] Label distribution for task={task}")

    if task == "exp":
        raw_vals = df[label_col].dropna().astype(str).str.strip()
        raw_unique = sorted({val if val != "" else "<EMPTY>" for val in raw_vals.unique()})
        print(f"{label_col} raw unique values: {raw_unique}")
        tokens = df[label_col].apply(normalize_exp_token)
        counts = tokens.value_counts().sort_index()
        valid = counts.loc[[str(i) for i in range(5)]].sum() if not counts.empty else 0
        na = counts.get("NA", 0)
        print(f"EXP tokens: total={len(tokens)}, valid(0-4)={valid}, NA={na}")
        print("EXP counts:", counts.to_dict())
    else:
        raw_vals = df[label_col].fillna("").astype(str).str.strip()
        raw_unique = sorted({val if val != "" else "<EMPTY>" for val in raw_vals.unique()})
        tokens = df[label_col].apply(normalize_icm_te_token)
        total = len(tokens)
        valid = tokens.isin(["0", "1", "2"]).sum()
        nd = (tokens == "ND").sum()
        na = (tokens == "NA").sum()
        print(f"{label_col} raw unique values: {raw_unique}")
        print(f"{label_col}: total={total}, valid(0/1/2)={valid}, ND={nd}, NA={na}")
        valid_counts = tokens[tokens.isin(["0", "1", "2"])].value_counts().sort_index()
        print("Valid distribution:", valid_counts.to_dict())


# ============================================================
# Evaluation masking
# ============================================================

def build_eval_mask(
    labels: Iterable,
    task: str
) -> np.ndarray:
    """
    Build boolean mask for evaluation.

    Rules:
    - EXP: evaluate ALL samples
    - ICM / TE: evaluate only labels in {0,1,2}

    Args:
        labels: iterable of raw labels
        task: exp / icm / te

    Returns:
        np.ndarray of shape (N,) dtype=bool
    """
    labels = list(labels)

    if task == "exp":
        return np.ones(len(labels), dtype=bool)

    mask = []
    for y in labels:
        tok = normalize_token(y)
        mask.append(tok in {"0", "1", "2"})
    return np.array(mask, dtype=bool)


# ============================================================
# Metric helpers
# ============================================================

def masked_accuracy(
    y_true: List[int],
    y_pred: List[int],
    mask: np.ndarray
) -> float:
    if mask.sum() == 0:
        return 0.0
    y_t = np.array(y_true)[mask]
    y_p = np.array(y_pred)[mask]
    return float((y_t == y_p).mean())


def masked_macro_f1(
    y_true: List[int],
    y_pred: List[int],
    mask: np.ndarray
) -> float:
    from sklearn.metrics import f1_score
    if mask.sum() == 0:
        return 0.0
    y_t = np.array(y_true)[mask]
    y_p = np.array(y_pred)[mask]
    return float(f1_score(y_t, y_p, average="macro"))


def masked_weighted_f1(
    y_true: List[int],
    y_pred: List[int],
    mask: np.ndarray
) -> float:
    from sklearn.metrics import f1_score
    if mask.sum() == 0:
        return 0.0
    y_t = np.array(y_true)[mask]
    y_p = np.array(y_pred)[mask]
    return float(f1_score(y_t, y_p, average="weighted"))


# ============================================================
# Pipeline validation
# ============================================================

def validate_label_encoding(df: pd.DataFrame, task: str) -> Dict[str, int]:
    """
    Validate label encoding matches spec for a given task.
    
    Rules:
    - EXP: must have values in {0,1,2,3,4} (grades 1,2,3,4,5)
    - ICM/TE: valid values {0,1,2} (A,B,C); ND/NA/empty are excluded but not errors
    
    Args:
        df: DataFrame with label columns
        task: 'exp', 'icm', or 'te'
        
    Returns:
        dict with counts: {'class_0': N, 'class_1': N, ...}
        
    Raises:
        ValueError if encoding is invalid
    """
    task = task.lower()
    if task == "exp":
        exp_vals = df["EXP"].dropna()
        exp_numeric = pd.to_numeric(exp_vals, errors="coerce")
        bad = exp_numeric[exp_numeric.isna()].index.tolist()
        if len(bad) > 0:
            raise ValueError(f"[{task}] Non-numeric EXP values at indices {bad}: {exp_vals.loc[bad].tolist()}")
        
        valid = set([0, 1, 2, 3, 4])
        uniq = set(exp_numeric.astype(int).unique())
        invalid = uniq - valid
        if invalid:
            raise ValueError(f"[{task}] Invalid EXP classes {invalid}. Expected {{0,1,2,3,4}}")
        
        counts = exp_numeric.astype(int).value_counts().sort_index().to_dict()
        return counts
    
    elif task in ["icm", "te"]:
        col = task.upper()
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        
        tokens = df[col].apply(normalize_token)
        valid_mask = tokens.isin(["0", "1", "2"])
        valid_tokens = tokens[valid_mask]
        
        if len(valid_tokens) == 0:
            raise ValueError(f"[{task}] No valid labels (0/1/2) found. Found: {set(tokens.unique())}")
        
        counts = valid_tokens.astype(int).value_counts().sort_index().to_dict()
        n_total = len(tokens)
        n_valid = len(valid_tokens)
        n_excluded = n_total - n_valid
        
        print(f"[{task}] Total: {n_total} | Valid(0/1/2): {n_valid} | Excluded: {n_excluded}")
        return counts
    
    else:
        raise ValueError(f"Unknown task: {task}")

