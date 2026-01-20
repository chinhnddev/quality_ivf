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

def normalize_token(x) -> str:
    """
    Normalize label tokens (mainly for ICM / TE).

    Output is one of:
        '0', '1', '2', 'ND', 'NA', ''

    Used BEFORE filtering / masking.
    """
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if float(x).is_integer():
            return str(int(x))
        return ""
    s = str(x).strip()
    if s == "":
        return ""
    s_up = s.upper()
    if s_up in {"ND", "NA"}:
        return s_up
    if s in {"0", "1", "2", "3", "4"}:
        return s
    return ""


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
        counts = df[label_col].value_counts().sort_index()
        print("EXP counts:", counts.to_dict())
    else:
        tokens = df[label_col].apply(normalize_token)
        total = len(tokens)
        valid = tokens.isin(["0", "1", "2"]).sum()
        nd = (tokens == "ND").sum()
        na = (tokens == "NA").sum()
        empty = (tokens == "").sum()

        print(
            f"{label_col}: total={total}, "
            f"valid(0/1/2)={valid}, ND={nd}, NA={na}, empty={empty}"
        )

        if valid > 0:
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

