from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def _map_consensus_label(value: Any) -> int:
    if pd.isna(value):
        return -1
    text = str(value).strip()
    if text == "":
        return -1
    upper = text.upper()
    if upper == "NA":
        return -1
    if upper == "ND":
        return 3
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Unable to parse consensus label '{value}'.") from exc


def _map_pred_exp(value: Any) -> int:
    if pd.isna(value):
        return 5
    try:
        candidate = int(float(value))
    except Exception:
        return 5
    return candidate if candidate in {0, 1, 2, 3, 4} else 5


def _map_pred_icm_te(value: Any) -> int:
    if pd.isna(value):
        return 3
    try:
        candidate = int(float(value))
    except Exception:
        return 3
    return candidate if candidate in {0, 1, 2} else 3


def _find_column_case_insensitive(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    name_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        normalized = candidate.lower()
        if normalized in name_map:
            return name_map[normalized]
    return None


def _resolve_image_column(df: pd.DataFrame) -> str:
    candidates = ["Image", "image", "Filename", "filename"]
    image_col = _find_column_case_insensitive(df, candidates)
    if image_col is None:
        raise KeyError(f"Could not find an image column in DataFrame columns: {df.columns.tolist()}")
    return image_col


def _build_classification_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Optional[List[int]] = None,
) -> Tuple[str, Dict[str, Any], List[List[int]]]:
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    if not y_true_list:
        return (
            "No samples available for this split/task.",
            {"warning": "empty"},
            [],
        )
    if labels is None:
        labels = sorted(set(y_true_list) | set(y_pred_list))
    report_str = classification_report(
        y_true_list, y_pred_list, labels=labels, zero_division=0
    )
    report_dict = classification_report(
        y_true_list, y_pred_list, labels=labels, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true_list, y_pred_list, labels=labels)
    return report_str, report_dict, cm.tolist()


class PaperEvaluator:
    PRED_CANDIDATES = {
        "exp": ["exp_pred", "EXP_pred", "pred_exp", "pred_EXP", "y_pred_exp", "y_pred_EXP"],
        "icm": ["icm_pred", "ICM_pred", "pred_icm", "pred_ICM", "y_pred_icm"],
        "te": ["te_pred", "TE_pred", "pred_te", "pred_TE", "y_pred_te"],
    }
    TASK_TO_DEFAULT_PRED = {"exp": "y_pred_raw", "icm": "y_pred_raw", "te": "y_pred_raw"}

    def __init__(
        self,
        preds_df: pd.DataFrame,
        consensus_df: pd.DataFrame,
        task: str,
        explicit_pred_cols: Optional[Dict[str, str]] = None,
        consensus_img_col: Optional[str] = None,
        preds_img_col: Optional[str] = None,
        pred_csv_path: Optional[str] = None,
    ) -> None:
        self.pred_csv_path = pred_csv_path
        self.task = task.lower()
        self.explicit_pred_cols = explicit_pred_cols or {}
        self.consensus_df = self._prepare_dataframe(consensus_df, consensus_img_col, True)
        self.preds_df = self._prepare_dataframe(preds_df, preds_img_col, False)
        self.total_consensus_rows = len(self.consensus_df)
        self.total_pred_rows = len(self.preds_df)
        self.prediction_columns: Dict[str, str] = {}
        self._resolve_prediction_columns()
        self.merged = self._merge_records()
        self._map_labels()

    def _prepare_dataframe(
        self,
        df: pd.DataFrame,
        image_col: Optional[str],
        is_consensus: bool,
    ) -> pd.DataFrame:
        if image_col is None:
            image_col = _resolve_image_column(df)
        df = df.rename(columns={image_col: "Image"})
        df = df.copy()
        df["Image"] = df["Image"].astype(str).str.strip()
        return df.drop_duplicates(subset=["Image"])

    def _resolve_prediction_columns(self) -> None:
        lower_cols = {col.lower(): col for col in self.preds_df.columns}
        for key in ["exp", "icm", "te"]:
            explicit = self.explicit_pred_cols.get(key)
            if explicit:
                if explicit not in self.preds_df.columns:
                    raise KeyError(f"Explicit prediction column '{explicit}' for {key.upper()} not found.")
                self.prediction_columns[key] = explicit
                continue
            candidates = self.PRED_CANDIDATES.get(key, [])
            found = _find_column_case_insensitive(self.preds_df, candidates)
            if found:
                self.prediction_columns[key] = found
                continue
            if key == self.task and self.TASK_TO_DEFAULT_PRED.get(key) in lower_cols:
                self.prediction_columns[key] = lower_cols[self.TASK_TO_DEFAULT_PRED[key]]
        if "exp" not in self.prediction_columns:
            raise KeyError("Could not resolve EXP prediction column required for paper evaluation.")

    def _merge_records(self) -> pd.DataFrame:
        consensus = self.consensus_df.rename(columns={"EXP": "EXP_gt", "ICM": "ICM_gt", "TE": "TE_gt"})
        merged = pd.merge(consensus, self.preds_df, on="Image", how="inner", suffixes=("_gt", ""))
        if merged.empty:
            raise ValueError("No matching rows found between consensus and prediction CSVs.")
        return merged

    def _map_labels(self) -> None:
        self.merged["EXP_gt_mapped"] = self.merged["EXP_gt"].apply(_map_consensus_label)
        self.merged["ICM_gt_mapped"] = self.merged["ICM_gt"].apply(_map_consensus_label)
        self.merged["TE_gt_mapped"] = self.merged["TE_gt"].apply(_map_consensus_label)
        for key, col in self.prediction_columns.items():
            if key == "exp":
                self.merged[f"exp_pred_mapped"] = self.merged[col].apply(_map_pred_exp)
            else:
                self.merged[f"{key}_pred_mapped"] = self.merged[col].apply(_map_pred_icm_te)

    def evaluate(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        total_matched = len(self.merged)
        exp_gt = self.merged["EXP_gt_mapped"]
        exp_pred = self.merged["exp_pred_mapped"]
        skip_mask = exp_gt.isin({0, 1}) | exp_pred.isin({0, 1})
        skipped_filenames = self.merged.loc[skip_mask, "Image"].astype(str).tolist()
        filtered = self.merged.loc[~skip_mask]
        reports: Dict[str, str] = {}
        metrics: Dict[str, Any] = {
            "eval_protocol": "paper",
            "total_matched": int(total_matched),
            "skipped_due_to_exp_rule": int(skip_mask.sum()),
            "skipped_filenames": skipped_filenames,
            "n_total_consensus_rows": int(self.total_consensus_rows),
            "n_total_pred_rows": int(self.total_pred_rows),
            "pred_csv": self.pred_csv_path,
        }

        report_str, report_dict, cm = _build_classification_report(
            exp_gt, exp_pred, labels=sorted(set(exp_gt.tolist()) | set(exp_pred.tolist()))
        )
        reports["exp"] = report_str
        metrics["n_eval_used_exp"] = int(total_matched)
        metrics["exp"] = {
            "classification_report": report_dict,
            "confusion_matrix": cm,
        }

        if "icm" in self.prediction_columns:
            icm_true = filtered["ICM_gt_mapped"].tolist()
            icm_pred = filtered["icm_pred_mapped"].tolist()
            report_str, report_dict, cm = _build_classification_report(
                icm_true, icm_pred, labels=[0, 1, 2, 3]
            )
            reports["icm"] = report_str
            metrics["n_eval_used_icm"] = int(len(icm_true))
            metrics["icm"] = {
                "classification_report": report_dict,
                "confusion_matrix": cm,
            }
        else:
            metrics["n_eval_used_icm"] = 0

        if "te" in self.prediction_columns:
            te_true = filtered["TE_gt_mapped"].tolist()
            te_pred = filtered["te_pred_mapped"].tolist()
            report_str, report_dict, cm = _build_classification_report(
                te_true, te_pred, labels=[0, 1, 2, 3]
            )
            reports["te"] = report_str
            metrics["n_eval_used_te"] = int(len(te_true))
            metrics["te"] = {
                "classification_report": report_dict,
                "confusion_matrix": cm,
            }
        else:
            metrics["n_eval_used_te"] = 0

        return metrics, reports
