#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd  # noqa: E402
from src.paper_eval import PaperEvaluator  # noqa: E402


def _resolve_consensus_csv(consensus_csv: str, splits_dir: str) -> str:
    if consensus_csv and os.path.exists(consensus_csv):
        return consensus_csv
    fallback = os.path.join(splits_dir, "test.csv") if splits_dir else None
    if fallback and os.path.exists(fallback):
        print(f"[PAPER EVAL] Consensus CSV not found at {consensus_csv}; using fallback {fallback}")
        return fallback
    raise FileNotFoundError(f"Consensus CSV missing at {consensus_csv} and no fallback available.")


def _split_length(splits_dir: str, split: str) -> int:
    path = Path(splits_dir) / f"{split}.csv"
    if path.exists():
        df = pd.read_csv(path)
        return len(df)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run Gardner paper-style evaluation on exported predictions.")
    parser.add_argument("--pred_csv", required=True, help="Predictions CSV with columns like exp_pred/icm_pred/te_pred.")
    parser.add_argument("--consensus_csv", default="annotations/test_rev.csv",
                        help="Consensus CSV (fallback to splits/test.csv when missing).")
    parser.add_argument("--splits_dir", default="splits", help="Directory containing train/val/test split CSVs.")
    parser.add_argument("--task", choices=["exp", "icm", "te"], default="icm", help="Primary task represented in the preds file.")
    parser.add_argument("--split", default="test", help="Split name for metadata (default test).")
    parser.add_argument("--out_dir", default="outputs/paper_eval", help="Where to write the parity metrics.")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    consensus_csv = _resolve_consensus_csv(args.consensus_csv, args.splits_dir)
    preds_df = pd.read_csv(args.pred_csv)
    consensus_df = pd.read_csv(consensus_csv)

    evaluator = PaperEvaluator(
        preds_df,
        consensus_df,
        task=args.task,
        pred_csv_path=args.pred_csv,
    )
    metrics_payload, reports = evaluator.evaluate()
    metrics_payload.update(
        {
            "task": args.task,
            "split": args.split,
            "consensus_csv": consensus_csv,
            "pred_csv": args.pred_csv,
            "splits_dir": args.splits_dir,
            "eval_protocol": "paper",
        }
    )
    split_len = _split_length(args.splits_dir, args.split)
    if split_len:
        metrics_payload["n_total_split_rows"] = int(split_len)

    for name in ("exp", "icm", "te"):
        report = reports.get(name)
        if report:
            print(f"\n[PAPER EVAL] {name.upper()} classification report:\n{report}")

    summary_msg = (
        f"Paper parity run | task={args.task} | matched={metrics_payload['total_matched']} | "
        f"icm_used={metrics_payload.get('n_eval_used_icm')} | te_used={metrics_payload.get('n_eval_used_te')}"
    )
    print(summary_msg)

    metrics_path = out_dir / f"paper_metrics_{args.split}.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2, ensure_ascii=False)
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
