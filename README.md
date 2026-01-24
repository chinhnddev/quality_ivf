"# quality_ivf"

## Key commands

1. **Train ICM or TE** using existing splits (adjust config paths as needed):
   ```sh
   python scripts/train_gardner_single.py --config configs/base.yaml \
       --task_cfg configs/task/icm.yaml --track_cfg configs/track/improved.yaml
   ```
   Add `--train_icmte_exclude_na_nd 1` if you want to drop class 3 (NA/ND) from the ICM/TE training set while keeping evaluation unchanged.
2. **Evaluate with the paper protocol** (matches `calculate_model_metrics.py`):
   ```sh
   python scripts/eval_gardner_single.py --task icm --eval_protocol paper \
       --consensus_csv annotations/test_rev.csv --checkpoint outputs/improved/icm/best.ckpt \
       --splits_dir splits --out_dir outputs/improved/icm/eval
   ```
3. **Parity-only runner** for a pre-exported predictions CSV:
   ```sh
   python scripts/gardner/paper_eval.py --pred_csv outputs/improved/icm/preds_test.csv \
       --consensus_csv annotations/test_rev.csv --task icm --split test
   ```
