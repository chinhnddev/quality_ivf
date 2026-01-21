# TODO: Fix Evaluation and Training Diagnostics for Gardner EXP

## 1. Fix Eval Output (scripts/eval_gardner_single.py)
- [x] Add --split flag (default 'test')
- [x] Ensure out_dir creation and save metrics_<split>.json, preds_<split>.csv
- [x] Print absolute paths after saving
- [x] Add defensive checks for checkpoint and csv existence
- [x] Handle dataset length mismatch by returning image_name from dataset

## 2. Add y_pred Distribution & Confusion Matrix (Eval)
- [x] Print y_pred counts and normalized ratios after predictions
- [x] Print confusion matrix and per-class recall using valid mask
- [x] Save confusion matrix in metrics json

## 3. Add Train-Time Diagnostics (scripts/train_gardner_single.py)
- [x] Print startup info: task, num_classes, train/val sizes, class distributions
- [x] Print loss settings confirmation (loss_name, label_smoothing, use_class_weights)
- [x] Print class weights tensor if applied
- [x] Add validation-time y_pred distribution logging each epoch
- [x] Ensure EXP uses CrossEntropyLoss with label_smoothing (PyTorch supports it)
- [x] Write debug_report.txt after each epoch with diagnostics

## 4. Diagnose Low Performance (Auto-Report)
- [x] Write debug_report.txt with best epoch, y_pred distribution, confusion matrix
- [x] Include class weights, sampler info, current LR
- [x] Detect majority-class collapse (>80% on single class)

## 5. Optional: Weighted Random Sampler
- [x] Add data.use_weighted_sampler config flag
- [x] Implement WeightedRandomSampler for train DataLoader if enabled
- [x] Print sampler weights when used
