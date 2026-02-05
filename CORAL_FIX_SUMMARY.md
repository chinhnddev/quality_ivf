# CORAL Bug Fix Summary

## Problem
The training script incorrectly applied CORAL (Consistent Rank Logits) ordinal regression to the TE (Trophectoderm) task, which should use standard nominal classification instead. This caused the model to output only 2 logits instead of 3, making it impossible to predict class 2.

## Root Cause
CORAL is only appropriate for the EXP task (ordinal: grade 1→2→3→4→5). The TE task is nominal classification (A/B/C with no inherent ordering) and should use CrossEntropyLoss or FocalLoss.

## Changes Made

### 1. CORAL Validation Warning (lines 426-430)
Added validation to warn users when `--use_coral=1` is used with non-EXP tasks:
```python
# Validate CORAL usage
if args.use_coral and task != "exp":
    print(f"[WARNING] --use_coral=1 is only valid for EXP task (ordinal regression).")
    print(f"[WARNING] Task '{task}' uses nominal classification. CORAL will be disabled.")
    use_coral = False
```

### 2. Safety Checks for Nominal Classification Tasks (lines 518-524)
Added sanity checks for ICM/TE tasks to verify correct number of output logits:
```python
elif task in ["icm", "te"]:
    # Sanity check for nominal classification tasks
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = model(dummy_input)
        assert dummy_output.shape[1] == num_classes, f"Expected {num_classes} logits for {task.upper()}, got {dummy_output.shape[1]}"
    print(f"[{task.upper()}] Model outputs {dummy_output.shape[1]} logits for nominal classification")
```

### 3. Note on Existing Fix (line 423)
The primary fix was already in place:
```python
use_coral = bool(args.use_coral) and task == "exp"
```
This ensures CORAL is only enabled for EXP tasks regardless of the `--use_coral` flag.

## Tests Added

### test_coral_fix.py
Unit tests validating model output shapes for different task/CORAL combinations:
- TE task (3 classes) outputs 3 logits without CORAL ✓
- ICM task (4 classes) outputs 4 logits without CORAL ✓  
- EXP task (5 classes) outputs 4 logits with CORAL ✓
- EXP task (5 classes) outputs 5 logits without CORAL ✓

### test_coral_integration.py
Integration tests for the training script:
- Validates warning message appears when using CORAL with TE task
- Validates CORAL works correctly for EXP task
- (Skips if data files are not available)

## Expected Results After Fix

When training TE model:
- Model outputs 3 logits (not 2)
- Loss function is FocalLoss (gamma=2.0) as configured
- Model can predict all 3 classes (0, 1, 2)
- No "[CORAL]" messages in logs for TE task
- "[TE] Model outputs 3 logits for nominal classification" message appears

When accidentally using `--use_coral=1` with TE:
- Warning message is displayed
- CORAL is automatically disabled
- Training proceeds normally with nominal classification

## Files Changed
1. `scripts/train_gardner_single.py` - Added validation warning and safety checks
2. `test_coral_fix.py` - Unit tests for model output validation
3. `test_coral_integration.py` - Integration tests for training script

## Testing Commands

Run unit tests:
```bash
python test_coral_fix.py
```

Run integration tests (requires data):
```bash
python test_coral_integration.py
```

Train TE model with fix:
```bash
python scripts/train_gardner_single.py \
    --config configs/gardner/base.yaml \
    --task_cfg configs/gardner/tasks/te.yaml \
    --track_cfg configs/gardner/tracks/improved.yaml \
    --use_coral 0 \
    --use_weighted_sampler 1 \
    --out_dir outputs/improved/te_fixed
```
