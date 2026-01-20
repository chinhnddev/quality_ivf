# IVF_EffiMorphPP Training Improvements

This document summarizes the improvements made to the IVF blastocyst quality assessment training pipeline to support the Gardner benchmark_fair (EXP/ICM/TE tasks).

---

## Phase A: Pipeline Validation ✅

### A1: Label Encoding Validation
- **Added** `validate_label_encoding()` in `src/utils.py` to verify:
  - EXP: must use classes {0,1,2,3,4} (corresponding to grades 1,2,3,4,5)
  - ICM/TE: must use classes {0,1,2} (corresponding to A,B,C); ND/NA/empty are excluded
- **Logs** class distribution at startup with counts and percentages

### A2: Sanity Overfit Test Mode
- **CLI flag**: `--sanity_overfit` enables a quick validation test
- **Test procedure**:
  - Trains on 200 random samples from training split for 30 epochs
  - Asserts `train_acc > 0.95` (indicates model can memorize small dataset)
  - Prints debug info on failure: unique labels, batch shapes, sample images
  - Resets model weights before actual training
- **Usage**: `python scripts/train_gardner_single.py ... --sanity_overfit`

### A3: Evaluation Robustness
- ✅ Checkpoint loading handles both raw `state_dict` and Lightning-style `{"state_dict": ...}`
- ✅ Masking uses positional indices (not DataFrame index)
- ✅ Length assertions prevent silent misalignment
- ✅ Fixed sys.path for module imports

---

## Phase B: Imbalance Fix ✅

### B4: Class-Balanced Training
- **Weights** computed as: `weight[c] = N_total / (num_classes * N_c)`
- **Logged** at startup showing weight for each class
- **Applied** automatically when `use_class_weights: true` in config
- **Handles** missing classes gracefully

### B5: Per-Class Metric Logging
- **Enhanced** `evaluate_on_val()` to return per-class metrics:
  - Precision, recall, F1, support for each class
  - Still logs overall accuracy, macro F1, weighted F1
- **Useful** for understanding imbalance-induced bias

---

## Phase C: Optimization & Training ✅

### C6: Optimizer & Schedule Alignment
- **Optimizer**: Switched to `AdamW` with `weight_decay=1e-4` (was Adam with wd=0)
- **Learning Rate**: Changed from `5e-4` → `1e-4` (more stable)
- **Schedule**: 
  - Cosine annealing with linear warmup
  - Warmup: 5 epochs (configurable)
  - Updates LR after each batch for finer control
  
**Updated base.yaml:**
```yaml
optimizer:
  name: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-4

scheduler:
  name: cosine
  warmup_epochs: 5
```

---

## Phase C (continued): Augmentation & Regularization

### C7: Augmentation (Planned)
- To be enhanced with:
  - Random resized crop (scale/aspect ratio)
  - Rotation ±15°
  - Horizontal/vertical flip
  - Mild color jitter (brightness, contrast, saturation)
  - Optional Gaussian blur

### C8: Regularization (Planned)
- Label smoothing (0.05-0.1)
- Dropout in classifier head (0.2-0.5)
- Optional stochastic depth in residual blocks

---

## Phase D: Model Architecture (Planned)

### D9: IVF_EffiMorphPP Improvements
- Currently: 4 stages (MultiScaleBlock → DW+PW chains)
- Proposed variants:
  1. **v0** (baseline): Current architecture
  2. **v1** (+ training tuning): No architecture change, just better training
  3. **v2** (+ light improvements): Add ECA attention, layer scaling, or improved head

### D10: Model Variants Config
- Will add `model.variant: v0 | v1 | v2` to config
- Load appropriate model dynamically in training script

---

## Phase E: Automation (Partial)

### E11: Enhanced CLI Flags
- ✅ `--sanity_overfit`: Quick validation test
- Planned:
  - `--save_preds`: Export predictions CSV for analysis
  - `--task`: Select task (exp/icm/te) from CLI

### E12: Result Summary (Planned)
- Print at end of training:
  - Best epoch, best macro F1, weighted F1
  - Compare against previous run (if exists)
  - Confusion matrix for debugging

---

## How to Train Now

### Standard Training (EXP task, benchmark_fair track)
```bash
python scripts/train_gardner_single.py \
  --config configs/gardner/base.yaml \
  --task_cfg configs/gardner/tasks/exp.yaml \
  --track_cfg configs/gardner/tracks/benchmark_fair.yaml
```

### With Sanity Test First
```bash
python scripts/train_gardner_single.py \
  --config configs/gardner/base.yaml \
  --task_cfg configs/gardner/tasks/exp.yaml \
  --track_cfg configs/gardner/tracks/benchmark_fair.yaml \
  --sanity_overfit
```

### Override LR (for sweep)
```bash
python scripts/train_gardner_single.py \
  --config configs/gardner/base.yaml \
  --task_cfg configs/gardner/tasks/exp.yaml \
  --track_cfg configs/gardner/tracks/benchmark_fair.yaml \
  --override optimizer.lr=3.0e-4
```

### Evaluation
```bash
python scripts/eval_gardner_single.py \
  --task exp \
  --checkpoint outputs/benchmark_fair/exp/best.ckpt \
  --splits_dir splits \
  --out_dir outputs/eval_exp
```

---

## Expected Improvements

With these changes, we expect:
1. **Better reproducibility**: Explicit label validation catches encoding bugs early
2. **Faster convergence**: AdamW + warmup schedule + proper LR
3. **Fairer evaluation**: Class weighting balances imbalance in EXP (especially class 3)
4. **Better debugging**: Per-class metrics show which classes are hard
5. **Production ready**: Sanity test validates model can learn before full training

---

## Files Modified

- `src/utils.py`: Added label validation utilities
- `src/model.py`: Fixed MultiScaleBlock channel mismatch
- `scripts/train_gardner_single.py`: Added sanity test, class weighting logging, AdamW, warmup scheduler
- `scripts/eval_gardner_single.py`: Already robust (verified)
- `configs/gardner/base.yaml`: Updated optimizer, LR, scheduler
- `requirements.txt`: Added torchinfo, omegaconf
- `.gitignore`: Added data/ folder
- `README.md` (this file): Documentation

---

## Next Steps

1. **Train with sanity test**: Verify model can overfit
2. **Monitor metrics**: Watch per-class F1 for imbalance issues
3. **Tune LR**: Try {5e-4, 3e-4, 1e-4} for each task
4. **Add augmentation**: Phase C7 improvements
5. **Try model variants**: Phase D9-10 (if still needed)

---

**Status**: Phase A-C complete. Phase D-E planned for future iteration.
