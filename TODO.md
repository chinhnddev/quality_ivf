# TODO: Update IVF-EffiMorphPP Pipeline for Gardner Experiments

## Tasks to Complete
- [ ] Modify src/dataset.py: Add val split, normalize labels, correct filtering for train/val/test
- [ ] Create src/losses.py: Implement cross_entropy, focal_loss, compute_class_weights
- [ ] Update src/train.py: Use splits_dir, val.csv for validation, add sanity checks
- [ ] Create scripts/train_gardner_single.py: CLI, training logic, output structure
- [ ] Create scripts/eval_gardner_single.py: CLI, evaluation with masking, save metrics/preds
- [ ] Test scripts: Run sample training/eval, verify acceptance tests
- [ ] Final: Provide commands for 6 experiments and markdown table template
