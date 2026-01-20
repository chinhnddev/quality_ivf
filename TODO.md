# Sanity Overfit Improvements TODO

## 1. Deterministic Transforms in Sanity Mode
- [x] Update GardnerDataset._build_transform to use Resize(224)+CenterCrop(224)+ToTensor+ImageNet Normalize for sanity_mode
- [x] Ensure train_transform == eval_transform in sanity mode

## 2. Disable Regularization in Sanity
- [x] Ensure dropout_p=0.0 for IVF_EffiMorphPP in sanity (already done)
- [x] Set label_smoothing=0.0 in make_loss_fn for sanity mode
- [x] Confirm mixup/cutmix not present (not needed)

## 3. Sanity Optimizer
- [x] Use Adam(lr=args.sanity_lr default 2e-3, wd=0) in sanity mode
- [x] No scheduler in sanity mode
- [x] Use sanity_epochs default 60

## 4. Fail-fast
- [x] After sanity training, evaluate deterministic train_acc on tiny set
- [x] If train_acc < 0.95, print diagnostics and exit(1)
- [x] Do NOT resume normal training if failed

## 5. Sanity Model Scale Preset
- [x] Add width_mult mapping: small=1.0, base=1.25, large=1.5 (or auto-bump to 2.0 if params <3M)
- [x] Print param count and channel config in sanity mode
- [x] Update model creation in sanity to use width_mult

## 6. Testing and Validation
- [x] Test the changes with example commands
- [x] Ensure backward compatibility
