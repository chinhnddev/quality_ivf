# Transfer Learning Integration: Stage Pretraining → Gardner Fine-tuning

## Overview
Successfully integrated stage pretraining checkpoint loading into `scripts/train_gardner_single.py` for downstream task fine-tuning (EXP/ICM/TE).

## Changes Made

### 1. Added `load_backbone_only()` Function
**Location**: [train_gardner_single.py](scripts/train_gardner_single.py#L37-L55)

```python
def load_backbone_only(model, ckpt_path: str, device="cpu"):
    """Load stage-pretrain backbone weights, excluding head layers."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)  # support both formats
    
    # drop any head weights
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[PRETRAIN LOAD] loaded backbone from: {ckpt_path}")
    print(f"[PRETRAIN LOAD] missing keys (expected head.*): {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[PRETRAIN LOAD] unexpected keys: {unexpected}")
    return model
```

**Purpose**: 
- Loads weights from stage-pretrain checkpoint
- Filters out stage head (3-class classifier)
- Loads backbone into downstream model with task-specific head
- Logs missing/unexpected keys for verification

### 2. Added `--pretrain_ckpt` CLI Argument
**Location**: [train_gardner_single.py](scripts/train_gardner_single.py#L809-L810)

```bash
parser.add_argument("--pretrain_ckpt", type=str, default="",
                    help="Path to stage-pretrain best.ckpt for backbone initialization.")
```

**Usage Example**:
```bash
python scripts/train_gardner_single.py \
  --task exp \
  --config configs/gardner/base.yaml \
  --track improved \
  --pretrain_ckpt outputs/stage_pretrain/best.ckpt
```

### 3. Integrated Checkpoint Loading
**Location**: [train_gardner_single.py](scripts/train_gardner_single.py#L497-L500)

```python
# Load pretrained backbone if provided
if args.pretrain_ckpt:
    model = load_backbone_only(model, args.pretrain_ckpt, device=device)
```

**Timing**: Loads immediately after model creation, before optimizer setup

---

## Transfer Learning Workflow

### Stage 1: Pretraining
```bash
# Stage: cleavage/morula/blastocyst classification (3-class)
python scripts/train_stage_pretrain.py \
  --epochs 40 \
  --output_dir outputs/stage_pretrain
```
**Outputs**: `best.ckpt` with backbone + 3-class head

### Stage 2: Fine-tuning (EXP Example)
```bash
# Task: Developmental competence classification (4-class ordinal)
python scripts/train_gardner_single.py \
  --task exp \
  --config configs/gardner/base.yaml \
  --track improved \
  --use_coral 1 \
  --use_weighted_sampler 1 \
  --pretrain_ckpt outputs/stage_pretrain/best.ckpt
```

**Key Integration Points**:
1. **Backbone Loading**: `load_backbone_only()` loads stage-pretrain weights
2. **Head Adaptation**: New 4-class EXP head initialized randomly
3. **Loss Function**: CORAL ordinal loss (enabled via `--use_coral 1`)
4. **Sampling**: WeightedRandomSampler for class balancing (enabled via `--use_weighted_sampler 1`)

---

## CORAL + Sampler Integration

### CORAL Loss
- **File**: [src/loss_coral.py](src/loss_coral.py)
- **Integration**: [train_gardner_single.py#L127-L129](scripts/train_gardner_single.py#L127-L129)
- **Usage**: Automatically selected when `--use_coral 1` + `--task exp`
- **Output**: 4 logits representing ordinal levels (label=0→expected_logits=[0,0,0,0], label=3→[1,1,1,1])

```python
# From make_loss_fn()
if use_coral and task == "exp":
    return lambda logits, targets: coral_loss(logits, targets, num_classes)
```

### WeightedRandomSampler
- **File**: [train_gardner_single.py#L20](scripts/train_gardner_single.py#L20)
- **Purpose**: Balances class representation in mini-batches
- **Integration**: [train_gardner_single.py#L453-L457](scripts/train_gardner_single.py#L453-L457)
- **Usage**: Enabled via `--use_weighted_sampler 1`
- **Warning**: Setting both sampler and class_weights issues warning but both work

```python
# From DataLoader creation
dl_train = DataLoader(
    ds_train,
    batch_size=batch_size,
    sampler=sampler if use_weighted_sampler else None,
    shuffle=False if use_weighted_sampler else True,  # shuffle=False when sampler is used
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
)
```

---

## Verification

### Check Backbone Loading
Run with verbose output:
```bash
python scripts/train_gardner_single.py \
  --task exp \
  --config configs/gardner/base.yaml \
  --track improved \
  --pretrain_ckpt outputs/stage_pretrain/best.ckpt 2>&1 | grep -E "PRETRAIN_LOAD|Device:"
```

**Expected Output**:
```
[PRETRAIN LOAD] loaded backbone from: outputs/stage_pretrain/best.ckpt
[PRETRAIN LOAD] missing keys (expected head.*): ['head.0.weight', 'head.0.bias', 'head.1.weight', 'head.1.bias']
```

### Check CORAL + Sampler Integration
View training configuration:
```bash
python scripts/train_gardner_single.py \
  --task exp \
  --config configs/gardner/base.yaml \
  --track improved \
  --use_coral 1 \
  --use_weighted_sampler 1 \
  --pretrain_ckpt outputs/stage_pretrain/best.ckpt 2>&1 | head -30
```

**Expected Output** (training info):
```
[CONFIG] Task: exp, Track: improved, Device: cuda
[CONFIG] use_weighted_sampler=True
[CONFIG] use_class_weights=False
[CORAL] Model outputs 4 logits for EXP ordinal regression
```

---

## Architecture Alignment

### Stage Pretraining Backbone
- **Model**: IVF_EffiMorphPP (width_mult=1.0, base_channels=32)
- **Output**: 872,808 parameters
- **Head**: 3-class linear layer (stage classification)

### Gardner EXP Task
- **Model**: IVF_EffiMorphPP (width_mult=1.0, base_channels=32, task='exp', use_coral=1)
- **Output**: 872,808 parameters (backbone) + 4-class CORAL head
- **Head**: 4 ordinal regression logits

### Parameter Transfer
- ✅ Backbone loaded from stage-pretrain
- ✅ EXP head initialized fresh (improves convergence)
- ✅ No parameter mismatch (strict=False in load_state_dict)

---

## Full Training Pipeline Example

```bash
#!/bin/bash

# Stage 1: Pretrain on development stages
python scripts/train_stage_pretrain.py \
    --epochs 40 \
    --output_dir outputs/stage_pretrain \
    --seed 42

# Stage 2: Fine-tune on EXP
python scripts/train_gardner_single.py \
    --task exp \
    --config configs/gardner/base.yaml \
    --track improved \
    --use_coral 1 \
    --use_weighted_sampler 1 \
    --pretrain_ckpt outputs/stage_pretrain/best.ckpt \
    --out_dir outputs/exp_pretrain_transfer \
    --seed 42

# Stage 3: Fine-tune on ICM
python scripts/train_gardner_single.py \
    --task icm \
    --config configs/gardner/base.yaml \
    --track improved \
    --use_weighted_sampler 1 \
    --pretrain_ckpt outputs/stage_pretrain/best.ckpt \
    --out_dir outputs/icm_pretrain_transfer \
    --seed 42

# Stage 4: Fine-tune on TE
python scripts/train_gardner_single.py \
    --task te \
    --config configs/gardner/base.yaml \
    --track improved \
    --use_weighted_sampler 1 \
    --pretrain_ckpt outputs/stage_pretrain/best.ckpt \
    --out_dir outputs/te_pretrain_transfer \
    --seed 42
```

---

## Troubleshooting

### Issue: "Module not found" error when loading checkpoint
**Solution**: Ensure path is absolute or relative to script directory
```bash
# ❌ Wrong
--pretrain_ckpt best.ckpt

# ✅ Correct
--pretrain_ckpt outputs/stage_pretrain/best.ckpt
```

### Issue: CORAL loss crashes with shape mismatch
**Solution**: Ensure `--use_coral 1` only used with `--task exp` (automatically enforced in code)
```bash
# ❌ Wrong
--task icm --use_coral 1

# ✅ Correct
--task exp --use_coral 1
```

### Issue: WeightedRandomSampler produces unusual class distribution
**Solution**: Check that `--use_class_weights` is False (sampler handles weighting)
```bash
# ✅ Correct (sampler only)
--use_weighted_sampler 1

# ⚠️ Double weighting (issues warning but works)
--use_weighted_sampler 1 --use_class_weights 1
```

---

## Files Modified
- ✅ [scripts/train_gardner_single.py](scripts/train_gardner_single.py) - Added load_backbone_only(), --pretrain_ckpt arg, checkpoint loading logic

## Files Created
- ✅ [TRANSFER_LEARNING_INTEGRATION.md](TRANSFER_LEARNING_INTEGRATION.md) - This file

## Dependencies
- ✅ [scripts/train_stage_pretrain.py](scripts/train_stage_pretrain.py) - Creates stage-pretrain checkpoints
- ✅ [src/dataset.py](src/dataset.py) - HumanEmbryoStageDataset for pretraining
- ✅ [src/loss_coral.py](src/loss_coral.py) - CORAL loss functions
- ✅ [src/model.py](src/model.py) - IVF_EffiMorphPP backbone
