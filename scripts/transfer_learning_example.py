#!/usr/bin/env python3
"""
Example: Loading Pretrained Stage Backbone for Downstream Tasks

This script demonstrates how to:
1. Load the pretrained stage backbone
2. Adapt it for downstream tasks (EXP, ICM, TE)
3. Fine-tune on downstream data
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import IVF_EffiMorphPP


def load_pretrained_backbone(pretrain_checkpoint_path: str, device: torch.device):
    """Load pretrained stage backbone."""
    checkpoint = torch.load(pretrain_checkpoint_path, map_location=device)
    pretrain_state = checkpoint["model_state_dict"]
    
    print(f"✓ Loaded pretrain checkpoint from: {pretrain_checkpoint_path}")
    print(f"  Best epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    print(f"  Val macro-F1: {checkpoint.get('val_macro_f1', 'N/A'):.4f}")
    
    return pretrain_state


def create_downstream_model(
    num_classes: int,
    task: str,
    pretrain_state: dict,
    use_coral: bool = False,
    device: torch.device = None,
):
    """
    Create a downstream model and initialize with pretrained backbone weights.
    
    Args:
        num_classes: Number of classes for downstream task
        task: 'exp', 'icm', or 'te'
        pretrain_state: State dict from pretrained stage model
        use_coral: If True (for EXP), use CORAL ordinal regression
        device: Device to move model to
    
    Returns:
        Initialized model ready for fine-tuning
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create downstream model
    model = IVF_EffiMorphPP(
        num_classes=num_classes,
        task=task,
        use_coral=use_coral,
    )
    
    # Load pretrained backbone weights (strict=False to allow head mismatch)
    incompatible = model.load_state_dict(pretrain_state, strict=False)
    
    print(f"\n✓ Created downstream model for task '{task}' with {num_classes} classes")
    print(f"  Use CORAL: {use_coral}")
    
    if incompatible.missing_keys:
        print(f"  Missing keys (head): {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"  Unexpected keys: {incompatible.unexpected_keys}")
    
    model.to(device)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    return model


def fine_tune_example(model: torch.nn.Module, task: str):
    """
    Example: How to fine-tune the pretrained model on downstream task.
    
    This is a pseudo-code example. Use your existing training script:
    scripts/train_gardner_single.py
    """
    
    print(f"\n{'='*80}")
    print(f"Fine-tuning on {task.upper()} task")
    print(f"{'='*80}")
    
    # 1. Prepare your downstream data
    # ds_train = GardnerDataset(csv_path, img_dir, task=task, split="train", augment=True)
    # ds_val = GardnerDataset(csv_path, img_dir, task=task, split="val", augment=False)
    
    # 2. Create dataloader
    # dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
    
    # 3. Setup optimizer (lower LR for fine-tuning)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 4. Setup loss
    if task == "exp":
        # For EXP: use CORAL loss (if use_coral=True) or CrossEntropyLoss
        loss_fn = torch.nn.CrossEntropyLoss()
        # OR: loss_fn = coral_loss (from src.loss_coral)
    else:
        # For ICM/TE: use CrossEntropyLoss or FocalLoss
        loss_fn = torch.nn.CrossEntropyLoss()
    
    # 5. Training loop (pseudo-code)
    # for epoch in range(num_epochs):
    #     model.train()
    #     for batch in dl_train:
    #         x, y = batch
    #         logits = model(x)
    #         loss = loss_fn(logits, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     
    #     # Validate and save best checkpoint
    #     val_metrics = validate(model, dl_val)
    #     ...
    
    print(f"✓ Model ready for fine-tuning")
    print(f"✓ Optimizer: AdamW(lr=1e-4)")
    print(f"✓ Loss: {loss_fn.__class__.__name__}")
    print(f"✓ Next: Use scripts/train_gardner_single.py with this initialized model")


# =============================================
# Main Example
# =============================================

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("Stage Pretraining → Downstream Transfer Learning Example")
    print(f"{'='*80}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Path to pretrained checkpoint
    pretrain_ckpt = "outputs/pretrain_stage/best.ckpt"
    
    # Load pretrained weights
    pretrain_state = load_pretrained_backbone(pretrain_ckpt, device)
    
    # -----------------------------------------------
    # Example 1: Transfer to EXP task (5 classes, CORAL)
    # -----------------------------------------------
    model_exp = create_downstream_model(
        num_classes=5,
        task="exp",
        pretrain_state=pretrain_state,
        use_coral=True,
        device=device,
    )
    fine_tune_example(model_exp, "exp")
    
    # -----------------------------------------------
    # Example 2: Transfer to ICM task (3 classes, CE)
    # -----------------------------------------------
    print("\n")
    model_icm = create_downstream_model(
        num_classes=3,
        task="icm",
        pretrain_state=pretrain_state,
        use_coral=False,
        device=device,
    )
    fine_tune_example(model_icm, "icm")
    
    # -----------------------------------------------
    # Example 3: Transfer to TE task (3 classes, CE)
    # -----------------------------------------------
    print("\n")
    model_te = create_downstream_model(
        num_classes=3,
        task="te",
        pretrain_state=pretrain_state,
        use_coral=False,
        device=device,
    )
    fine_tune_example(model_te, "te")
    
    print(f"\n{'='*80}")
    print("✓ All models ready for fine-tuning on downstream tasks!")
    print(f"{'='*80}\n")
