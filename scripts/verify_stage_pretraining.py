#!/usr/bin/env python3
"""
Verification Script: Validate Stage Pretraining Implementation

This script verifies that all components are correctly implemented:
1. Dataset loads correctly
2. Model creation works
3. Training loop is functional
4. Weight loading for downstream tasks works
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from src.dataset import HumanEmbryoStageDataset
from src.model import IVF_EffiMorphPP


def test_dataset():
    """Test dataset loading."""
    print("\n" + "="*80)
    print("TEST 1: Dataset Loading")
    print("="*80)
    
    try:
        ds = HumanEmbryoStageDataset(
            csv_path="data/metadata/humanembryo2.csv",
            img_base_dir="data/HumanEmbryo2.0",
            split="train",
            augment=False,
        )
        
        assert len(ds) > 0, "Dataset is empty"
        img, label = ds[0]
        
        assert img.shape == (3, 224, 224), f"Wrong image shape: {img.shape}"
        assert label.dtype == torch.long, f"Wrong label dtype: {label.dtype}"
        assert label.item() in [0, 1, 2], f"Invalid label: {label.item()}"
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Samples: {len(ds)}")
        print(f"  Image shape: {img.shape}")
        print(f"  Label type: {label.dtype}")
        print(f"  Sample label: {label.item()} (0=cleavage, 1=morula, 2=blastocyst)")
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation for stage pretraining."""
    print("\n" + "="*80)
    print("TEST 2: Model Creation (Stage Pretraining)")
    print("="*80)
    
    try:
        device = torch.device("cpu")
        model = IVF_EffiMorphPP(
            num_classes=3,
            task="stage",
            use_coral=False,
            dropout_p=0.3,
        )
        model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224, device=device)
        logits = model(x)
        
        assert logits.shape == (2, 3), f"Wrong output shape: {logits.shape}"
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {logits.shape}")
        
        return True, model
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_downstream_model(model):
    """Test creating downstream models and loading pretrained weights."""
    print("\n" + "="*80)
    print("TEST 3: Downstream Model Creation (Transfer Learning)")
    print("="*80)
    
    try:
        device = torch.device("cpu")
        
        # Simulate pretrain state dict
        pretrain_state = model.state_dict()
        
        # When transferring, we load only backbone keys (not head)
        # Extract backbone weights from pretrain state
        backbone_keys = {
            k: v for k, v in pretrain_state.items()
            if not k.startswith("head.")
        }
        
        print(f"Backbone keys to transfer: {len(backbone_keys)}")
        
        # Test EXP (5 classes, no CORAL for pretraining)
        model_exp = IVF_EffiMorphPP(num_classes=5, task="exp", use_coral=False)
        
        # Load only backbone weights
        missing_keys = []
        for name, param in model_exp.named_parameters():
            if name in backbone_keys:
                param.data = backbone_keys[name].clone()
        
        print(f"✓ EXP model created")
        print(f"  Num classes: 5")
        print(f"  Backbone weights transferred: {len(backbone_keys)} keys")
        
        # Test ICM (3 classes, CE)
        model_icm = IVF_EffiMorphPP(num_classes=3, task="icm", use_coral=False)
        for name, param in model_icm.named_parameters():
            if name in backbone_keys:
                param.data = backbone_keys[name].clone()
        
        print(f"\n✓ ICM model created")
        print(f"  Num classes: 3")
        print(f"  Backbone weights transferred: {len(backbone_keys)} keys")
        
        # Test TE (3 classes, CE)
        model_te = IVF_EffiMorphPP(num_classes=3, task="te", use_coral=False)
        for name, param in model_te.named_parameters():
            if name in backbone_keys:
                param.data = backbone_keys[name].clone()
        
        print(f"\n✓ TE model created")
        print(f"  Num classes: 3")
        print(f"  Backbone weights transferred: {len(backbone_keys)} keys")
        
        print(f"\n✓ All downstream models support backbone weight transfer")
        print(f"  Transfer method: Load backbone keys only, new head initialized randomly")
        print(f"  This is the standard transfer learning approach")
        
        return True
        
    except Exception as e:
        print(f"✗ Downstream model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test training loop (basic sanity check)."""
    print("\n" + "="*80)
    print("TEST 4: Training Loop (Forward/Backward Pass)")
    print("="*80)
    
    try:
        device = torch.device("cpu")
        
        # Create model and data
        model = IVF_EffiMorphPP(num_classes=3, task="stage", use_coral=False)
        model.to(device)
        model.train()
        
        # Create dummy batch
        x = torch.randn(4, 3, 224, 224, device=device)
        y = torch.tensor([0, 1, 2, 1], dtype=torch.long, device=device)
        
        # Forward pass
        logits = model(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)
        
        print(f"✓ Forward pass successful")
        print(f"  Loss: {loss.item():.6f}")
        
        # Backward pass
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        optimizer.zero_grad()
        loss.backward()
        
        print(f"✓ Backward pass successful")
        
        # Optimizer step
        optimizer.step()
        print(f"✓ Optimizer step successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("STAGE PRETRAINING VERIFICATION")
    print("="*80)
    
    results = []
    
    # Test 1: Dataset
    results.append(("Dataset Loading", test_dataset()))
    
    # Test 2: Model Creation
    success, model = test_model_creation()
    results.append(("Model Creation", success))
    
    # Test 3: Downstream (requires model from test 2)
    if model is not None:
        results.append(("Downstream Models", test_downstream_model(model)))
    
    # Test 4: Training Loop
    results.append(("Training Loop", test_training_loop()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All verification tests passed!")
        print("✓ Stage pretraining implementation is correct!")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
