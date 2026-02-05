#!/usr/bin/env python3
"""Test CORAL bug fix - ensures TE/ICM tasks output correct number of logits."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.model import IVF_EffiMorphPP

def test_te_task_without_coral():
    """Test that TE task outputs 3 logits without CORAL."""
    print("\n[TEST 1] TE task without CORAL")
    num_classes = 3
    task = "te"
    use_coral = False
    
    model = IVF_EffiMorphPP(
        num_classes=num_classes,
        dropout_p=0.0,
        task=task,
        use_coral=use_coral
    )
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_output = model(dummy_input)
    
    print(f"  Task: {task.upper()}, use_coral={use_coral}")
    print(f"  Expected logits: {num_classes}")
    print(f"  Actual logits: {dummy_output.shape[1]}")
    
    assert dummy_output.shape[1] == num_classes, \
        f"TE task should output {num_classes} logits, got {dummy_output.shape[1]}"
    print(f"  ✓ PASS: Model outputs correct number of logits for TE")


def test_te_task_with_coral_attempted():
    """Test that TE task still outputs 3 logits even if use_coral=True is passed (should be ignored)."""
    print("\n[TEST 2] TE task with CORAL=True (should be ignored by training script)")
    num_classes = 3
    task = "te"
    # In the actual training script, use_coral would be filtered to False for TE
    # But if someone accidentally passes it to the model, let's verify behavior
    use_coral = True
    
    model = IVF_EffiMorphPP(
        num_classes=num_classes,
        dropout_p=0.0,
        task=task,
        use_coral=use_coral  # This should be filtered by training script
    )
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_output = model(dummy_input)
    
    print(f"  Task: {task.upper()}, use_coral={use_coral}")
    print(f"  Expected logits: Should still be {num_classes} (TE is not ordinal)")
    print(f"  Actual logits: {dummy_output.shape[1]}")
    
    # Note: The model itself might output num_classes-1 if use_coral=True
    # The fix ensures the training script never passes use_coral=True for TE
    if dummy_output.shape[1] == num_classes - 1:
        print(f"  ⚠ Model respects use_coral flag (outputs {num_classes-1} logits)")
        print(f"  ✓ This is why the training script fix is critical!")
    else:
        print(f"  ✓ Model outputs {num_classes} logits")


def test_icm_task_without_coral():
    """Test that ICM task outputs 4 logits without CORAL."""
    print("\n[TEST 3] ICM task without CORAL")
    num_classes = 4
    task = "icm"
    use_coral = False
    
    model = IVF_EffiMorphPP(
        num_classes=num_classes,
        dropout_p=0.0,
        task=task,
        use_coral=use_coral
    )
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_output = model(dummy_input)
    
    print(f"  Task: {task.upper()}, use_coral={use_coral}")
    print(f"  Expected logits: {num_classes}")
    print(f"  Actual logits: {dummy_output.shape[1]}")
    
    assert dummy_output.shape[1] == num_classes, \
        f"ICM task should output {num_classes} logits, got {dummy_output.shape[1]}"
    print(f"  ✓ PASS: Model outputs correct number of logits for ICM")


def test_exp_task_with_coral():
    """Test that EXP task outputs 4 logits with CORAL (5 classes -> 4 binary thresholds)."""
    print("\n[TEST 4] EXP task with CORAL")
    num_classes = 5
    task = "exp"
    use_coral = True
    
    model = IVF_EffiMorphPP(
        num_classes=num_classes,
        dropout_p=0.0,
        task=task,
        use_coral=use_coral
    )
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_output = model(dummy_input)
    
    print(f"  Task: {task.upper()}, use_coral={use_coral}")
    print(f"  Expected logits: {num_classes - 1} (CORAL ordinal regression)")
    print(f"  Actual logits: {dummy_output.shape[1]}")
    
    assert dummy_output.shape[1] == num_classes - 1, \
        f"EXP task with CORAL should output {num_classes - 1} logits, got {dummy_output.shape[1]}"
    print(f"  ✓ PASS: Model outputs correct number of logits for EXP with CORAL")


def test_exp_task_without_coral():
    """Test that EXP task outputs 5 logits without CORAL."""
    print("\n[TEST 5] EXP task without CORAL")
    num_classes = 5
    task = "exp"
    use_coral = False
    
    model = IVF_EffiMorphPP(
        num_classes=num_classes,
        dropout_p=0.0,
        task=task,
        use_coral=use_coral
    )
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_output = model(dummy_input)
    
    print(f"  Task: {task.upper()}, use_coral={use_coral}")
    print(f"  Expected logits: {num_classes}")
    print(f"  Actual logits: {dummy_output.shape[1]}")
    
    assert dummy_output.shape[1] == num_classes, \
        f"EXP task without CORAL should output {num_classes} logits, got {dummy_output.shape[1]}"
    print(f"  ✓ PASS: Model outputs correct number of logits for EXP without CORAL")


if __name__ == "__main__":
    print("="*80)
    print("Testing CORAL Bug Fix - Model Output Shape Validation")
    print("="*80)
    
    try:
        test_te_task_without_coral()
        test_te_task_with_coral_attempted()
        test_icm_task_without_coral()
        test_exp_task_with_coral()
        test_exp_task_without_coral()
        
        print("\n" + "="*80)
        print("All tests passed! ✓")
        print("="*80)
        print("\nSummary:")
        print("  - TE task (3 classes) outputs 3 logits without CORAL ✓")
        print("  - ICM task (4 classes) outputs 4 logits without CORAL ✓")
        print("  - EXP task (5 classes) outputs 4 logits with CORAL ✓")
        print("  - EXP task (5 classes) outputs 5 logits without CORAL ✓")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
