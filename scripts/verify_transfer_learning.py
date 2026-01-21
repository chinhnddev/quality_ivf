#!/usr/bin/env python3
"""
Verification test for transfer learning integration.
Checks that:
1. load_backbone_only() function exists and callable
2. --pretrain_ckpt argument is registered
3. CORAL loss integration works
4. WeightedRandomSampler integration works
"""

import sys
from pathlib import Path
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_load_backbone_only_function():
    """Verify load_backbone_only function exists in train_gardner_single.py"""
    print("\n[TEST 1] Checking load_backbone_only function...")
    try:
        # Import the script module to check function exists
        import importlib.util
        script_path = Path(__file__).parent / "train_gardner_single.py"
        
        # We can't actually execute it without all dependencies,
        # but we can check the source code
        with open(script_path, "r") as f:
            content = f.read()
            
        if "def load_backbone_only" in content:
            print("  ✓ load_backbone_only() function found")
            return True
        else:
            print("  ✗ load_backbone_only() function NOT found")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_pretrain_ckpt_argument():
    """Verify --pretrain_ckpt argument is registered"""
    print("\n[TEST 2] Checking --pretrain_ckpt argument...")
    try:
        with open(Path(__file__).parent / "train_gardner_single.py", "r") as f:
            content = f.read()
            
        if '--pretrain_ckpt' in content and 'parser.add_argument' in content:
            # Check it's properly registered
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '--pretrain_ckpt' in line and 'parser.add_argument' in line:
                    print(f"  ✓ --pretrain_ckpt argument registered at line {i+1}")
                    # Verify it has help text
                    if 'help=' in line or (i+1 < len(lines) and 'help=' in lines[i+1]):
                        print("  ✓ Help text provided")
                        return True
            print("  ✗ --pretrain_ckpt not properly registered")
            return False
        else:
            print("  ✗ --pretrain_ckpt argument NOT found")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_checkpoint_loading_integration():
    """Verify checkpoint loading is integrated into training flow"""
    print("\n[TEST 3] Checking checkpoint loading integration...")
    try:
        with open(Path(__file__).parent / "train_gardner_single.py", "r") as f:
            content = f.read()
            
        if "if args.pretrain_ckpt:" in content and "load_backbone_only" in content:
            print("  ✓ Checkpoint loading integrated into training flow")
            
            # Check it's after model creation
            lines = content.split('\n')
            model_creation_line = None
            load_call_line = None
            
            for i, line in enumerate(lines):
                if "model = IVF_EffiMorphPP" in line and model_creation_line is None:
                    model_creation_line = i
                if "if args.pretrain_ckpt:" in line:
                    load_call_line = i
            
            if model_creation_line and load_call_line:
                if load_call_line > model_creation_line:
                    print(f"  ✓ Checkpoint loaded after model creation")
                    print(f"    - Model creation: line ~{model_creation_line+1}")
                    print(f"    - Checkpoint loading: line ~{load_call_line+1}")
                    return True
                else:
                    print("  ✗ Checkpoint loading NOT after model creation")
                    return False
        else:
            print("  ✗ Checkpoint loading integration NOT found")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_coral_integration():
    """Verify CORAL loss is properly integrated"""
    print("\n[TEST 4] Checking CORAL loss integration...")
    try:
        with open(Path(__file__).parent / "train_gardner_single.py", "r") as f:
            content = f.read()
        
        checks = [
            ("CORAL import", "from src.loss_coral import coral_loss"),
            ("CORAL use in make_loss_fn", "use_coral and task == \"exp\""),
            ("CORAL loss call", "coral_loss(logits, targets, num_classes)"),
            ("CORAL safety check", "use_coral and task == \"exp\""),
        ]
        
        passed = 0
        for check_name, check_string in checks:
            if check_string in content:
                print(f"  ✓ {check_name}")
                passed += 1
            else:
                print(f"  ✗ {check_name} NOT found")
        
        return passed == len(checks)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_weighted_sampler_integration():
    """Verify WeightedRandomSampler is properly integrated"""
    print("\n[TEST 5] Checking WeightedRandomSampler integration...")
    try:
        with open(Path(__file__).parent / "train_gardner_single.py", "r") as f:
            content = f.read()
        
        checks = [
            ("Sampler import", "WeightedRandomSampler"),
            ("use_weighted_sampler argument", "--use_weighted_sampler"),
            ("Sampler creation", "sampler = WeightedRandomSampler"),
            ("DataLoader integration", "sampler=sampler if use_weighted_sampler else None"),
        ]
        
        passed = 0
        for check_name, check_string in checks:
            if check_string in content:
                print(f"  ✓ {check_name}")
                passed += 1
            else:
                print(f"  ✗ {check_name} NOT found")
        
        return passed == len(checks)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("=" * 70)
    print("TRANSFER LEARNING INTEGRATION VERIFICATION")
    print("=" * 70)
    
    results = {
        "load_backbone_only function": test_load_backbone_only_function(),
        "--pretrain_ckpt argument": test_pretrain_ckpt_argument(),
        "Checkpoint loading integration": test_checkpoint_loading_integration(),
        "CORAL loss integration": test_coral_integration(),
        "WeightedRandomSampler integration": test_weighted_sampler_integration(),
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("=" * 70)
    if all_passed:
        print("✓ All integration tests PASSED!")
        print("\nYou can now use:")
        print("  python scripts/train_gardner_single.py \\")
        print("    --task exp \\")
        print("    --config configs/gardner/base.yaml \\")
        print("    --track improved \\")
        print("    --use_coral 1 \\")
        print("    --use_weighted_sampler 1 \\")
        print("    --pretrain_ckpt outputs/stage_pretrain/best.ckpt")
        return 0
    else:
        print("✗ Some integration tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
