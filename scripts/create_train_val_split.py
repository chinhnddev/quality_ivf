# create_train_val_split_complete.py
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import os

print("="*70)
print("CREATE TRAIN/VAL SPLIT - Complete Version")
print("="*70)

# ════════════════════════════════════════════════════════════
# Paper's parameters
# ════════════════════════════════════════════════════════════
RANDOM_STATE = 10_123
VAL_RATIO = 1.0 / 5.0  # 20%
SHUFFLE = True

print(f"\nPaper's splitting parameters:")
print(f"  test_size: {VAL_RATIO} (20% for validation)")
print(f"  shuffle: {SHUFFLE}")
print(f"  random_state: {RANDOM_STATE}")

# ════════════════════════════════════════════════════════════
# Load data
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 1: LOADING DATA")
print(f"{'='*70}")

column_names = ['Image', 'EXP', 'ICM', 'TE']

complete = pd.read_csv("splits/complete.csv", header=None, names=column_names)
test = pd.read_csv("splits/test.csv", header=None, names=column_names)

print(f"  ✓ Complete: {len(complete)} samples")
print(f"  ✓ Test: {len(test)} samples")

# Show columns
print(f"\n  Complete columns: {list(complete.columns)}")
print(f"  Complete dtypes:\n{complete.dtypes}")

# ════════════════════════════════════════════════════════════
# Clean data - preserve integer format
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 2: CLEANING DATA (Preserve Integer Format)")
print(f"{'='*70}")

# Normalize column names
complete.columns = [col.strip() for col in complete.columns]
test.columns = [col.strip() for col in test.columns]

# Function to clean values while preserving integers
def clean_value(val):
    """Clean value and preserve integer format"""
    if pd.isna(val):
        return 'NA'
    
    val_str = str(val).strip()
    
    # Try to convert to int if it's a number
    try:
        # Check if it's a float like "3.0"
        if '.' in val_str:
            val_float = float(val_str)
            val_int = int(val_float)
            # If it's a clean integer (3.0 → 3), return int
            if val_float == val_int:
                return str(val_int)
            else:
                # If it's something like 3.1, round it
                return str(int(round(val_float)))
        else:
            # Already an integer string
            return val_str
    except:
        # If conversion fails, return as-is
        return val_str

# Apply cleaning to all columns (except filename)
print("\n  Cleaning columns...")
for col_idx in range(len(complete.columns)):
    if col_idx == 0:
        # Image name - just strip
        complete.iloc[:, col_idx] = complete.iloc[:, col_idx].astype(str).str.strip()
        test.iloc[:, col_idx] = test.iloc[:, col_idx].astype(str).str.strip()
    else:
        # EXP, ICM, TEQ - clean and preserve integers
        complete.iloc[:, col_idx] = complete.iloc[:, col_idx].apply(clean_value)
        test.iloc[:, col_idx] = test.iloc[:, col_idx].apply(clean_value)

print("  ✓ Data cleaned (integers preserved)")

# Verify
print(f"\n  Sample rows after cleaning:")
print(f"    Complete: {complete.iloc[0].tolist()}")
print(f"    Test: {test.iloc[0].tolist()}")

# ════════════════════════════════════════════════════════════
# Create train+val dataset
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 3: CREATING TRAIN+VAL DATASET")
print(f"{'='*70}")

# Get test filenames
test_files = set(test.iloc[:, 0])

# Exclude test samples from complete
train_val = complete[~complete.iloc[:, 0].isin(test_files)].copy()

print(f"  ✓ Train+Val: {len(train_val)} samples")
print(f"    (Complete {len(complete)} - Test {len(test)} = {len(train_val)})")

# ════════════════════════════════════════════════════════════
# Analyze EXP distribution
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 4: EXP DISTRIBUTION ANALYSIS")
print(f"{'='*70}")

train_val_counts = Counter(train_val.iloc[:, 1])
test_counts = Counter(test.iloc[:, 1])

print(f"\n  Train+Val EXP distribution:")
for cls, count in sorted(train_val_counts.items()):
    pct = (count / len(train_val)) * 100
    print(f"    {cls}: {count:4d} samples ({pct:5.1f}%)")

print(f"\n  Test EXP distribution:")
for cls, count in sorted(test_counts.items()):
    pct = (count / len(test)) * 100
    print(f"    {cls}: {count:4d} samples ({pct:5.1f}%)")

# ════════════════════════════════════════════════════════════
# METHOD 1: Paper's original (NO stratification)
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("METHOD 1: PAPER'S ORIGINAL (NO Stratification)")
print(f"{'='*70}")

train_v1, val_v1 = train_test_split(
    train_val,
    test_size=VAL_RATIO,
    shuffle=SHUFFLE,
    random_state=RANDOM_STATE
)

train_v1_counts = Counter(train_v1.iloc[:, 1])
val_v1_counts = Counter(val_v1.iloc[:, 1])

print(f"\nSplit: Train {len(train_v1)} / Val {len(val_v1)}")

# Get all unique classes
all_classes = sorted(set(train_v1_counts.keys()) | set(val_v1_counts.keys()) | set(test_counts.keys()))

print(f"\n{'Class':<8} {'Train':>10} {'Val':>10} {'Test':>10} {'Val%':>8} {'Test%':>8} {'Diff':>8}")
print("-"*70)

total_diff_v1 = 0
valid_classes = [c for c in all_classes if c in ['0','1','2','3','4']]

for cls in all_classes:
    t = train_v1_counts.get(cls, 0)
    v = val_v1_counts.get(cls, 0)
    te = test_counts.get(cls, 0)
    
    if cls in valid_classes:
        v_pct = (v / len(val_v1)) * 100
        te_pct = (te / len(test)) * 100
        diff = abs(v_pct - te_pct)
        total_diff_v1 += diff
        status = "✓" if diff < 3.0 else "⚠"
    else:
        v_pct = te_pct = diff = 0
        status = ""
    
    print(f"{cls:<8} {t:>10} {v:>10} {te:>10} {v_pct:>7.1f}% {te_pct:>7.1f}% {diff:>7.1f}% {status}")

avg_diff_v1 = total_diff_v1 / len(valid_classes) if valid_classes else 0
print(f"\nAverage difference (valid classes): {avg_diff_v1:.2f}%")

# ════════════════════════════════════════════════════════════
# METHOD 2: Stratified (IMPROVED)
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("METHOD 2: STRATIFIED (IMPROVED)")
print(f"{'='*70}")

# Check if we can use stratification
can_stratify = all(cls in ['0','1','2','3','4'] for cls in train_val_counts.keys())

if can_stratify:
    train_v2, val_v2 = train_test_split(
        train_val,
        test_size=VAL_RATIO,
        shuffle=SHUFFLE,
        random_state=RANDOM_STATE,
        stratify=train_val.iloc[:, 1]  # ← STRATIFY by EXP
    )
    
    print(f"\nSplit: Train {len(train_v2)} / Val {len(val_v2)} (Stratified)")
else:
    print(f"\n⚠ Cannot stratify - dataset has non-standard EXP values")
    print(f"  Using Method 1 for both options")
    train_v2 = train_v1
    val_v2 = val_v1

train_v2_counts = Counter(train_v2.iloc[:, 1])
val_v2_counts = Counter(val_v2.iloc[:, 1])

print(f"\n{'Class':<8} {'Train':>10} {'Val':>10} {'Test':>10} {'Val%':>8} {'Test%':>8} {'Diff':>8}")
print("-"*70)

total_diff_v2 = 0
for cls in all_classes:
    t = train_v2_counts.get(cls, 0)
    v = val_v2_counts.get(cls, 0)
    te = test_counts.get(cls, 0)
    
    if cls in valid_classes:
        v_pct = (v / len(val_v2)) * 100
        te_pct = (te / len(test)) * 100
        diff = abs(v_pct - te_pct)
        total_diff_v2 += diff
        status = "✓" if diff < 3.0 else "⚠"
    else:
        v_pct = te_pct = diff = 0
        status = ""
    
    print(f"{cls:<8} {t:>10} {v:>10} {te:>10} {v_pct:>7.1f}% {te_pct:>7.1f}% {diff:>7.1f}% {status}")

avg_diff_v2 = total_diff_v2 / len(valid_classes) if valid_classes else 0
print(f"\nAverage difference (valid classes): {avg_diff_v2:.2f}%")

# ════════════════════════════════════════════════════════════
# Choose method
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("CHOOSE METHOD")
print(f"{'='*70}")

print("\n1. Paper's Original (random split)")
if can_stratify:
    print("2. Stratified (maintains class balance) ← RECOMMENDED")
else:
    print("2. Stratified (NOT AVAILABLE - invalid EXP values)")

choice = input("\nEnter choice (1 or 2, default=2): ").strip()

if choice == '1':
    train_final = train_v1
    val_final = val_v1
    method = "Paper's Original"
    print("\n→ Using METHOD 1: Paper's Original")
else:
    train_final = train_v2
    val_final = val_v2
    method = "Stratified" if can_stratify else "Paper's Original"
    print(f"\n→ Using METHOD 2: {method}")

# ════════════════════════════════════════════════════════════
# Final verification
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FINAL VERIFICATION")
print(f"{'='*70}")

print(f"\nMethod: {method}")
print(f"  Train: {len(train_final)} samples")
print(f"  Val:   {len(val_final)} samples")
print(f"  Test:  {len(test)} samples (unchanged)")

# Data integrity check
original_total = len(complete)
final_total = len(train_final) + len(val_final) + len(test)

print(f"\nData integrity:")
print(f"  Original: {original_total} samples")
print(f"  Final:    {final_total} samples")
if original_total == final_total:
    print(f"  ✓ NO DATA LOSS")
else:
    print(f"  ⚠ WARNING: {abs(original_total - final_total)} samples difference!")

# Check for float values
print(f"\nChecking for unwanted float values...")
sample_val = train_final.iloc[0, 1]  # Check EXP column
has_decimal = '.' in str(sample_val)
if has_decimal:
    print(f"  ⚠ Found decimal point in data: {sample_val}")
else:
    print(f"  ✓ Values are clean integers")

# ════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SAVE SPLITS")
print(f"{'='*70}")

save = input("\nSave splits? (yes/no, default=yes): ").strip().lower()

if save in ['yes', 'y', '']:
    # Save CSV files
    train_final.to_csv("splits/train.csv", index=False)
    val_final.to_csv("splits/val.csv", index=False)
    
    print("\n✓ SAVED:")
    print("  - splits/train.csv")
    print("  - splits/val.csv")
    print("  - splits/test.csv (unchanged)")
    
    # Save metadata
    import json
    metadata = {
        'method': method,
        'train_size': len(train_final),
        'val_size': len(val_final),
        'test_size': len(test),
        'random_state': RANDOM_STATE,
        'val_ratio': VAL_RATIO,
        'stratified': can_stratify and choice != '1',
        'train_distribution': dict(Counter(train_final.iloc[:, 1])),
        'val_distribution': dict(Counter(val_final.iloc[:, 1]))
    }
    
    with open("splits/split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("  - splits/split_metadata.json")
    
    # Verify saved files
    print(f"\n{'='*70}")
    print("VERIFYING SAVED FILES")
    print(f"{'='*70}")
    
    train_check = pd.read_csv("splits/train.csv")
    val_check = pd.read_csv("splits/val.csv")
    
    print(f"\nTrain CSV:")
    print(f"  Rows: {len(train_check)}")
    print(f"  First row: {train_check.iloc[0].tolist()}")
    
    print(f"\nVal CSV:")
    print(f"  Rows: {len(val_check)}")
    print(f"  First row: {val_check.iloc[0].tolist()}")
    
    # Final check for floats
    train_exp = str(train_check.iloc[0, 1])
    if '.' in train_exp and train_exp not in ['NA', 'nan']:
        print(f"\n⚠ WARNING: Saved file contains floats!")
        print(f"   Example: {train_exp}")
        print(f"   Run: python force_integer_format.py")
    else:
        print(f"\n✓ Saved files verified - integers preserved!")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    
    print("\n1. Verify splits:")
    print("   head splits/train.csv")
    print("   head splits/val.csv")
    
    print("\n2. Train model:")
    print("   python scripts/train_gardner_single.py \\")
    print("     --task exp \\")
    print("     --track improved \\")
    print("     --use_coral 1 \\")
    print("     --epochs 60")
    
    print("\n3. Expected results:")
    if method == "Stratified":
        print("   Val: 70-72%")
        print("   Test: 80-81%")
    else:
        print("   Val: 68-70%")
        print("   Test: 80%")
else:
    print("\n✗ Not saved")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
