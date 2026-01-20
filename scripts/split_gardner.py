#!/usr/bin/env python3
"""
Dataset split script for Gardner blastocyst dataset.

Follows the split protocol from the paper:
"An annotated human blastocyst dataset to benchmark deep learning architectures for IVF"

Creates train/val/test splits from silver and gold annotations.
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings


def load_csv(filepath, delimiter=';'):
    """Load CSV with specified delimiter, handling trailing delimiters."""
    df = pd.read_csv(filepath, delimiter=delimiter, engine='python')
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    return df


def validate_datasets(gold_df, silver_df):
    """Validate dataset assumptions."""
    gold_count = len(gold_df)
    silver_count = len(silver_df)

    assert gold_count == 300, f"Gold test set should have 300 images, got {gold_count}"
    assert silver_count == 2044, f"Silver train set should have 2044 images, got {silver_count}"

    gold_images = set(gold_df['Image'])
    silver_images = set(silver_df['Image'])

    intersection = gold_images & silver_images
    if intersection:
        raise ValueError(f"Overlap between gold and silver images: {intersection}")

    print(f"✓ Gold test set: {gold_count} images")
    print(f"✓ Silver train set: {silver_count} images")
    print(f"✓ No overlap between gold and silver images")


def process_gold_labels(df):
    """Process gold labels: EXP 0-4, ICM/TE 0/1/2 or ND/NA/empty."""
    # EXP_gold is already 0-4 based on examples
    df = df.rename(columns={'EXP_gold': 'EXP', 'ICM_gold': 'ICM', 'TE_gold': 'TE'})

    # Convert EXP to int if numeric, else keep as string
    df['EXP'] = pd.to_numeric(df['EXP'], errors='coerce').fillna(df['EXP'])
    df['EXP'] = df['EXP'].apply(lambda x: int(x) if pd.notna(x) and isinstance(x, (int, float)) else x)

    # Map ICM/TE: A/B/C -> 0/1/2, keep ND/NA/empty as strings
    icm_te_mapping = {'A': 0, 'B': 1, 'C': 2}
    for col in ['ICM', 'TE']:
        df[col] = df[col].map(icm_te_mapping).fillna(df[col])
        # Convert to numeric, keep strings for ND/NA/empty
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col])
        # Convert to int if numeric, else keep as string
        df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) and isinstance(x, (int, float)) else x)

    return df[['Image', 'EXP', 'ICM', 'TE']]


def process_silver_labels(df):
    """Process silver labels: EXP 0-4, ICM/TE 0/1/2 or ND."""
    df = df.rename(columns={'EXP_silver': 'EXP', 'ICM_silver': 'ICM', 'TE_silver': 'TE'})

    # ICM/TE: 0/1/2 are A/B/C, 3 is ND
    for col in ['ICM', 'TE']:
        df[col] = df[col].replace(3, 'ND')

    return df[['Image', 'EXP', 'ICM', 'TE']]


def create_stratified_split(df, val_ratio=0.2, seed=42):
    """Create stratified train/val split by EXP."""
    df = df.reset_index(drop=True)  # Ensure unique indices
    try:
        train_df, val_df = train_test_split(
            df, test_size=val_ratio, stratify=df['EXP'],
            random_state=seed
        )
        print("✓ Stratified split by EXP successful")
    except ValueError as e:
        warnings.warn(f"Stratification failed: {e}. Falling back to random split.")
        train_df, val_df = train_test_split(
            df, test_size=val_ratio, random_state=seed
        )

    return train_df, val_df


def print_summary_report(train_df, val_df, test_df):
    """Print validation summary report."""
    print("\n" + "="*50)
    print("DATASET SPLIT SUMMARY")
    print("="*50)

    # Counts per EXP class
    print("\nEXP class distribution:")
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = df['EXP'].value_counts().sort_index()
        print(f"  {split_name}: {dict(counts)}")

    # ICM/TE valid vs ND/NA
    for col in ['ICM', 'TE']:
        print(f"\n{col} distribution:")
        for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            total = len(df)
            valid = df[col].notna() & (df[col] != 'ND') & (df[col] != 'NA') & (df[col] != '')
            valid_count = valid.sum()
            nd_count = (df[col] == 'ND').sum()
            na_count = (df[col] == 'NA').sum()
            empty_count = (df[col] == '').sum()
            print(f"  {split_name}: Valid={valid_count}, ND={nd_count}, NA={na_count}, Empty={empty_count}")

    print(f"\nTotal images: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


def save_csv(df, filepath, delimiter=','):
    """Save DataFrame to CSV with clean label formatting (no 2.0)."""

    def norm_cell(x):
        # Keep empty
        if pd.isna(x):
            return ""
        # Normalize strings
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return ""
            return s
        # Normalize numbers: 2.0 -> "2"
        if isinstance(x, (int,)):
            return str(x)
        if isinstance(x, (float,)):
            # If it is an integer float, remove .0
            if float(x).is_integer():
                return str(int(x))
            return str(x)
        return str(x)

    out = df.copy()
    for col in ["EXP", "ICM", "TE"]:
        if col in out.columns:
            out[col] = out[col].apply(norm_cell)

    out.to_csv(filepath, index=False, sep=delimiter)
    print(f"✓ Saved {filepath}")



def main():
    parser = argparse.ArgumentParser(description='Split Gardner blastocyst dataset')
    parser.add_argument('--gold_csv', required=True, help='Path to gold test CSV')
    parser.add_argument('--silver_csv', required=True, help='Path to silver train CSV')
    parser.add_argument('--out_dir', default='splits', help='Output directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--delimiter_in', default=';', help='Input CSV delimiter')
    parser.add_argument('--delimiter_out', default=',', help='Output CSV delimiter')

    args = parser.parse_args()

    # Load datasets
    gold_df = load_csv(args.gold_csv, args.delimiter_in)
    silver_df = load_csv(args.silver_csv, args.delimiter_in)

    # Validate
    validate_datasets(gold_df, silver_df)

    # Process labels
    test_df = process_gold_labels(gold_df)
    silver_processed = process_silver_labels(silver_df)

    # Ensure unique images in silver set
    silver_processed = silver_processed.drop_duplicates(subset='Image')

    # Create train/val split
    train_df, val_df = create_stratified_split(silver_processed, args.val_ratio, args.seed)

    # Validate splits
    train_images = set(train_df['Image'])
    val_images = set(val_df['Image'])
    silver_images = set(silver_processed['Image'])

    assert train_images | val_images == silver_images, "Train + Val != Silver"
    assert train_images & val_images == set(), "Train and Val overlap"

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Save files
    save_csv(test_df, os.path.join(args.out_dir, 'test.csv'), args.delimiter_out)
    save_csv(train_df, os.path.join(args.out_dir, 'train.csv'), args.delimiter_out)
    save_csv(val_df, os.path.join(args.out_dir, 'val.csv'), args.delimiter_out)

    # Print summary
    print_summary_report(train_df, val_df, test_df)


if __name__ == '__main__':
    main()
