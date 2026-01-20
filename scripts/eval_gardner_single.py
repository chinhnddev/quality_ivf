#!/usr/bin/env python3
import argparse
import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from src.dataset import GardnerDataset
from src.model import IVF_EffiMorphPP


def main():
    parser = argparse.ArgumentParser(description='Evaluate IVF-EffiMorphPP on Gardner test set')
    parser.add_argument('--task', type=str, required=True, choices=['exp', 'icm', 'te'], help='Task to evaluate: exp, icm, or te')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--splits_dir', type=str, required=True, help='Directory containing test.csv')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for metrics and preds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    img_dir = 'blastocyst_Dataset/Images'
    test_csv = os.path.join(args.splits_dir, 'test.csv')

    # Load raw test data for masking
    test_df = pd.read_csv(test_csv)
    test_df['EXP'] = test_df['EXP'].astype(str).str.strip()
    test_df['ICM'] = test_df['ICM'].astype(str).str.strip()
    test_df['TE'] = test_df['TE'].astype(str).str.strip()
    test_df['EXP'] = pd.to_numeric(test_df['EXP'], errors='coerce')

    # Dataset (loads all samples)
    test_dataset = GardnerDataset(test_csv, img_dir, task=args.task, split='test')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Model
    num_classes = 5 if args.task == 'exp' else 3
    model = IVF_EffiMorphPP(num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Predictions
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Prepare preds.csv
    preds_df = test_df[['Image']].copy()
    preds_df['y_true'] = test_df[args.task.upper()]
    preds_df['y_pred'] = all_preds
    for i in range(num_classes):
        preds_df[f'prob_{i}'] = [p[i] for p in all_probs]

    # Masking for metrics
    if args.task == 'exp':
        valid_mask = test_df['EXP'].isin([0,1,2,3,4])
        y_true = test_df.loc[valid_mask, 'EXP'].astype(int).tolist()
        y_pred = [all_preds[i] for i in valid_mask[valid_mask].index]
    else:
        valid_mask = test_df[args.task.upper()].isin(['0','1','2'])
        y_true = test_df.loc[valid_mask, args.task.upper()].map({'0':0,'1':1,'2':2}).astype(int).tolist()
        y_pred = [all_preds[i] for i in valid_mask[valid_mask].index]

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'n_total_test': len(test_df),
        'n_eval_used': len(y_true),
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'confusion_matrix': cm.tolist()
    }

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'metrics_test.json'), 'w') as f:
        json.dump(metrics, f)
    preds_df.to_csv(os.path.join(args.out_dir, 'preds_test.csv'), index=False)

    print(f"Evaluation complete. n_eval_used: {len(y_true)}, Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")


if __name__ == '__main__':
    main()
