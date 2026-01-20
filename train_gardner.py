#!/usr/bin/env python3
import argparse
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(description='Train IVF-EffiMorphPP for Gardner criteria prediction')
    parser.add_argument('--task', type=str, required=True, choices=['exp', 'icm', 'te'], help='Task to train: exp, icm, or te')
    parser.add_argument('--mode', type=str, required=True, choices=['benchmark_fair', 'improved'], help='Training mode')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Path to save checkpoints')

    args = parser.parse_args()

    train_model(
        task=args.task,
        mode=args.mode,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
