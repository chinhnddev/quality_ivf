"""
Domain-specific augmentation pipelines for embryo grading using Albumentations.

This module provides task-specific augmentation strategies for:
- ICM/TE tasks: Cell boundary focused with CLAHE enhancement
- EXP task: Overall structure focused with lighter augmentation

Key features:
- Conservative transforms for medical imaging (preserve biological structures)
- Microscope-realistic noise and blur simulation
- Selective heavy augmentation for minority class oversampling
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(task="te", img_size=224):
    """
    Task-specific training augmentation pipeline.
    
    Args:
        task: One of 'icm', 'te', or 'exp'
        img_size: Target image size (default: 224)
    
    Returns:
        Albumentations Compose object with task-specific transforms
    """
    task = task.lower()
    
    if task in ["icm", "te"]:
        # Cell boundary focused augmentation
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.6, border_mode=0),  # Conservative: ±15° only
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),  # Cell boundary enhancement
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4
            ),
            A.GaussNoise(std_range=(5.0/255, 20.0/255), mean_range=(0, 0), p=0.3),  # Microscope noise
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=0,
                p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    elif task == "exp":
        # Lighter augmentation for expansion (global feature)
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.6, border_mode=0),
            A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=0.3),  # Lighter than ICM/TE
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.GaussNoise(std_range=(5.0/255, 15.0/255), mean_range=(0, 0), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    else:
        raise ValueError(f"Unknown task: {task}. Must be one of ['icm', 'te', 'exp']")


def get_val_transforms(img_size=224):
    """
    Validation/test pipeline with no augmentation.
    
    Args:
        img_size: Target image size (default: 224)
    
    Returns:
        Albumentations Compose object with deterministic transforms only
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_heavy_augmentation(task="te", img_size=224):
    """
    Heavy augmentation for minority class oversampling.
    More aggressive transforms for replicated samples.
    
    Args:
        task: One of 'icm', 'te', or 'exp'
        img_size: Target image size (default: 224)
    
    Returns:
        Albumentations Compose object with aggressive transforms
    """
    task = task.lower()
    
    if task in ["icm", "te"]:
        # More aggressive cell boundary augmentation
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.85, 1.15), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.7, border_mode=0),  # Slightly more rotation
            A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.6),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(std_range=(10.0/255, 25.0/255), mean_range=(0, 0), p=0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 1.0), p=0.3),
            A.CoarseDropout(
                num_holes_range=(2, 5),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                fill=0,
                p=0.4
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    elif task == "exp":
        # More aggressive global augmentation
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.85, 1.15), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.7, border_mode=0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4
            ),
            A.GaussNoise(std_range=(10.0/255, 20.0/255), mean_range=(0, 0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(9, 18),
                hole_width_range=(9, 18),
                fill=0,
                p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    else:
        raise ValueError(f"Unknown task: {task}. Must be one of ['icm', 'te', 'exp']")


# Test script
if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    
    print("Testing augmentation pipelines...")
    
    # Create a dummy image
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test each pipeline
    for task in ["icm", "te", "exp"]:
        print(f"\nTesting task: {task.upper()}")
        
        # Train transforms
        train_transform = get_train_transforms(task=task, img_size=224)
        augmented = train_transform(image=dummy_img)
        print(f"  Train transform output shape: {augmented['image'].shape}")
        
        # Heavy transforms
        heavy_transform = get_heavy_augmentation(task=task, img_size=224)
        augmented_heavy = heavy_transform(image=dummy_img)
        print(f"  Heavy transform output shape: {augmented_heavy['image'].shape}")
    
    # Val transforms
    print(f"\nTesting validation transforms")
    val_transform = get_val_transforms(img_size=224)
    augmented_val = val_transform(image=dummy_img)
    print(f"  Val transform output shape: {augmented_val['image'].shape}")
    
    print("\n✓ All augmentation pipelines work correctly!")
