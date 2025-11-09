"""
Split dataset into train and validation sets.
"""

import os
import random
import shutil
from pathlib import Path


def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Split dataset into train and val sets.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO label files
        output_dir: Output directory for split dataset
        train_ratio: Ratio of training data (default: 0.8)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Get all image files
    image_files = list(Path(images_dir).glob('*.png')) + list(Path(images_dir).glob('*.jpg'))

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    # Shuffle
    random.shuffle(image_files)

    # Split
    train_count = int(len(image_files) * train_ratio)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]

    print(f"Total images: {len(image_files)}")
    print(f"Training: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"Validation: {len(val_files)} ({len(val_files)/len(image_files)*100:.1f}%)")
    print()

    # Create directories
    for split in ['train', 'val']:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)

    # Copy files
    for img_file in train_files:
        label_file = Path(labels_dir) / (img_file.stem + '.txt')

        # Copy image
        shutil.copy(img_file, f"{output_dir}/train/images/{img_file.name}")

        # Copy label if exists
        if label_file.exists():
            shutil.copy(label_file, f"{output_dir}/train/labels/{label_file.name}")

    for img_file in val_files:
        label_file = Path(labels_dir) / (img_file.stem + '.txt')

        # Copy image
        shutil.copy(img_file, f"{output_dir}/val/images/{img_file.name}")

        # Copy label if exists
        if label_file.exists():
            shutil.copy(label_file, f"{output_dir}/val/labels/{label_file.name}")

    print(f"Dataset split complete!")
    print(f"Train images: {output_dir}/train/images")
    print(f"Train labels: {output_dir}/train/labels")
    print(f"Val images: {output_dir}/val/images")
    print(f"Val labels: {output_dir}/val/labels")


if __name__ == '__main__':
    split_dataset(
        images_dir='train_data/images',
        labels_dir='train_data/labels',
        output_dir='train_data',
        train_ratio=0.8,
        seed=42
    )