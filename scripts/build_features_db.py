"""
Build CLIP features database from dataset/images/

This script extracts CLIP features from all item images and saves them
for fast similarity search during inference.
"""

import clip
import torch
import pickle
import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import ssl
import urllib.request

# Disable SSL verification (for downloading CLIP model)
ssl._create_default_https_context = ssl._create_unverified_context


def build_features_db(
    images_dir='../dataset/images',
    labels_file='../dataset/labels.json',
    output_file='../dataset/features_db.pkl',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=32
):
    """
    Build CLIP features database from item images.

    Args:
        images_dir: Directory containing item images
        labels_file: JSON file with item metadata
        output_file: Path to save features database
        device: Device to run CLIP on
        batch_size: Batch size for feature extraction
    """
    print("=" * 70)
    print("Building CLIP Features Database")
    print("=" * 70)
    print(f"Images directory: {images_dir}")
    print(f"Labels file: {labels_file}")
    print(f"Output file: {output_file}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print("=" * 70)
    print()

    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"CLIP model loaded on {device}")
    print()

    # Load item metadata
    print("Loading item metadata...")
    with open(labels_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract items list
    labels_data = data.get('items', [])

    print(f"Loaded {len(labels_data)} items")
    print()

    # Get all image paths
    image_dir_path = Path(images_dir)
    image_files = sorted(list(image_dir_path.glob('*.png')))

    print(f"Found {len(image_files)} images")
    print()

    # Extract features
    print("Extracting CLIP features...")

    features_dict = {}
    item_code_to_info = {}

    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:i+batch_size]

        # Load and preprocess images
        batch_images = []
        batch_codes = []

        for img_file in batch_files:
            try:
                img = Image.open(img_file).convert('RGB')
                img_tensor = preprocess(img)
                batch_images.append(img_tensor)

                # Get item code from filename
                item_code = img_file.stem  # e.g., '11010018'
                batch_codes.append(item_code)

            except Exception as e:
                print(f"\\nError loading {img_file}: {e}")
                continue

        if not batch_images:
            continue

        # Stack into batch
        batch_tensor = torch.stack(batch_images).to(device)

        # Extract features
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features = features.cpu().numpy()

        # Store features
        for code, feat in zip(batch_codes, features):
            features_dict[code] = feat

            # Store metadata
            # Find metadata for this item code
            item_info = None
            for item in labels_data:
                if str(item.get('item_code')) == code:
                    item_info = item
                    break

            if item_info:
                item_code_to_info[code] = {
                    'name': item_info.get('name', item_info.get('chinese_name', 'Unknown')),
                    'primary_category': item_info.get('primary_category', ''),
                    'secondary_category': item_info.get('secondary_category', ''),
                    'rarity': item_info.get('rarity', ''),
                }

    print()
    print(f"Extracted features for {len(features_dict)} items")
    print()

    # Save database
    print(f"Saving features database to {output_file}...")

    db = {
        'features': features_dict,
        'metadata': item_code_to_info,
        'model': 'ViT-B/32',
        'feature_dim': 512
    }

    with open(output_file, 'wb') as f:
        pickle.dump(db, f)

    file_size = Path(output_file).stat().st_size / (1024**2)
    print(f"Database saved ({file_size:.1f} MB)")
    print()
    print("=" * 70)
    print("Features database built successfully!")
    print("=" * 70)
    print()
    print(f"Total items: {len(features_dict)}")
    print(f"Feature dimension: 512")
    print(f"Database file: {output_file}")


if __name__ == '__main__':
    build_features_db()