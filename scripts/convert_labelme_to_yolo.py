"""
Convert LabelMe JSON annotations to YOLO format.

LabelMe format:
- JSON files with absolute coordinates
- Format: {"shapes": [{"points": [[x1, y1], [x2, y2]], "label": "class_name"}]}

YOLO format:
- TXT files with normalized coordinates
- Format: class_id x_center y_center width height (all values 0-1)
"""

import json
import os
from pathlib import Path
from PIL import Image


def convert_labelme_to_yolo(json_path, output_dir, classes):
    """
    Convert a single LabelMe JSON file to YOLO format.

    Args:
        json_path: Path to LabelMe JSON file
        output_dir: Directory to save YOLO txt file
        classes: List of class names (e.g., ['item'])
    """
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get image dimensions
    img_width = data.get('imageWidth')
    img_height = data.get('imageHeight')

    # If dimensions not in JSON, try to read from image
    if img_width is None or img_height is None:
        image_path = json_path.replace('.json', '.png')
        if not os.path.exists(image_path):
            image_path = json_path.replace('.json', '.jpg')

        if os.path.exists(image_path):
            img = Image.open(image_path)
            img_width, img_height = img.size
        else:
            print(f"Warning: Could not find image for {json_path}")
            return

    # Convert annotations
    yolo_annotations = []

    for shape in data.get('shapes', []):
        label = shape['label']
        points = shape['points']

        # Get class index
        if label not in classes:
            print(f"Warning: Unknown class '{label}' in {json_path}")
            continue

        class_id = classes.index(label)

        # Convert rectangle coordinates to YOLO format
        # points: [[x1, y1], [x2, y2]]
        x1, y1 = points[0]
        x2, y2 = points[1]

        # Calculate center, width, height
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Normalize to 0-1
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        # Format: class_id x_center y_center width height
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save to txt file
    output_path = os.path.join(output_dir, Path(json_path).stem + '.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))

    return len(yolo_annotations)


def batch_convert(json_dir, output_dir, classes=['item']):
    """
    Convert all LabelMe JSON files in a directory to YOLO format.

    Args:
        json_dir: Directory containing JSON files
        output_dir: Directory to save YOLO txt files
        classes: List of class names
    """
    os.makedirs(output_dir, exist_ok=True)

    json_files = list(Path(json_dir).glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    print(f"Found {len(json_files)} JSON files")
    print(f"Classes: {classes}")
    print(f"Output directory: {output_dir}")
    print()

    total_annotations = 0

    for json_file in json_files:
        num_annotations = convert_labelme_to_yolo(str(json_file), output_dir, classes)
        if num_annotations:
            total_annotations += num_annotations
            print(f"[OK] {json_file.name}: {num_annotations} objects")

    print()
    print(f"Conversion complete!")
    print(f"Total files converted: {len(json_files)}")
    print(f"Total annotations: {total_annotations}")
    print(f"Average annotations per image: {total_annotations / len(json_files):.1f}")


if __name__ == '__main__':
    # Convert LabelMe JSON to YOLO format
    json_dir = 'train_data/label'
    output_dir = 'train_data/labels'
    classes = ['item']  # Single class: item

    batch_convert(json_dir, output_dir, classes)