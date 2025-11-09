"""
Train YOLOv8 model for PUBG item detection.

This script trains a YOLOv8 model to detect item grid cells in game screenshots.
After detection, the cropped items will be matched against the item database using CLIP.
"""

import argparse
from ultralytics import YOLO


def train_yolo(
    data_yaml='train_data/dataset.yaml',
    model_size='n',  # n, s, m, l, x
    epochs=100,
    imgsz=640,
    batch=16,
    device='0',  # '0' for GPU, 'cpu' for CPU
    project='runs/detect',
    name='pubg_item_detection',
    patience=20,
    resume=False
):
    """
    Train YOLOv8 detection model.

    Args:
        data_yaml: Path to dataset.yaml configuration file
        model_size: Model size (n, s, m, l, x). 'n' is fastest, 'x' is most accurate
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size (reduce if OOM)
        device: Device to use ('0' for GPU, 'cpu' for CPU)
        project: Project directory
        name: Experiment name
        patience: Early stopping patience
        resume: Resume from last checkpoint
    """

    print("=" * 60)
    print("YOLOv8 Training - PUBG Item Detection")
    print("=" * 60)
    print(f"Dataset: {data_yaml}")
    print(f"Model size: {model_size}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print("=" * 60)
    print()

    # Load model
    if resume:
        print("Resuming from last checkpoint...")
        model = YOLO(f'{project}/{name}/weights/last.pt')
    else:
        print(f"Loading YOLOv8{model_size} model...")
        model = YOLO(f'yolov8{model_size}.pt')

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save=True,
        plots=True,
        val=True,
        # Augmentation settings
        hsv_h=0.015,      # HSV-Hue augmentation
        hsv_s=0.7,        # HSV-Saturation augmentation
        hsv_v=0.4,        # HSV-Value augmentation
        degrees=0.0,      # Rotation (disabled for grid layout)
        translate=0.1,    # Translation
        scale=0.5,        # Scale augmentation
        shear=0.0,        # Shear (disabled for grid layout)
        perspective=0.0,  # Perspective (disabled for grid layout)
        flipud=0.0,       # Vertical flip (disabled)
        fliplr=0.0,       # Horizontal flip (disabled)
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.0,        # Mixup augmentation
        copy_paste=0.0,   # Copy-paste augmentation
    )

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best model: {project}/{name}/weights/best.pt")
    print(f"Last model: {project}/{name}/weights/last.pt")
    print(f"Results: {project}/{name}/")
    print("=" * 60)

    # Validate
    print()
    print("Validating best model...")
    model = YOLO(f'{project}/{name}/weights/best.pt')
    metrics = model.val()

    print()
    print("Validation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")

    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 for PUBG item detection')
    parser.add_argument('--data', type=str, default='train_data/dataset.yaml',
                        help='Path to dataset.yaml')
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='pubg_item_detection',
                        help='Experiment name')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')

    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume
    )