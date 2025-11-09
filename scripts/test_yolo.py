"""
Test trained YOLOv8 model on a sample image.
"""

import argparse
from ultralytics import YOLO
from pathlib import Path
import cv2


def test_yolo(model_path, image_path, conf_threshold=0.25, output_dir='results'):
    """
    Test YOLOv8 model on a single image.

    Args:
        model_path: Path to trained model weights
        image_path: Path to test image
        conf_threshold: Confidence threshold for detections
        output_dir: Directory to save results
    """
    print("=" * 60)
    print("YOLOv8 Detection Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print("=" * 60)
    print()

    # Load model
    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        project=output_dir,
        name='yolo_detection',
        exist_ok=True
    )

    # Print results
    print(f"Detected {len(results[0].boxes)} objects")
    print()

    # Show detection details
    for i, box in enumerate(results[0].boxes):
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        xyxy = box.xyxy[0].tolist()

        print(f"Object {i+1}:")
        print(f"  Class: {model.names[cls]}")
        print(f"  Confidence: {conf:.4f}")
        print(f"  BBox: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")

    print()
    print("=" * 60)
    print(f"Results saved to: {output_dir}/yolo_detection/")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test YOLOv8 model')
    parser.add_argument('--model', type=str,
                        default='runs/detect/pubg_item_detection3/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--image', type=str,
                        default='Snipaste_2025-10-25_11-49-08.png',
                        help='Path to test image')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')

    args = parser.parse_args()

    test_yolo(args.model, args.image, args.conf, args.output)