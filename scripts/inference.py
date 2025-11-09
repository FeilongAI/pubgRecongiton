"""
Complete inference pipeline: YOLOv8 Detection + CLIP Matching

This script performs end-to-end PUBG item recognition:
1. Detect item grid cells using trained YOLOv8
2. Extract features from detected regions using CLIP
3. Match against item database using similarity search
4. Output recognized items with names and metadata
"""

import argparse
import pickle
import json
import clip
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path


class PUBGItemRecognizer:
    def __init__(
        self,
        yolo_model_path='runs/detect/pubg_item_detection3/weights/best.pt',
        features_db_path='dataset/features_db.pkl',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        conf_threshold=0.25
    ):
        """
        Initialize PUBG item recognizer.

        Args:
            yolo_model_path: Path to trained YOLOv8 model
            features_db_path: Path to CLIP features database
            device: Device to run models on
            conf_threshold: Confidence threshold for YOLOv8 detections
        """
        self.device = device
        self.conf_threshold = conf_threshold

        print("Initializing PUBG Item Recognizer...")
        print(f"Device: {device}")
        print()

        # Load YOLOv8 model
        print("Loading YOLOv8 detection model...")
        self.yolo_model = YOLO(yolo_model_path)
        print("YOLOv8 loaded")

        # Load CLIP model
        print("Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        print("CLIP loaded")

        # Load features database
        print("Loading features database...")
        with open(features_db_path, 'rb') as f:
            self.db = pickle.load(f)

        self.features_dict = self.db['features']
        self.metadata = self.db['metadata']

        # Build feature matrix and item codes list
        self.item_codes = list(self.features_dict.keys())
        self.feature_matrix = np.stack([self.features_dict[code] for code in self.item_codes])

        print(f"Database loaded: {len(self.item_codes)} items")
        print()

    def detect_items(self, image_path):
        """
        Detect item grid cells using YOLOv8.

        Args:
            image_path: Path to input image

        Returns:
            List of detections with bounding boxes
        """
        results = self.yolo_model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False
        )

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf
            })

        return detections

    def extract_clip_features(self, image, bbox):
        """
        Extract CLIP features from a bounding box region.

        Args:
            image: PIL Image or numpy array
            bbox: [x1, y1, x2, y2]

        Returns:
            Feature vector (512-dim)
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Crop region
        x1, y1, x2, y2 = bbox
        region = image.crop((x1, y1, x2, y2))

        # Preprocess and extract features
        image_tensor = self.preprocess(region).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensor)
            features = features.cpu().numpy()[0]

        return features

    def find_similar_item(self, query_features, top_k=5):
        """
        Find most similar items in database.

        Args:
            query_features: Query feature vector
            top_k: Number of top matches to return

        Returns:
            List of (item_code, similarity) tuples
        """
        # Normalize features
        query_norm = query_features / np.linalg.norm(query_features)
        db_norm = self.feature_matrix / np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(db_norm, query_norm)

        # Get top-k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            item_code = self.item_codes[idx]
            similarity = similarities[idx]
            results.append((item_code, float(similarity)))

        return results

    def recognize(self, image_path, output_dir='results', save_visualization=True):
        """
        Complete recognition pipeline.

        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            save_visualization: Whether to save visualization

        Returns:
            List of recognized items
        """
        print("=" * 70)
        print("PUBG Item Recognition")
        print("=" * 70)
        print(f"Input image: {image_path}")
        print()

        # Load image
        image_cv = cv2.imread(image_path)
        image_pil = Image.open(image_path).convert('RGB')

        # Step 1: Detect items
        print("Step 1: Detecting item grid cells with YOLOv8...")
        detections = self.detect_items(image_path)
        print(f"Detected {len(detections)} items")
        print()

        # Step 2: Recognize each item
        print("Step 2: Recognizing items with CLIP...")
        recognized_items = []

        for i, det in enumerate(detections):
            bbox = det['bbox']
            det_conf = det['confidence']

            # Extract CLIP features
            features = self.extract_clip_features(image_pil, bbox)

            # Find similar items
            matches = self.find_similar_item(features, top_k=1)
            item_code, similarity = matches[0]

            # Get metadata
            meta = self.metadata.get(item_code, {})

            result = {
                'id': i + 1,
                'bbox': bbox,
                'detection_confidence': det_conf,
                'item_code': item_code,
                'similarity': similarity,
                'name': meta.get('name', 'Unknown'),
                'primary_category': meta.get('primary_category', ''),
                'secondary_category': meta.get('secondary_category', ''),
                'rarity': meta.get('rarity', '')
            }

            recognized_items.append(result)

            print(f"  [{i+1}/{len(detections)}] {result['name']} (相似度: {similarity:.3f})")

        print()

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_file = output_path / f"{Path(image_path).stem}_recognized.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(recognized_items, f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {json_file}")

        # Save visualization
        if save_visualization:
            vis_image = image_cv.copy()

            for item in recognized_items:
                x1, y1, x2, y2 = item['bbox']
                name = item['name']
                similarity = item['similarity']

                # Draw bbox
                color = (0, 255, 0)  # Green
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{name} ({similarity:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

                # Background for text
                cv2.rectangle(vis_image,
                              (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1),
                              color, -1)

                # Text
                cv2.putText(vis_image, label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1)

            vis_file = output_path / f"{Path(image_path).stem}_recognized.png"
            cv2.imwrite(str(vis_file), vis_image)
            print(f"Visualization saved to: {vis_file}")

        print()
        print("=" * 70)
        print(f"Recognition complete! Found {len(recognized_items)} items")
        print("=" * 70)

        return recognized_items


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PUBG Item Recognition')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--yolo-model', type=str,
                        default='runs/detect/pubg_item_detection3/weights/best.pt',
                        help='Path to YOLOv8 model')
    parser.add_argument('--features-db', type=str,
                        default='dataset/features_db.pkl',
                        help='Path to features database')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='YOLOv8 confidence threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Initialize recognizer
    recognizer = PUBGItemRecognizer(
        yolo_model_path=args.yolo_model,
        features_db_path=args.features_db,
        device=args.device,
        conf_threshold=args.conf
    )

    # Recognize items
    results = recognizer.recognize(args.image, output_dir=args.output)