# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PUBG (PlayerUnknown's Battlegrounds) item classification and recognition system. The project automatically identifies items from game screenshots and outputs their unique codes or names.

### System Goal

Extract and identify all items appearing in game screenshots, integrating:
- Image detection (object detection)
- Visual feature extraction
- Vector retrieval (ANN search)
- Database management and incremental updates

### Business Workflow (详见 物品识别业务流程.txt)

The complete pipeline consists of 9 stages:

1. **Screenshot Collection**: JPEG/PNG inputs from players or automated capture
2. **Image Preprocessing**: Resolution normalization (512×512), brightness/contrast enhancement, UI noise removal
3. **Object Detection**: Detect candidate item regions using YOLO/DETR/GroundingDINO
4. **Feature Extraction**: Extract high-dimensional feature vectors (512/1024-dim) using CLIP/ViT/ConvNeXt
5. **Vector Retrieval**: Fast ANN search in item database using Faiss/ScaNN/Milvus
6. **Result Filtering**: Confidence thresholding, NMS for duplicate removal, result merging
7. **Database Layer**: Item metadata, vector indices, version management
8. **Incremental Updates**: Automatic embedding generation for new items
9. **Output Layer**: Recognition results (item ID, name, position, confidence)

**Data Flow**: Image Input → Preprocessing → Detection → Feature Extraction → Vector Search → Filtering → Output Results

## Dataset Structure

The project uses a structured dataset located in `dataset/`:

- **images/**: Contains 8,642 PNG images of PUBG items
- **labels.json**: Complete metadata for all items (2.7MB, 86,432 lines)
  - Structure: item_code, filename, Chinese name, primary_category, secondary_category, rarity, url, filepath
- **category_mapping.json**: Maps Chinese category names to numeric indices
  - Categories: 其他(Other), 外观(Appearance), 服装(Clothing), 武器(Weapons), 盒子(Boxes), 装备(Equipment)
- **item_code_mapping.json**: Maps item codes (e.g., "11010018") to indices (403KB)
- **features_db.pkl**: Pre-computed feature database (1.4GB) for item recognition

### Item Code Format

Item codes follow a pattern like `11010018`, `11020019`, etc. The prefix appears to indicate category:
- `1101xxxx`: Clothing/Upper body
- `1102xxxx`: Clothing/Lower body
- `1103xxxx`: Clothing/Footwear
- `1104xxxx`: Clothing/Hands
- Similar patterns for weapons, equipment, etc.

## Current Implementation Status (2025-11-07)

### Adopted Technical Approach: Two-Stage Pipeline

Due to limited training data (game screenshots are not abundant), we use a two-stage approach:

**Stage 1: YOLOv8 Object Detection**
- Task: Detect item grid cells in game screenshots (single class: "item")
- Training requirement: 100-200 labeled game screenshots
- Output: Bounding boxes for each item cell

**Stage 2: CLIP Feature Matching**
- Extract features from detected item regions
- Match against pre-computed features from dataset/images (8,642 items)
- Use Faiss for fast similarity search
- Output: Item identification (code, name, category)

### Why Two-Stage vs End-to-End Classification?

| Approach | Training Data Needed | Annotation Effort | Extensibility |
|----------|---------------------|-------------------|---------------|
| **Two-Stage (Adopted)** | 100-200 screenshots | Low (single class) | Easy to add new items |
| End-to-End YOLOv8 | 5,000-10,000+ screenshots | Very High (8,642 classes) | Requires retraining |

## Technology Stack

### Computer Vision & ML
- **Object Detection**: YOLOv8 (Ultralytics) for item cell detection
- **Feature Extraction**: OpenAI CLIP for visual embeddings
- **Vector Search**: Faiss for approximate nearest neighbor search

### Core Dependencies
```
ultralytics>=8.0.0      # YOLOv8
torch>=2.0.0            # PyTorch
openai-clip             # CLIP model
faiss-cpu               # Vector similarity search
opencv-python>=4.8.0    # Image processing
pillow>=9.0.0           # Image I/O
numpy>=1.24.0           # Numerical operations
```

## Development Workflow

### Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. **Annotate Training Data** (100-200 game screenshots)
   ```bash
   # Install LabelImg for annotation
   pip install labelImg
   labelImg train_data/images/ train_data/label/
   ```
   - Mark each item cell with bounding box
   - Set class name to "item"
   - Save in YOLO format

2. **Verify Feature Database**
   ```bash
   python scripts/extract_features.py --verify
   ```

### Training YOLOv8

```bash
# Train item detection model
python scripts/train_yolo.py --data train_data/dataset.yaml --epochs 100
```

### Run Inference

```bash
# Detect and identify items from a screenshot
python scripts/inference.py --input path/to/screenshot.png --output results/
```

### Output Files

Generated outputs include:
- `results/*_detected.png`: Visualization with bounding boxes
- `results/*_identified.json`: Recognition results (item codes, names, positions)
- `extracted_cells/`: Individual cropped item images
- `calibration_samples/`: Debug/tuning samples

## Data Handling Notes

- The labels.json file is very large (2.7MB) - use streaming/chunked reads for processing
- The features_db.pkl is 1.4GB - ensure sufficient memory when loading
- Item names and categories are in Chinese
- Each item has associated rarity levels: 经典(Classic), 特殊(Special), 未知(Unknown), etc.

## Key Architecture Considerations

1. **Multi-Stage Pipeline**: The system follows a modular architecture:
   - Input Layer: Screenshot upload and logging
   - Preprocessing Layer: Image normalization and noise removal
   - Detection Layer: Candidate region extraction (bounding boxes + confidence)
   - Feature Layer: Visual encoding to high-dimensional vectors
   - Retrieval Layer: Fast similarity search in vector database
   - Filtering Layer: Confidence thresholding and NMS
   - Output Layer: Final recognition results

2. **Feature Database**: Pre-computed features stored in features_db.pkl (1.4GB)
   - Contains embeddings for all 8,642 items in the reference database
   - Enables fast similarity-based retrieval without re-encoding reference items
   - Requires periodic updates when new items are added to the game

3. **Multi-level Classification**: Items have hierarchical organization:
   - primary_category (6 main types): 服装/Clothing, 武器/Weapons, 装备/Equipment, etc.
   - secondary_category (subcategories): 上衣/Upper, 腿部/Legs, 手部/Hands, 鞋子/Footwear, etc.

4. **Dual-Direction Data Flow**:
   - Forward: Models use database vector indices for retrieval
   - Backward: Database is updated with new embeddings generated by models

5. **Version Management**: Support for game version updates
   - Incremental item additions
   - Historical version tracking
   - Automatic embedding generation pipeline for new items

## Project Structure

```
pubgItem/
├── dataset/                          # Reference item database
│   ├── images/                       # 8,642 item images (PNG)
│   ├── labels.json                   # Item metadata (2.7MB)
│   ├── category_mapping.json         # Category indices
│   ├── item_code_mapping.json        # Item code to index mapping
│   └── features_db.pkl               # Pre-computed CLIP features (1.4GB)
│
├── train_data/                       # YOLOv8 training data (to be created)
│   ├── images/                       # Game screenshot images
│   ├── labels/                       # YOLO format annotations
│   └── dataset.yaml                  # YOLOv8 dataset config
│
├── models/                           # Trained models (to be created)
│   └── yolov8_item_detection.pt      # Trained YOLOv8 weights
│
├── scripts/                          # Implementation scripts (to be created)
│   ├── train_yolo.py                 # Train YOLOv8 detector
│   ├── extract_features.py           # Build/update feature database
│   ├── inference.py                  # End-to-end inference pipeline
│   └── utils.py                      # Shared utilities
│
├── requirements.txt                  # Python dependencies
├── CLAUDE.md                         # This file
└── .gitignore                        # Git ignore rules
```

## Implementation Modules

### Current Status

- ✅ **Dataset**: 8,642 reference item images available
- ✅ **Feature Database**: features_db.pkl exists (may need verification/rebuild)
- ⏳ **YOLOv8 Training Data**: Needs collection and annotation (100-200 screenshots)
- ⏳ **Detection Module**: YOLOv8 training script to be created
- ⏳ **Inference Pipeline**: End-to-end pipeline to be implemented

### To Be Implemented

1. **Data Annotation Workflow**
   - Collect 100-200 game screenshots (item grid interface)
   - Annotate using LabelImg or Roboflow (single class: "item")
   - Export to YOLO format

2. **YOLOv8 Training Module** (`scripts/train_yolo.py`)
   - Load annotated dataset
   - Train YOLOv8 detection model
   - Evaluate and export weights

3. **Feature Database Manager** (`scripts/extract_features.py`)
   - Verify existing features_db.pkl
   - Rebuild if necessary using CLIP
   - Support incremental updates for new items

4. **Inference Pipeline** (`scripts/inference.py`)
   - Stage 1: YOLOv8 detection → crop item regions
   - Stage 2: CLIP feature extraction → Faiss search
   - Stage 3: Post-processing and result output

5. **Utilities** (`scripts/utils.py`)
   - Image preprocessing helpers
   - Metadata loading and parsing
   - Visualization functions

## Dataset Sources

Images are sourced from the PUBG Items CDN:
- Base URL: https://cdn.pubgitems.info/i-icons/
- Format: {item_code}.png (e.g., 11030012.png)

## Performance Considerations

### Detection Stage
- **Grid Layout**: Items arranged in regular grid (typically 4 columns)
- **Overlapping Detections**: Apply NMS (IoU threshold ~0.5) to remove duplicates
- **Confidence Threshold**: Set to 0.5+ to filter out false positives

### Feature Matching Stage
- **Vector Search**: Faiss IndexFlatL2 or IndexIVFFlat for 8,642 items
- **Top-K Results**: Return top-5 matches with similarity scores
- **Similarity Threshold**: Filter results with cosine similarity > 0.8

### Memory Optimization
- **features_db.pkl (1.4GB)**: Load once at startup, keep in memory for batch processing
- **CLIP Model**: Load on GPU if available, fallback to CPU
- **Batch Inference**: Process multiple detected items simultaneously

### Expected Performance
- **YOLOv8 Inference**: ~50-100ms per screenshot (on GPU)
- **CLIP Encoding**: ~20-30ms per item (on GPU)
- **Faiss Search**: <1ms for top-5 retrieval from 8,642 items
- **Total Pipeline**: ~500ms-1s per screenshot (depends on item count)

## Key Constraints & Decisions

1. **Limited Training Data**: Only ~100-200 game screenshots available
   - ✅ Solution: Single-class detection (YOLOv8) + feature matching (CLIP)
   - ❌ Avoid: Multi-class classification requiring thousands of labeled examples

2. **Large Item Catalog**: 8,642 unique items
   - ✅ Solution: Pre-compute features once, use Faiss for fast retrieval
   - ❌ Avoid: Real-time feature extraction for all reference items

3. **Regular Grid Layout**: Items appear in consistent grid pattern
   - ✅ Opportunity: Can use grid constraints to improve detection accuracy
   - ✅ Opportunity: Post-processing to align detections to grid

4. **Game Updates**: New items added periodically
   - ✅ Solution: Incremental feature database updates
   - No need to retrain YOLOv8 (detection task unchanged)