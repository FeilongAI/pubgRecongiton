"""
FastAPI RESTful API for PUBG Item Recognition

This API accepts multiple images and returns deduplicated item codes.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import tempfile
import os
from pathlib import Path
import uvicorn

from scripts.inference import PUBGItemRecognizer


# Initialize FastAPI app
app = FastAPI(
    title="PUBG Item Recognition API",
    description="Upload game screenshots to identify and extract item codes",
    version="1.0.0"
)

# Global recognizer instance (lazy loading)
recognizer = None


def get_recognizer():
    """Get or initialize the recognizer instance."""
    global recognizer
    if recognizer is None:
        print("Initializing PUBG Item Recognizer...")
        recognizer = PUBGItemRecognizer(
            yolo_model_path='runs/detect/pubg_item_detection3/weights/best.pt',
            features_db_path='dataset/features_db.pkl',
            conf_threshold=0.25
        )
        print("Recognizer initialized successfully!")
    return recognizer


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "PUBG Item Recognition API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    try:
        rec = get_recognizer()
        return {
            "status": "healthy",
            "model_loaded": rec is not None,
            "database_items": len(rec.item_codes) if rec else 0
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/recognize")
async def recognize_items(images: List[UploadFile] = File(...)):
    """
    Recognize PUBG items from multiple screenshots.

    Args:
        images: List of image files (JPEG/PNG)

    Returns:
        JSON with deduplicated item codes:
        {
            "item_codes": ["11010018", "11020019", ...],
            "total_items_detected": 15,
            "unique_items": 12,
            "images_processed": 3
        }
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")

    # Validate file types
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    for img in images:
        ext = Path(img.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {img.filename}. Only JPG/PNG allowed."
            )

    try:
        # Get recognizer instance
        rec = get_recognizer()

        all_item_codes = []
        total_detections = 0

        # Process each image
        for img_file in images:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(img_file.filename).suffix) as tmp:
                content = await img_file.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                # Run recognition (without saving visualization)
                results = rec.recognize(
                    tmp_path,
                    output_dir='temp_results',
                    save_visualization=False
                )

                # Extract item codes
                for item in results:
                    all_item_codes.append(item['item_code'])
                    total_detections += 1

            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Deduplicate item codes (preserve order of first occurrence)
        seen = set()
        unique_item_codes = []
        for code in all_item_codes:
            if code not in seen:
                seen.add(code)
                unique_item_codes.append(code)

        return {
            "item_codes": unique_item_codes,
            "total_items_detected": total_detections,
            "unique_items": len(unique_item_codes),
            "images_processed": len(images)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recognition failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Preload models on startup."""
    print("=" * 70)
    print("Starting PUBG Item Recognition API...")
    print("=" * 70)
    # Lazy loading - models will be loaded on first request
    print("Models will be loaded on first request")
    print("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down API...")


if __name__ == "__main__":
    # Run with: python app.py
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )
