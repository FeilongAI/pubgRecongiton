"""
FastAPI RESTful API for PUBG Item Recognition

This API accepts multiple images and returns deduplicated item codes.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel, HttpUrl
import tempfile
import os
from pathlib import Path
import uvicorn
import requests
from urllib.parse import urlparse

from scripts.inference import PUBGItemRecognizer


# Initialize FastAPI app
app = FastAPI(
    title="PUBG Item Recognition API",
    description="Upload game screenshots to identify and extract item codes",
    version="1.0.0"
)

# Pydantic models for request validation
class ImageUrlRequest(BaseModel):
    """Request model for URL-based image recognition."""
    image_urls: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "image_urls": [
                    "https://example.com/image1.png",
                    "https://example.com/image2.jpg"
                ]
            }
        }


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


def download_image_from_url(url: str, timeout: int = 30) -> str:
    """
    Download an image from URL to a temporary file.

    Args:
        url: Image URL to download
        timeout: Request timeout in seconds

    Returns:
        Path to the downloaded temporary file

    Raises:
        HTTPException: If download fails
    """
    try:
        # Send GET request with timeout
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        # Verify content type is an image
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"URL does not point to an image. Content-Type: {content_type}"
            )

        # Determine file extension from URL or content type
        parsed_url = urlparse(url)
        url_path = parsed_url.path
        ext = Path(url_path).suffix.lower()

        if not ext or ext not in {'.jpg', '.jpeg', '.png', '.webp'}:
            # Fallback to content type
            ext_map = {
                'image/jpeg': '.jpg',
                'image/png': '.png',
                'image/webp': '.webp'
            }
            ext = ext_map.get(content_type, '.jpg')

        # Create temporary file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)

        # Download in chunks
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp_file.write(chunk)

        tmp_file.close()
        return tmp_file.name

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=408,
            detail=f"Download timeout for URL: {url}"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image from URL: {str(e)}"
        )


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


@app.post("/recognize_urls")
async def recognize_items_from_urls(request: ImageUrlRequest):
    """
    Recognize PUBG items from images hosted at URLs (e.g., WeChat cloud storage).

    This endpoint is designed for batch processing of images from cloud storage.
    It downloads images from the provided URLs, processes them, and returns
    deduplicated item codes.

    Args:
        request: JSON body containing list of image URLs
                {
                    "image_urls": [
                        "https://example.com/image1.png",
                        "https://example.com/image2.jpg"
                    ]
                }

    Returns:
        JSON with deduplicated item codes:
        {
            "item_codes": ["11010018", "11020019", ...],
            "total_items_detected": 15,
            "unique_items": 12,
            "images_processed": 3,
            "failed_downloads": 0
        }

    Example:
        POST /recognize_urls
        {
            "image_urls": [
                "https://cloud.example.com/inventory1.png",
                "https://cloud.example.com/inventory2.png"
            ]
        }
    """
    if not request.image_urls:
        raise HTTPException(status_code=400, detail="No image URLs provided")

    if len(request.image_urls) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 images per request. Please batch your requests."
        )

    try:
        # Get recognizer instance
        rec = get_recognizer()

        all_item_codes = []
        total_detections = 0
        failed_downloads = 0
        downloaded_files = []

        # Download and process each image
        for idx, url in enumerate(request.image_urls, 1):
            print(f"Processing image {idx}/{len(request.image_urls)}: {url[:100]}...")

            try:
                # Download image from URL
                tmp_path = download_image_from_url(url)
                downloaded_files.append(tmp_path)

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

                print(f"  → Detected {len(results)} items")

            except HTTPException:
                # Re-raise HTTP exceptions (download failures)
                failed_downloads += 1
                print(f"  → Failed to download image")
                continue
            except Exception as e:
                # Log other errors but continue processing
                failed_downloads += 1
                print(f"  → Recognition error: {str(e)}")
                continue

        # Clean up all downloaded temporary files
        for tmp_path in downloaded_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {tmp_path}: {e}")

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
            "images_processed": len(request.image_urls) - failed_downloads,
            "failed_downloads": failed_downloads
        }

    except Exception as e:
        # Clean up temporary files on error
        for tmp_path in downloaded_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass

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
