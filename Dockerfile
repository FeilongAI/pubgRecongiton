# Multi-stage build for smaller final image
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Final stage
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app.py .
COPY scripts/ ./scripts/

# Copy dataset and models
# Note: These are large files (features_db.pkl is 1.4GB)
COPY dataset/features_db.pkl ./dataset/features_db.pkl
COPY dataset/labels.json ./dataset/labels.json
COPY dataset/category_mapping.json ./dataset/category_mapping.json
COPY dataset/item_code_mapping.json ./dataset/item_code_mapping.json

# Copy trained YOLOv8 model
COPY runs/detect/pubg_item_detection3/weights/best.pt ./runs/detect/pubg_item_detection3/weights/best.pt

# Create directory for temporary results
RUN mkdir -p temp_results

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
