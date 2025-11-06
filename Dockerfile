# Dockerfile for Railway deployment with OpenCV support
# Based on GitHub issue #370: https://github.com/opencv/opencv-python/issues/370
# Solution: Install system libraries required by opencv-python-headless

FROM python:3.13-slim

# Install system dependencies required for OpenCV
# These are needed even for opencv-python-headless in some cases
# Note: libgl1-mesa-glx was replaced with libgl1 in newer Debian versions (Trixie+)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgthread-2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables to prevent OpenCV from loading GUI libraries
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=offscreen
ENV OPENCV_DISABLE_LIBGL=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure directories exist
RUN mkdir -p files models

# Verify model file if it exists
RUN if [ -f "models/license_plate_detector.pt" ]; then \
        echo "✓ Custom model file found in models/"; \
        ls -lh models/*.pt; \
    else \
        echo "⚠ No custom model file found - will use default YOLO model"; \
    fi

# Verify video file if it exists
RUN if [ -f "files/2.mp4" ]; then \
        echo "✓ Test video file found"; \
    else \
        echo "⚠ Test video file not found - video endpoints may not work"; \
    fi

# Make boot.sh executable
RUN chmod +x boot.sh

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Use boot.sh wrapper, then run uvicorn
# boot.sh will handle PORT variable expansion
CMD ["./boot.sh", "uvicorn", "main:app"]

