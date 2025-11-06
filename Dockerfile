# Dockerfile for Railway deployment with OpenCV support
# Based on GitHub issue #370: https://github.com/opencv/opencv-python/issues/370
# Solution: Install system libraries required by opencv-python-headless

FROM python:3.13-slim

# Install system dependencies required for OpenCV
# These are needed even for opencv-python-headless in some cases
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
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

# Make boot.sh executable
RUN chmod +x boot.sh

# Expose port (Railway will set PORT env var)
EXPOSE ${PORT:-8000}

# Use boot.sh wrapper, then run uvicorn
CMD ["./boot.sh", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]

