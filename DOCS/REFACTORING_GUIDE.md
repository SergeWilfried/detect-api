# API Refactoring Guide

## Overview
This document explains the new modular structure for the License Plate Detection API and how to complete the refactoring.

## ✅ Completed Steps

### 1. Project Structure Created
```
detect-api/
├── core/
│   ├── __init__.py
│   ├── config.py          ✅ Centralized configuration
│   └── exceptions.py      ✅ Custom exceptions
├── models/
│   ├── __init__.py
│   ├── requests.py        ✅ Request models
│   └── responses.py       ✅ Response models
├── services/
│   ├── __init__.py
│   ├── storage_service.py ✅ MongoDB/Redis operations
│   ├── gemini_service.py  ✅ Gemini AI service
│   └── video_service.py   ✅ Video processing logic
├── api/
│   ├── __init__.py
│   ├── dependencies.py    ✅ FastAPI dependencies
│   └── routes/
│       ├── __init__.py
│       ├── health.py      ⏳ TO DO
│       ├── detection.py   ⏳ TO DO
│       ├── video.py       ⏳ TO DO
│       ├── jobs.py        ⏳ TO DO
│       └── gemini.py      ⏳ TO DO
├── main_old.py            ✅ Backup of original
├── main.py                ⏳ TO DO (new simplified version)
└── detection_service.py   ✅ Existing (keep as is)
```

### 2. Core Modules

#### `core/config.py`
- Centralized configuration using Pydantic Settings
- All environment variables in one place
- Type-safe configuration with validation
- Usage: `from core.config import settings`

#### `core/exceptions.py`
- Custom exception classes for better error handling
- Types: DetectionError, VideoProcessingError, OCRError, etc.

### 3. Models

#### `models/requests.py`
- DetectRequest
- GeminiDetectRequest
- GeminiSegmentRequest

#### `models/responses.py`
- DetectResponse, Detection, BoundingBox
- VideoProcessResponse, PlateSummary, VideoProcessingStats
- JobStatus, JobSubmitResponse
- GeminiDetectResponse, GeminiSegmentResponse

### 4. Services

#### `services/storage_service.py`
- MongoDB and Redis connection management
- CRUD operations for detections, jobs, frames
- Job status management
- Caching operations

#### `services/gemini_service.py`
- Extracted from detection_service.py
- Object detection and segmentation
- Image loading and visualization

#### `services/video_service.py`
- Background video processing logic
- Integrates with storage_service and detector

#### `api/dependencies.py`
- Singleton pattern for detector and services
- Dependency injection functions

## 📋 Next Steps

### Step 1: Create Route Files

Create the following route files by extracting endpoints from `main_old.py`:

#### `api/routes/health.py`
```python
from fastapi import APIRouter
from api.dependencies import get_storage_service

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    \"\"\"Health check endpoint\"\"\"
    storage = get_storage_service()

    health_status = {
        "status": "healthy",
        "service": "license-plate-detection-api",
        "version": "1.0.0",
        "databases": {
            "mongodb": "connected" if storage.mongo_client is not None else "disconnected",
            "redis": "connected" if storage.redis_client is not None else "disconnected"
        }
    }

    # Test database connections
    if storage.mongo_client is not None:
        try:
            storage.mongo_client.admin.command('ping')
            health_status["databases"]["mongodb"] = "connected"
        except:
            health_status["databases"]["mongodb"] = "error"

    if storage.redis_client is not None:
        try:
            storage.redis_client.ping()
            health_status["databases"]["redis"] = "connected"
        except:
            health_status["databases"]["redis"] = "error"

    return health_status

@router.get("/")
async def root():
    \"\"\"Root endpoint with API information\"\"\"
    from core.config import settings

    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "endpoints": {
            "/health": "Health check",
            "/detect": "POST - Detect license plates (base64 or URL)",
            "/detect/upload": "POST - Detect license plates (file upload)",
            "/process/video/upload/async": "POST - Process video (async)",
            "/jobs/{job_id}": "GET - Get job status",
            "/docs": "API documentation"
        }
    }
```

#### `api/routes/detection.py`
```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import cv2
import numpy as np
import io

from api.dependencies import get_detector
from models.requests import DetectRequest
from models.responses import DetectResponse, Detection, BoundingBox

router = APIRouter(prefix="/detect", tags=["detection"])

@router.post("/", response_model=DetectResponse)
async def detect(request: DetectRequest):
    \"\"\"License plate detection from base64 or URL\"\"\"
    detector = get_detector()

    result = {}
    image = None

    if request.image_url:
        result = detector.detect_from_url(request.image_url)
        if result.get("detected") and request.include_visualization:
            image = detector.load_image_from_url(request.image_url)
    elif request.data:
        result = detector.detect_from_base64(request.data)
        if result.get("detected") and request.include_visualization:
            image = detector.load_image_from_base64(request.data)
    else:
        raise HTTPException(400, "Either 'data' or 'image_url' must be provided")

    if "error" in result:
        raise HTTPException(400, result["error"])

    # Calculate average confidence
    avg_confidence = None
    if result["detections"]:
        avg_confidence = sum(d["confidence"] for d in result["detections"]) / len(result["detections"])

    # Get visualization
    visualization = None
    if request.include_visualization and image is not None and result["detections"]:
        visualization = detector.get_visualization(image, result["detections"])

    # Convert to response model
    detections = [
        Detection(
            class_name=d["class_name"],
            confidence=d["confidence"],
            bbox=BoundingBox(**d["bbox"]),
            plate_text=d.get("plate_text", ""),
            ocr_confidence=d.get("ocr_confidence", 0.0)
        )
        for d in result["detections"]
    ]

    return DetectResponse(
        detected=result["detected"],
        count=result["count"],
        detections=detections,
        message=f"Found {result['count']} detection(s)" if result["detected"] else "No license plates detected",
        confidence=round(avg_confidence, 4) if avg_confidence else None,
        image_shape=result.get("image_shape"),
        visualization=visualization
    )

@router.post("/upload", response_model=DetectResponse)
async def detect_upload(
    file: UploadFile = File(...),
    include_visualization: bool = Form(False)
):
    \"\"\"License plate detection from uploaded file\"\"\"
    detector = get_detector()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Read file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect
    detections = detector.detect_license_plates(image_np)

    # Calculate confidence
    avg_confidence = None
    if detections:
        avg_confidence = sum(d["confidence"] for d in detections) / len(detections)

    # Visualization
    visualization = None
    if include_visualization and detections:
        visualization = detector.get_visualization(image_np, detections)

    # Convert to response
    detection_models = [
        Detection(
            class_name=d["class_name"],
            confidence=d["confidence"],
            bbox=BoundingBox(**d["bbox"]),
            plate_text=d.get("plate_text", ""),
            ocr_confidence=d.get("ocr_confidence", 0.0)
        )
        for d in detections
    ]

    return DetectResponse(
        detected=len(detections) > 0,
        count=len(detections),
        detections=detection_models,
        message=f"Found {len(detections)} detection(s)" if detections else "No license plates detected",
        confidence=round(avg_confidence, 4) if avg_confidence else None,
        image_shape={"height": image_np.shape[0], "width": image_np.shape[1]},
        visualization=visualization
    )
```

#### `api/routes/video.py`
Extract video endpoints: `/process/video/upload/async`, etc.

#### `api/routes/jobs.py`
Extract job endpoints: `/jobs/{job_id}`, `/jobs/{job_id}/result`, etc.

#### `api/routes/gemini.py`
Extract Gemini endpoints: `/gemini/detect`, `/gemini/segment`, etc.

### Step 2: Create New main.py

```python
# Prevent OpenCV from trying to load GUI libraries
import os
os.environ['OPENCV_DISABLE_LIBGL'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import settings
from api.dependencies import get_detector, get_storage_service

# Import route modules (create these next)
from api.routes import health, detection, video, jobs, gemini


@asynccontextmanager
async def lifespan(app: FastAPI):
    \"\"\"Startup and shutdown events\"\"\"
    # Startup
    print("Initializing License Plate Detection API...")

    # Initialize services
    storage = get_storage_service()
    storage.connect_mongodb()
    storage.connect_redis()

    # Initialize detector
    try:
        get_detector()
        print("✓ Detector initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize detector: {e}")

    yield

    # Shutdown
    print("Shutting down...")
    storage.close_connections()


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "status_code": 422}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "status_code": 500}
    )

# Include routers
app.include_router(health.router)
app.include_router(detection.router)
app.include_router(video.router)
app.include_router(jobs.router)
app.include_router(gemini.router)
```

### Step 3: Update requirements.txt

Add pydantic-settings:
```
pydantic-settings>=2.0.0
```

### Step 4: Update Imports

Since GeminiImageService moved, update imports in `detection_service.py` if needed (or keep it separate).

### Step 5: Test

```bash
# Install new dependency
pip install pydantic-settings

# Run the application
uvicorn main:app --reload

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/
```

## Benefits of New Structure

1. **Maintainability**: Each file < 300 lines, easy to understand
2. **Testability**: Services can be tested independently
3. **Scalability**: Easy to add new routes/services
4. **Type Safety**: Centralized configuration with validation
5. **Separation of Concerns**: Clear boundaries between layers
6. **Reusability**: Services can be imported anywhere

## Migration Checklist

- [x] Create directory structure
- [x] Create core modules (config, exceptions)
- [x] Create models (requests, responses)
- [x] Create services (storage, gemini, video)
- [x] Create api/dependencies.py
- [x] Backup main_old.py
- [ ] Create api/routes/health.py
- [ ] Create api/routes/detection.py
- [ ] Create api/routes/video.py
- [ ] Create api/routes/jobs.py
- [ ] Create api/routes/gemini.py
- [ ] Create new main.py
- [ ] Test all endpoints
- [ ] Update documentation
- [ ] Deploy

## Next Actions

1. Copy the route code from `main_old.py` to respective route files
2. Update imports in route files to use new models/services
3. Create new `main.py` with router inclusions
4. Test thoroughly
5. Delete `main_old.py` once confirmed working

## Questions?

Refer to:
- `main_old.py` for original endpoint implementations
- `core/config.py` for configuration usage
- `api/dependencies.py` for service injection patterns
