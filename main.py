from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
from pathlib import Path
from ultralytics import YOLO
from detection_service import LicensePlateDetector
import cv2
import numpy as np
from PIL import Image
import io

# File paths for testing
MODEL_PATH = "./models/license_plate_detector.pt"
VIDEO_PATH = "./files/deneme.mp4"

# Global variables for loaded resources
license_plate_detector: Optional[YOLO] = None
video_cap: Optional[cv2.VideoCapture] = None
app = FastAPI(
    title="License Plate Detection API",
    description="License plate detection using YOLOv11 - Based on Medium article implementation",
    version="1.0.0"
)

# Initialize the detector (singleton pattern)
detector: Optional[LicensePlateDetector] = None


def get_detector() -> LicensePlateDetector:
    """Get or create the detector instance"""
    global detector
    if detector is None:
        # You can specify a custom model path via environment variable
        model_path = os.getenv("YOLO_MODEL_PATH")
        confidence = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
        detector = LicensePlateDetector(model_path=model_path, confidence_threshold=confidence)
    return detector


class DetectRequest(BaseModel):
    """Request model for detection endpoint"""
    data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL of the image to process")
    include_visualization: bool = Field(False, description="Include annotated image in response")


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float


class Detection(BaseModel):
    """Single detection result"""
    class_name: str
    confidence: float
    bbox: BoundingBox


class DetectResponse(BaseModel):
    """Response model for detection endpoint"""
    detected: bool
    count: int
    detections: List[Detection]
    message: str
    confidence: Optional[float] = None
    image_shape: Optional[Dict[str, int]] = None
    visualization: Optional[str] = Field(None, description="Base64 encoded annotated image")


def load_test_resources():
    """Load model and video file for testing"""
    global license_plate_detector, video_cap
    
    # Load model if it exists
    try:
        if Path(MODEL_PATH).exists():
            try:
                license_plate_detector = YOLO(MODEL_PATH)
                print(f"✓ Loaded model from {MODEL_PATH}")
            except Exception as e:
                print(f"⚠ Warning: Could not load model from {MODEL_PATH}: {e}")
                print("   Using default detector instead")
                license_plate_detector = None
        else:
            print(f"⚠ Model file not found: {MODEL_PATH}")
            print("   Using default detector instead")
            license_plate_detector = None
    except Exception as e:
        print(f"⚠ Error checking model path: {e}")
        license_plate_detector = None
    
    # Load video if it exists
    try:
        if Path(VIDEO_PATH).exists():
            try:
                video_cap = cv2.VideoCapture(VIDEO_PATH)
                if video_cap.isOpened():
                    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = video_cap.get(cv2.CAP_PROP_FPS)
                    print(f"✓ Loaded video from {VIDEO_PATH}")
                    print(f"  - Frames: {frame_count}, FPS: {fps:.2f}")
                else:
                    print(f"⚠ Warning: Could not open video file: {VIDEO_PATH}")
                    video_cap = None
            except Exception as e:
                print(f"⚠ Warning: Error loading video from {VIDEO_PATH}: {e}")
                video_cap = None
        else:
            print(f"⚠ Video file not found: {VIDEO_PATH}")
            video_cap = None
    except Exception as e:
        print(f"⚠ Error checking video path: {e}")
        video_cap = None


@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup"""
    try:
        print("Initializing License Plate Detector...")
        
        # Load test resources (model and video)
        load_test_resources()
        
        # Initialize main detector
        try:
            get_detector()
            print("Detector ready!")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize default detector: {e}")
            print("   Some endpoints may not work properly")
    except Exception as e:
        print(f"⚠ Error during startup: {e}")
        print("   Server will continue but may have limited functionality")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "license-plate-detection-api",
        "version": "1.0.0"
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """
    License plate detection endpoint
    
    Accepts either:
    - Base64 encoded image data (via 'data' field)
    - Image URL (via 'image_url' field)
    
    Returns detection results with bounding boxes and confidence scores.
    """
    detector = get_detector()
    
    result: Dict[str, Any] = {}
    image = None
    
    # Process based on input type
    if request.image_url:
        result = detector.detect_from_url(request.image_url)
        if result.get("detected"):
            # Load image for visualization if needed
            if request.include_visualization:
                image = detector.load_image_from_url(request.image_url)
    elif request.data:
        result = detector.detect_from_base64(request.data)
        if result.get("detected"):
            # Load image for visualization if needed
            if request.include_visualization:
                image = detector.load_image_from_base64(request.data)
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'data' (base64) or 'image_url' must be provided"
        )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Calculate average confidence if detections exist
    avg_confidence = None
    if result["detections"]:
        avg_confidence = sum(d["confidence"] for d in result["detections"]) / len(result["detections"])
    
    # Get visualization if requested
    visualization = None
    if request.include_visualization and image is not None and result["detections"]:
        visualization = detector.get_visualization(image, result["detections"])
    
    # Convert detections to response model
    detections = [
        Detection(
            class_name=d["class_name"],
            confidence=d["confidence"],
            bbox=BoundingBox(**d["bbox"])
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


@app.post("/detect/upload")
async def detect_upload(
    file: UploadFile = File(..., description="Image file to process"),
    include_visualization: bool = Form(False, description="Include annotated image in response")
):
    """
    License plate detection from uploaded file
    
    Accepts image files via multipart/form-data upload
    """
    detector = get_detector()
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read uploaded file
    contents = await file.read()
    
    # Convert to numpy array
    image = Image.open(io.BytesIO(contents))
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Detect license plates
    detections = detector.detect_license_plates(image_np)
    
    # Calculate average confidence
    avg_confidence = None
    if detections:
        avg_confidence = sum(d["confidence"] for d in detections) / len(detections)
    
    # Get visualization if requested
    visualization = None
    if include_visualization and detections:
        visualization = detector.get_visualization(image_np, detections)
    
    # Convert detections to response model
    detection_models = [
        Detection(
            class_name=d["class_name"],
            confidence=d["confidence"],
            bbox=BoundingBox(**d["bbox"])
        )
        for d in detections
    ]
    
    return DetectResponse(
        detected=len(detections) > 0,
        count=len(detections),
        detections=detection_models,
        message=f"Found {len(detections)} detection(s)" if detections else "No license plates detected",
        confidence=round(avg_confidence, 4) if avg_confidence else None,
        image_shape={
            "height": image_np.shape[0],
            "width": image_np.shape[1]
        },
        visualization=visualization
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    # Safely check if resources are loaded
    model_loaded = license_plate_detector is not None
    video_loaded = False
    if video_cap is not None:
        try:
            video_loaded = video_cap.isOpened()
        except:
            video_loaded = False
    
    return {
        "name": "License Plate Detection API",
        "version": "1.0.0",
        "description": "License plate detection using YOLOv11",
        "endpoints": {
            "/health": "Health check",
            "/detect": "POST - Detect license plates (base64 or URL)",
            "/detect/upload": "POST - Detect license plates (file upload)",
            "/detect/video": "GET - Process test video frame",
            "/docs": "API documentation"
        },
        "test_resources": {
            "model_loaded": model_loaded,
            "video_loaded": video_loaded,
            "model_path": MODEL_PATH,
            "video_path": VIDEO_PATH
        }
    }


@app.get("/detect/video")
async def detect_video_frame(frame_number: Optional[int] = None):
    """
    Process a frame from the test video file
    
    Args:
        frame_number: Optional frame number to process. If not provided, processes next frame.
    
    Returns:
        Detection results for the video frame
    """
    global video_cap, license_plate_detector
    
    if video_cap is None or not video_cap.isOpened():
        raise HTTPException(
            status_code=404,
            detail=f"Video file not loaded. Check if {VIDEO_PATH} exists."
        )
    
    # Use test model if available, otherwise use default detector
    detector_model = license_plate_detector if license_plate_detector is not None else get_detector().model
    
    # Set frame position if specified
    if frame_number is not None:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read frame
    ret, frame = video_cap.read()
    
    if not ret:
        # Reset to beginning if we've reached the end
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = video_cap.read()
        if not ret:
            raise HTTPException(status_code=500, detail="Could not read frame from video")
    
    # Get current frame number
    current_frame = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Run detection
    if license_plate_detector is not None:
        # Use direct YOLO model
        results = license_plate_detector(frame, conf=0.25, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = license_plate_detector.names[class_id]
                
                detections.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1)
                    }
                })
    else:
        # Use detector service
        detector = get_detector()
        detections = detector.detect_license_plates(frame)
    
    # Calculate average confidence
    avg_confidence = None
    if detections:
        avg_confidence = sum(d["confidence"] for d in detections) / len(detections)
    
    # Get visualization
    detector = get_detector()
    visualization = None
    if detections:
        visualization = detector.get_visualization(frame, detections)
    
    # Convert to response model
    detection_models = [
        Detection(
            class_name=d["class_name"],
            confidence=d["confidence"],
            bbox=BoundingBox(**d["bbox"])
        )
        for d in detections
    ]
    
    return DetectResponse(
        detected=len(detections) > 0,
        count=len(detections),
        detections=detection_models,
        message=f"Processed frame {current_frame}/{total_frames}. Found {len(detections)} detection(s)",
        confidence=round(avg_confidence, 4) if avg_confidence else None,
        image_shape={
            "height": frame.shape[0],
            "width": frame.shape[1]
        },
        visualization=visualization
    )


