# Prevent OpenCV from trying to load GUI libraries (libGL.so.1)
import os
os.environ['OPENCV_DISABLE_LIBGL'] = '1'
# Set headless backend for OpenCV
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
from ultralytics import YOLO
from detection_service import LicensePlateDetector, GeminiImageService
import cv2
import numpy as np
from PIL import Image
import io
import time
import base64
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# File paths for testing
MODEL_PATH = "./models/license_plate_detector.pt"
VIDEO_PATH = "./files/2.mp4"  # Using 2.mp4 for testing

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
        # Check for model path in this order: environment variable, then MODEL_PATH constant
        model_path = os.getenv("YOLO_MODEL_PATH")
        if not model_path:
            # Resolve relative path to absolute
            resolved_path = Path(MODEL_PATH).resolve()
            if resolved_path.exists():
                model_path = str(resolved_path)
                print(f"Using model from MODEL_PATH constant: {model_path}")
            else:
                print(f"Model path not found: {resolved_path}")
        else:
            print(f"Using model from YOLO_MODEL_PATH env var: {model_path}")
        confidence = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
        
        # OCR engine configuration
        ocr_engine = os.getenv("OCR_ENGINE", "easyocr").lower()  # 'easyocr' or 'gemini'
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        detector = LicensePlateDetector(
            model_path=model_path, 
            confidence_threshold=confidence,
            ocr_engine=ocr_engine,
            gemini_api_key=gemini_api_key
        )
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
    plate_text: Optional[str] = Field(None, description="Extracted license plate text")
    ocr_confidence: Optional[float] = Field(None, description="OCR confidence score")


class DetectResponse(BaseModel):
    """Response model for detection endpoint"""
    detected: bool
    count: int
    detections: List[Detection]
    message: str
    confidence: Optional[float] = None
    image_shape: Optional[Dict[str, int]] = None
    visualization: Optional[str] = Field(None, description="Base64 encoded annotated image")


class PlateOccurrence(BaseModel):
    """Occurrence of a license plate in a frame"""
    frame_number: int
    timestamp_seconds: float
    confidence: float
    ocr_confidence: Optional[float] = None
    bbox: BoundingBox


class PlateSummary(BaseModel):
    """Summary statistics for a detected license plate"""
    plate_text: str
    total_occurrences: int
    first_seen_frame: int
    last_seen_frame: int
    first_seen_timestamp: float
    last_seen_timestamp: float
    average_confidence: float
    average_ocr_confidence: Optional[float] = None
    frames_with_detection: List[int]
    occurrences: List[PlateOccurrence]


class VideoProcessingStats(BaseModel):
    """Summary statistics for video processing"""
    total_frames: int
    processed_frames: int
    frames_with_detections: int
    total_detections: int
    unique_plates: int
    video_duration_seconds: float
    processing_time_seconds: float
    average_fps: Optional[float] = None
    detection_rate: float  # Detections per second


class VideoProcessResponse(BaseModel):
    """Response model for video processing endpoint"""
    success: bool
    message: str
    video_info: Dict[str, Any]
    statistics: VideoProcessingStats
    plate_summaries: List[PlateSummary]
    all_detections: List[Dict[str, Any]]  # Detailed list of all detections
    processing_parameters: Dict[str, Any]


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
            "/process/video": "POST - Process entire video with statistics",
            "/gemini/detect": "POST - Gemini object detection (2.0+)",
            "/gemini/segment": "POST - Gemini segmentation (2.5+)",
            "/gemini/detect/upload": "POST - Gemini detection from file upload",
            "/gemini/segment/upload": "POST - Gemini segmentation from file upload",
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
    
    # Use detector service (which includes OCR functionality)
    detector = get_detector()
    detections = detector.detect_license_plates(frame)
    
    # If we have a direct model and want to use it, we still need OCR from detector
    # So we'll always use the detector service for OCR support
    
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
        message=f"Processed frame {current_frame}/{total_frames}. Found {len(detections)} detection(s)",
        confidence=round(avg_confidence, 4) if avg_confidence else None,
        image_shape={
            "height": frame.shape[0],
            "width": frame.shape[1]
        },
        visualization=visualization
    )


@app.post("/process/video", response_model=VideoProcessResponse)
async def process_entire_video(
    frame_skip: int = 1,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    min_confidence: Optional[float] = None
):
    """
    Process the entire test video and return summary statistics
    
    Args:
        frame_skip: Process every Nth frame (1 = all frames, 2 = every 2nd frame, etc.). Default: 1
        start_frame: Start processing from this frame (optional)
        end_frame: Stop processing at this frame (optional)
        min_confidence: Minimum confidence threshold for detections (optional, overrides default)
    
    Returns:
        Comprehensive statistics and all detections from the video
    """
    global video_cap
    
    if video_cap is None or not video_cap.isOpened():
        raise HTTPException(
            status_code=404,
            detail=f"Video file not loaded. Check if {VIDEO_PATH} exists."
        )
    
    start_time = time.time()
    
    # Get video properties
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / fps if fps > 0 else 0
    
    # Determine frame range
    start_frame_num = start_frame if start_frame is not None else 0
    end_frame_num = end_frame if end_frame is not None else total_frames - 1
    
    if start_frame_num < 0 or end_frame_num >= total_frames or start_frame_num > end_frame_num:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid frame range: {start_frame_num} to {end_frame_num} (total frames: {total_frames})"
        )
    
    # Get detector
    detector = get_detector()
    
    # Override confidence threshold if provided
    original_threshold = detector.confidence_threshold
    if min_confidence is not None:
        detector.confidence_threshold = min_confidence
    
    try:
        # Storage for all detections
        all_detections = []
        plate_data = defaultdict(list)  # plate_text -> list of occurrences
        
        processed_frames = 0
        frames_with_detections = 0
        
        # Process frames
        for frame_num in range(start_frame_num, end_frame_num + 1, frame_skip):
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = video_cap.read()
            
            if not ret:
                break
            
            # Detect license plates in this frame
            detections = detector.detect_license_plates(frame)
            
            processed_frames += 1
            
            if detections:
                frames_with_detections += 1
                timestamp = frame_num / fps if fps > 0 else 0
                
                for det in detections:
                    plate_text = det.get("plate_text", "").strip()
                    if not plate_text:
                        plate_text = f"UNKNOWN_{len(all_detections)}"
                    
                    occurrence_data = {
                        "frame_number": frame_num,
                        "timestamp_seconds": timestamp,
                        "confidence": det["confidence"],
                        "ocr_confidence": det.get("ocr_confidence"),
                        "bbox": det["bbox"],
                        "plate_text": plate_text,
                        "class_name": det["class_name"]
                    }
                    
                    all_detections.append(occurrence_data)
                    plate_data[plate_text].append(occurrence_data)
            
            # Print progress every 50 frames
            if processed_frames % 50 == 0:
                print(f"Processed {processed_frames} frames... ({len(all_detections)} detections so far)")
        
    finally:
        # Restore original confidence threshold
        detector.confidence_threshold = original_threshold
    
    processing_time = time.time() - start_time
    
    # Calculate statistics
    total_detections = len(all_detections)
    unique_plates = len(plate_data)
    detection_rate = total_detections / video_duration if video_duration > 0 else 0
    avg_fps = processed_frames / processing_time if processing_time > 0 else None
    
    # Generate plate summaries
    plate_summaries = []
    for plate_text, occurrences in plate_data.items():
        confidences = [occ["confidence"] for occ in occurrences]
        ocr_confidences = [occ["ocr_confidence"] for occ in occurrences if occ["ocr_confidence"] is not None]
        frame_numbers = [occ["frame_number"] for occ in occurrences]
        
        plate_summaries.append(PlateSummary(
            plate_text=plate_text,
            total_occurrences=len(occurrences),
            first_seen_frame=min(frame_numbers),
            last_seen_frame=max(frame_numbers),
            first_seen_timestamp=min(occ["timestamp_seconds"] for occ in occurrences),
            last_seen_timestamp=max(occ["timestamp_seconds"] for occ in occurrences),
            average_confidence=sum(confidences) / len(confidences),
            average_ocr_confidence=sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None,
            frames_with_detection=sorted(frame_numbers),
            occurrences=[
                PlateOccurrence(
                    frame_number=occ["frame_number"],
                    timestamp_seconds=occ["timestamp_seconds"],
                    confidence=occ["confidence"],
                    ocr_confidence=occ["ocr_confidence"],
                    bbox=BoundingBox(**occ["bbox"])
                )
                for occ in occurrences
            ]
        ))
    
    # Sort plate summaries by total occurrences (most frequent first)
    plate_summaries.sort(key=lambda x: x.total_occurrences, reverse=True)
    
    # Create statistics
    stats = VideoProcessingStats(
        total_frames=total_frames,
        processed_frames=processed_frames,
        frames_with_detections=frames_with_detections,
        total_detections=total_detections,
        unique_plates=unique_plates,
        video_duration_seconds=round(video_duration, 2),
        processing_time_seconds=round(processing_time, 2),
        average_fps=round(avg_fps, 2) if avg_fps else None,
        detection_rate=round(detection_rate, 2)
    )
    
    # Prepare response
    video_info = {
        "path": VIDEO_PATH,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "resolution": {
            "width": width,
            "height": height
        },
        "duration_seconds": round(video_duration, 2)
    }
    
    processing_params = {
        "frame_skip": frame_skip,
        "start_frame": start_frame_num,
        "end_frame": end_frame_num,
        "confidence_threshold": min_confidence if min_confidence is not None else detector.confidence_threshold
    }
    
    return VideoProcessResponse(
        success=True,
        message=f"Processed {processed_frames} frames. Found {unique_plates} unique license plate(s) with {total_detections} total detection(s).",
        video_info=video_info,
        statistics=stats,
        plate_summaries=plate_summaries,
        all_detections=all_detections,
        processing_parameters=processing_params
    )


# Global Gemini service instance
gemini_service: Optional[GeminiImageService] = None


def get_gemini_service() -> GeminiImageService:
    """Get or create the Gemini service instance"""
    global gemini_service
    if gemini_service is None:
        try:
            gemini_service = GeminiImageService()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Could not initialize Gemini service: {str(e)}"
            )
    return gemini_service


class GeminiDetectRequest(BaseModel):
    """Request model for Gemini detection endpoint"""
    data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL of the image to process")
    prompt: Optional[str] = Field(None, description="Custom detection prompt")
    include_visualization: bool = Field(False, description="Include annotated image in response")


class GeminiSegmentRequest(BaseModel):
    """Request model for Gemini segmentation endpoint"""
    data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL of the image to process")
    prompt: Optional[str] = Field(None, description="Custom segmentation prompt")
    include_visualization: bool = Field(False, description="Include annotated image with masks")
    alpha: float = Field(0.5, description="Transparency for mask overlay (0.0 to 1.0)")


class GeminiDetectionItem(BaseModel):
    """Single Gemini detection result"""
    label: str
    box_2d: List[int] = Field(..., description="Bounding box as [x1, y1, x2, y2]")
    box_2d_normalized: List[float] = Field(..., description="Normalized bounding box [ymin, xmin, ymax, xmax] 0-1000")
    confidence: float


class GeminiDetectResponse(BaseModel):
    """Response model for Gemini detection endpoint"""
    detected: bool
    count: int
    detections: List[GeminiDetectionItem]
    message: str
    image_shape: Optional[Dict[str, int]] = None
    visualization: Optional[str] = Field(None, description="Base64 encoded annotated image")


class GeminiSegmentationItem(BaseModel):
    """Single Gemini segmentation result"""
    label: str
    box_2d: List[int] = Field(..., description="Bounding box as [x1, y1, x2, y2]")
    box_2d_normalized: List[float] = Field(..., description="Normalized bounding box [ymin, xmin, ymax, xmax] 0-1000")
    confidence: float
    mask_base64: Optional[str] = Field(None, description="Base64 encoded mask image")


class GeminiSegmentResponse(BaseModel):
    """Response model for Gemini segmentation endpoint"""
    detected: bool
    count: int
    segmentations: List[GeminiSegmentationItem]
    message: str
    image_shape: Optional[Dict[str, int]] = None
    visualization: Optional[str] = Field(None, description="Base64 encoded image with mask overlays")


@app.post("/gemini/detect", response_model=GeminiDetectResponse)
async def gemini_detect(request: GeminiDetectRequest):
    """
    Object detection using Gemini 2.0+ enhanced detection capabilities
    
    Detects objects in images with bounding box coordinates.
    Supports custom prompts for detecting specific object types.
    """
    service = get_gemini_service()
    
    # Load image
    image = None
    if request.image_url:
        image = service.load_image_from_url(request.image_url)
    elif request.data:
        image = service.load_image_from_base64(request.data)
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'data' (base64) or 'image_url' must be provided"
        )
    
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to load image")
    
    # Default prompt for license plates or use custom prompt
    prompt = request.prompt or "Detect all license plates in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
    
    # Detect objects
    detections = service.detect_objects(image, prompt)
    
    # Get visualization if requested
    visualization = None
    if request.include_visualization and detections:
        annotated_image = service.visualize_detections(image, detections)
        
        # Convert to base64
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        visualization = base64.b64encode(img_bytes).decode('utf-8')
    
    # Convert to response model
    detection_items = [
        GeminiDetectionItem(
            label=d["label"],
            box_2d=d["box_2d"],
            box_2d_normalized=d["box_2d_normalized"],
            confidence=d.get("confidence", 1.0)
        )
        for d in detections
    ]
    
    return GeminiDetectResponse(
        detected=len(detections) > 0,
        count=len(detections),
        detections=detection_items,
        message=f"Found {len(detections)} detection(s)" if detections else "No objects detected",
        image_shape={
            "width": image.size[0],
            "height": image.size[1]
        },
        visualization=visualization
    )


@app.post("/gemini/segment", response_model=GeminiSegmentResponse)
async def gemini_segment(request: GeminiSegmentRequest):
    """
    Object segmentation using Gemini 2.5+ segmentation capabilities
    
    Segments objects in images with pixel-level masks.
    Supports custom prompts for segmenting specific object types.
    """
    service = get_gemini_service()
    
    # Load image
    image = None
    if request.image_url:
        image = service.load_image_from_url(request.image_url)
    elif request.data:
        image = service.load_image_from_base64(request.data)
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'data' (base64) or 'image_url' must be provided"
        )
    
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to load image")
    
    # Default prompt for license plates or use custom prompt
    prompt = request.prompt or "Give the segmentation masks for all license plates in the image. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels."
    
    # Segment objects
    segmentations = service.segment_objects(image, prompt)
    
    # Get visualization if requested
    visualization = None
    if request.include_visualization and segmentations:
        annotated_image = service.visualize_segmentations(image, segmentations, alpha=request.alpha)
        
        # Convert to base64
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        visualization = base64.b64encode(img_bytes).decode('utf-8')
    
    # Convert masks to base64 for response
    import base64 as b64
    segmentation_items = []
    for seg in segmentations:
        # Convert mask array to base64 image
        mask_base64 = None
        if "mask_image" in seg:
            mask_buffer = io.BytesIO()
            seg["mask_image"].save(mask_buffer, format="PNG")
            mask_bytes = mask_buffer.getvalue()
            mask_base64 = b64.b64encode(mask_bytes).decode('utf-8')
        
        segmentation_items.append(
            GeminiSegmentationItem(
                label=seg["label"],
                box_2d=seg["box_2d"],
                box_2d_normalized=seg["box_2d_normalized"],
                confidence=seg.get("confidence", 1.0),
                mask_base64=mask_base64
            )
        )
    
    return GeminiSegmentResponse(
        detected=len(segmentations) > 0,
        count=len(segmentations),
        segmentations=segmentation_items,
        message=f"Found {len(segmentations)} segmentation(s)" if segmentations else "No objects segmented",
        image_shape={
            "width": image.size[0],
            "height": image.size[1]
        },
        visualization=visualization
    )


@app.post("/gemini/detect/upload")
async def gemini_detect_upload(
    file: UploadFile = File(..., description="Image file to process"),
    prompt: Optional[str] = Form(None, description="Custom detection prompt"),
    include_visualization: bool = Form(False, description="Include annotated image in response")
):
    """
    Object detection from uploaded file using Gemini
    """
    service = get_gemini_service()
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Default prompt or use custom
    detection_prompt = prompt or "Detect all license plates in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
    
    # Detect objects
    detections = service.detect_objects(image, detection_prompt)
    
    # Get visualization if requested
    visualization = None
    if include_visualization and detections:
        annotated_image = service.visualize_detections(image, detections)
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        visualization = base64.b64encode(img_bytes).decode('utf-8')
    
    # Convert to response model
    detection_items = [
        GeminiDetectionItem(
            label=d["label"],
            box_2d=d["box_2d"],
            box_2d_normalized=d["box_2d_normalized"],
            confidence=d.get("confidence", 1.0)
        )
        for d in detections
    ]
    
    return GeminiDetectResponse(
        detected=len(detections) > 0,
        count=len(detections),
        detections=detection_items,
        message=f"Found {len(detections)} detection(s)" if detections else "No objects detected",
        image_shape={
            "width": image.size[0],
            "height": image.size[1]
        },
        visualization=visualization
    )


@app.post("/gemini/segment/upload")
async def gemini_segment_upload(
    file: UploadFile = File(..., description="Image file to process"),
    prompt: Optional[str] = Form(None, description="Custom segmentation prompt"),
    include_visualization: bool = Form(False, description="Include annotated image with masks"),
    alpha: float = Form(0.5, description="Transparency for mask overlay (0.0 to 1.0)")
):
    """
    Object segmentation from uploaded file using Gemini
    """
    service = get_gemini_service()
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Default prompt or use custom
    segmentation_prompt = prompt or "Give the segmentation masks for all license plates in the image. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels."
    
    # Segment objects
    segmentations = service.segment_objects(image, segmentation_prompt)
    
    # Get visualization if requested
    visualization = None
    if include_visualization and segmentations:
        annotated_image = service.visualize_segmentations(image, segmentations, alpha=alpha)
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        visualization = base64.b64encode(img_bytes).decode('utf-8')
    
    # Convert masks to base64 for response
    import base64 as b64
    segmentation_items = []
    for seg in segmentations:
        mask_base64 = None
        if "mask_image" in seg:
            mask_buffer = io.BytesIO()
            seg["mask_image"].save(mask_buffer, format="PNG")
            mask_bytes = mask_buffer.getvalue()
            mask_base64 = b64.b64encode(mask_bytes).decode('utf-8')
        
        segmentation_items.append(
            GeminiSegmentationItem(
                label=seg["label"],
                box_2d=seg["box_2d"],
                box_2d_normalized=seg["box_2d_normalized"],
                confidence=seg.get("confidence", 1.0),
                mask_base64=mask_base64
            )
        )
    
    return GeminiSegmentResponse(
        detected=len(segmentations) > 0,
        count=len(segmentations),
        segmentations=segmentation_items,
        message=f"Found {len(segmentations)} segmentation(s)" if segmentations else "No objects segmented",
        image_shape={
            "width": image.size[0],
            "height": image.size[1]
        },
        visualization=visualization
    )


