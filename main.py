# Prevent OpenCV from trying to load GUI libraries (libGL.so.1)
import os
os.environ['OPENCV_DISABLE_LIBGL'] = '1'
# Set headless backend for OpenCV
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import asyncio
import uuid
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
from datetime import datetime
import json
from bson import ObjectId

# Database imports
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    print("⚠ Warning: pymongo not installed. MongoDB features will be disabled.")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠ Warning: redis not installed. Redis features will be disabled.")

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers to ensure all errors return JSON
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensure HTTPException always returns JSON"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Ensure validation errors return JSON"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "status_code": 422}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch all exceptions and return JSON"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "status_code": 500}
    )

# Initialize the detector (singleton pattern)
detector: Optional[LicensePlateDetector] = None

# Database connections
mongo_client: Optional[Any] = None
mongo_db: Optional[Any] = None
redis_client: Optional[Any] = None


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
                # Model file not found - will use default YOLO model
                print(f"⚠ Model path not found: {resolved_path}")
                print("   Will use default YOLO pretrained model")
                model_path = None  # Let LicensePlateDetector use default
        else:
            print(f"Using model from YOLO_MODEL_PATH env var: {model_path}")
        confidence = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
        
        # OCR engine configuration
        ocr_engine = os.getenv("OCR_ENGINE", "easyocr").lower()  # 'easyocr' or 'gemini'
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Create detector - model_path can be None to use default YOLO model
        detector = LicensePlateDetector(
            model_path=model_path if model_path else None, 
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


class JobStatus(BaseModel):
    """Job status model"""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class JobSubmitResponse(BaseModel):
    """Response when submitting a job"""
    job_id: str
    status: str
    message: str
    status_url: str


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


def get_mongo_client():
    """Get or create MongoDB client"""
    global mongo_client, mongo_db
    
    if not PYMONGO_AVAILABLE:
        return None, None
    
    if mongo_client is None:
        mongo_uri = os.getenv("MONGODB_URI") or os.getenv("MONGO_URL")
        if mongo_uri:
            try:
                mongo_client = MongoClient(
                    mongo_uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000
                )
                # Test connection
                mongo_client.admin.command('ping')
                db_name = os.getenv("MONGODB_DB_NAME", "detect_api")
                mongo_db = mongo_client[db_name]
                print(f"✓ Connected to MongoDB: {db_name}")
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                print(f"⚠ Warning: Could not connect to MongoDB: {e}")
                mongo_client = None
                mongo_db = None
        else:
            print("⚠ Warning: MONGODB_URI not set. MongoDB features disabled.")
    
    return mongo_client, mongo_db


def get_redis_client():
    """Get or create Redis client"""
    global redis_client
    
    if not REDIS_AVAILABLE:
        return None
    
    if redis_client is None:
        redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_URI")
        if redis_url:
            try:
                redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                # Test connection
                redis_client.ping()
                print("✓ Connected to Redis")
            except Exception as e:
                print(f"⚠ Warning: Could not connect to Redis: {e}")
                redis_client = None
        else:
            print("⚠ Warning: REDIS_URL not set. Redis features disabled.")
    
    return redis_client


def save_detection_to_mongodb(detection_data: Dict[str, Any], collection_name: str = "detections"):
    """Save detection result to MongoDB"""
    if mongo_db is None:
        return None
    
    try:
        collection = mongo_db[collection_name]
        detection_data["created_at"] = datetime.utcnow()
        result = collection.insert_one(detection_data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"⚠ Error saving to MongoDB: {e}")
        return None


def save_annotated_frame_to_mongodb(
    frame: np.ndarray, 
    detections: List[Dict[str, Any]], 
    frame_number: int,
    timestamp_seconds: float,
    job_id: Optional[str] = None,
    use_gridfs: bool = False
) -> Optional[str]:
    """
    Save annotated frame (with bounding boxes) to MongoDB
    
    Args:
        frame: Original frame as numpy array (BGR format from OpenCV)
        detections: List of detection dictionaries
        frame_number: Frame number in video
        timestamp_seconds: Timestamp in seconds
        job_id: Optional job ID for grouping frames
        use_gridfs: If True, use GridFS for storage (for large images > 16MB)
    
    Returns:
        MongoDB document ID or GridFS file ID, or None if failed
    """
    if mongo_db is None:
        return None
    
    try:
        detector = get_detector()
        
        # Generate annotated frame (base64 encoded)
        annotated_base64 = detector.get_visualization(frame, detections)
        
        # Prepare frame document
        frame_doc = {
            "frame_number": frame_number,
            "timestamp_seconds": timestamp_seconds,
            "job_id": job_id,
            "detection_count": len(detections),
            "created_at": datetime.utcnow()
        }
        
        if use_gridfs:
            # Use GridFS for large images
            from gridfs import GridFS
            fs = GridFS(mongo_db)
            
            # Decode base64 to bytes
            frame_bytes = base64.b64decode(annotated_base64)
            
            # Store in GridFS
            file_id = fs.put(
                frame_bytes,
                filename=f"frame_{frame_number}_{job_id or 'unknown'}.png",
                content_type="image/png",
                frame_number=frame_number,
                timestamp=timestamp_seconds,
                job_id=job_id
            )
            
            frame_doc["gridfs_file_id"] = str(file_id)
            frame_doc["storage_type"] = "gridfs"
            
            # Save metadata
            result = mongo_db["annotated_frames"].insert_one(frame_doc)
            return str(result.inserted_id)
        else:
            # Use base64 in document (works for images < 16MB)
            frame_doc["image_base64"] = annotated_base64
            frame_doc["storage_type"] = "base64"
            frame_doc["image_format"] = "PNG"
            
            # Save to MongoDB
            result = mongo_db["annotated_frames"].insert_one(frame_doc)
            return str(result.inserted_id)
            
    except Exception as e:
        print(f"⚠ Error saving annotated frame to MongoDB: {e}")
        return None


def get_from_redis(key: str) -> Optional[Any]:
    """Get value from Redis cache"""
    if redis_client is None:
        return None
    
    try:
        value = redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        print(f"⚠ Error reading from Redis: {e}")
        return None


def set_to_redis(key: str, value: Any, expire_seconds: Optional[int] = None):
    """Set value in Redis cache"""
    if redis_client is None:
        return False
    
    try:
        json_value = json.dumps(value) if not isinstance(value, str) else value
        if expire_seconds:
            redis_client.setex(key, expire_seconds, json_value)
        else:
            redis_client.set(key, json_value)
        return True
    except Exception as e:
        print(f"⚠ Error writing to Redis: {e}")
        return False


# Job management functions
def create_job(job_id: str, job_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new job record"""
    job_data = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "pending",
        "progress": 0.0,
        "message": "Job queued",
        "created_at": datetime.utcnow().isoformat(),
        "metadata": metadata
    }
    
    # Save to MongoDB
    if mongo_db is not None:
        try:
            mongo_db["jobs"].insert_one(job_data.copy())
        except Exception as e:
            print(f"⚠ Error saving job to MongoDB: {e}")
    
    # Save to Redis for quick access (expires in 24 hours)
    set_to_redis(f"job:{job_id}", job_data, expire_seconds=86400)
    
    return job_data


def update_job_status(job_id: str, status: str, progress: float = None, message: str = None, error: str = None, result: Dict[str, Any] = None):
    """Update job status"""
    update_data = {
        "status": status,
        "updated_at": datetime.utcnow().isoformat()
    }
    
    if progress is not None:
        update_data["progress"] = progress
    
    if message is not None:
        update_data["message"] = message
    
    if error is not None:
        update_data["error"] = error
    
    if status == "processing" and "started_at" not in update_data:
        update_data["started_at"] = datetime.utcnow().isoformat()
    
    if status in ["completed", "failed"]:
        update_data["completed_at"] = datetime.utcnow().isoformat()
        if result:
            update_data["result"] = result
    
    # Update MongoDB
    if mongo_db is not None:
        try:
            mongo_db["jobs"].update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
        except Exception as e:
            print(f"⚠ Error updating job in MongoDB: {e}")
    
    # Update Redis
    current_job = get_from_redis(f"job:{job_id}")
    if current_job:
        current_job.update(update_data)
        set_to_redis(f"job:{job_id}", current_job, expire_seconds=86400)
    
    return update_data


def get_job_status_data(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status data from Redis or MongoDB"""
    # Try Redis first (faster)
    job = get_from_redis(f"job:{job_id}")
    
    if not job and mongo_db is not None:
        # Fallback to MongoDB
        try:
            job_doc = mongo_db["jobs"].find_one({"job_id": job_id})
            if job_doc:
                job = dict(job_doc)
                # Remove MongoDB _id
                job.pop("_id", None)
                # Cache in Redis
                set_to_redis(f"job:{job_id}", job, expire_seconds=86400)
        except Exception as e:
            print(f"⚠ Error getting job from MongoDB: {e}")
    
    return job


# Maximum file size (500MB)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


def process_video_background(job_id: str, video_path: str, frame_skip: int, start_frame: Optional[int], end_frame: Optional[int], min_confidence: Optional[float]):
    """Background task to process video (runs in background thread)"""
    start_time = time.time()
    try:
        update_job_status(job_id, "processing", progress=0.0, message="Opening video file...")
        
        # Open video with OpenCV
        video_cap = cv2.VideoCapture(video_path)
        
        if not video_cap.isOpened():
            update_job_status(job_id, "failed", error="Could not open video file")
            return
        
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
            video_cap.release()
            update_job_status(job_id, "failed", error=f"Invalid frame range: {start_frame_num} to {end_frame_num}")
            return
        
        # Get detector
        detector = get_detector()
        
        # Override confidence threshold if provided
        original_threshold = detector.confidence_threshold
        if min_confidence is not None:
            detector.confidence_threshold = min_confidence
        
        try:
            # Storage for all detections
            all_detections = []
            plate_data = defaultdict(list)
            
            processed_frames = 0
            frames_with_detections = 0
            frames_to_process = len(range(start_frame_num, end_frame_num + 1, frame_skip))
            
            # Process frames
            for idx, frame_num in enumerate(range(start_frame_num, end_frame_num + 1, frame_skip)):
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
                    
                    # Save annotated frame to MongoDB
                    frame_id = save_annotated_frame_to_mongodb(
                        frame=frame,
                        detections=detections,
                        frame_number=frame_num,
                        timestamp_seconds=timestamp,
                        job_id=job_id,
                        use_gridfs=False  # Use base64 for now, can switch to GridFS if needed
                    )
                    
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
                            "class_name": det["class_name"],
                            "frame_id": frame_id  # Reference to annotated frame in MongoDB
                        }
                        
                        all_detections.append(occurrence_data)
                        plate_data[plate_text].append(occurrence_data)
                
                # Update progress every 10 frames or every 1%
                if processed_frames % 10 == 0 or (idx + 1) % max(1, frames_to_process // 100) == 0:
                    progress = (idx + 1) / frames_to_process
                    update_job_status(
                        job_id, 
                        "processing", 
                        progress=progress,
                        message=f"Processed {processed_frames}/{frames_to_process} frames ({len(all_detections)} detections)"
                    )
                    # Small sleep to prevent blocking
                    time.sleep(0.01)
            
        finally:
            detector.confidence_threshold = original_threshold
            video_cap.release()
        
        # Calculate statistics
        processing_time = time.time() - start_time
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
            
            plate_summaries.append({
                "plate_text": plate_text,
                "total_occurrences": len(occurrences),
                "first_seen_frame": min(frame_numbers),
                "last_seen_frame": max(frame_numbers),
                "first_seen_timestamp": min(occ["timestamp_seconds"] for occ in occurrences),
                "last_seen_timestamp": max(occ["timestamp_seconds"] for occ in occurrences),
                "average_confidence": sum(confidences) / len(confidences),
                "average_ocr_confidence": sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None,
                "frames_with_detection": sorted(frame_numbers),
                "occurrences": occurrences
            })
        
        # Sort by total occurrences
        plate_summaries.sort(key=lambda x: x["total_occurrences"], reverse=True)
        
        # Prepare result
        result = {
            "success": True,
            "message": f"Processed {processed_frames} frames. Found {unique_plates} unique license plate(s) with {total_detections} total detection(s).",
            "video_info": {
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "resolution": {"width": width, "height": height},
                "duration_seconds": round(video_duration, 2)
            },
            "statistics": {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "frames_with_detections": frames_with_detections,
                "total_detections": total_detections,
                "unique_plates": unique_plates,
                "video_duration_seconds": round(video_duration, 2),
                "processing_time_seconds": round(processing_time, 2),
                "average_fps": round(avg_fps, 2) if avg_fps else None,
                "detection_rate": round(detection_rate, 2)
            },
            "plate_summaries": plate_summaries,
            "all_detections": all_detections,
            "processing_parameters": {
                "frame_skip": frame_skip,
                "start_frame": start_frame_num,
                "end_frame": end_frame_num,
                "confidence_threshold": min_confidence if min_confidence is not None else detector.confidence_threshold
            }
        }
        
        # Save result to MongoDB
        if mongo_db is not None:
            try:
                mongo_db["video_results"].insert_one({
                    "job_id": job_id,
                    "result": result,
                    "created_at": datetime.utcnow()
                })
            except Exception as e:
                print(f"⚠ Error saving result to MongoDB: {e}")
        
        # Update job status
        update_job_status(job_id, "completed", progress=1.0, message="Processing completed", result=result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"⚠ Error processing video job {job_id}: {error_msg}")
        update_job_status(job_id, "failed", error=error_msg)
    finally:
        # Clean up temporary file
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except:
                pass


@app.on_event("startup")
async def startup_event():
    """Initialize detector and database connections on startup"""
    try:
        print("Initializing License Plate Detector...")
        
        # Initialize database connections
        get_mongo_client()
        get_redis_client()
        
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


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown"""
    global mongo_client, redis_client
    
    if mongo_client is not None:
        try:
            mongo_client.close()
            print("✓ MongoDB connection closed")
        except Exception as e:
            print(f"⚠ Error closing MongoDB connection: {e}")
    
    if redis_client is not None:
        try:
            redis_client.close()
            print("✓ Redis connection closed")
        except Exception as e:
            print(f"⚠ Error closing Redis connection: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": "license-plate-detection-api",
        "version": "1.0.0",
        "databases": {
            "mongodb": "connected" if mongo_client is not None else "disconnected",
            "redis": "connected" if redis_client is not None else "disconnected"
        }
    }
    
    # Test database connections
    if mongo_client is not None:
        try:
            mongo_client.admin.command('ping')
            health_status["databases"]["mongodb"] = "connected"
        except:
            health_status["databases"]["mongodb"] = "error"
    
    if redis_client is not None:
        try:
            redis_client.ping()
            health_status["databases"]["redis"] = "connected"
        except:
            health_status["databases"]["redis"] = "error"
    
    return health_status


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
            "/process/video/upload": "POST - Process uploaded video file (synchronous)",
            "/process/video/upload/async": "POST - Process uploaded video file (async, recommended for large files)",
            "/api/jobs": "GET - List all jobs with pagination and filters",
            "/api/detections": "GET - List all detections with pagination and filters",
            "/api/violations": "GET - List all violations with pagination and filters",
            "/jobs/{job_id}": "GET - Get job status and progress",
            "/jobs/{job_id}/result": "GET - Get completed job result",
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
        },
        "databases": {
            "mongodb": {
                "connected": mongo_client is not None,
                "database": os.getenv("MONGODB_DB_NAME", "detect_api") if mongo_client is not None else None
            },
            "redis": {
                "connected": redis_client is not None
            }
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
                
                # Save annotated frame to MongoDB
                frame_id = save_annotated_frame_to_mongodb(
                    frame=frame,
                    detections=detections,
                    frame_number=frame_num,
                    timestamp_seconds=timestamp,
                    job_id=None,  # No job_id for synchronous processing
                    use_gridfs=False
                )
                
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
                        "class_name": det["class_name"],
                        "frame_id": frame_id  # Reference to annotated frame in MongoDB
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


@app.post("/process/video/upload", response_model=VideoProcessResponse)
async def process_video_upload(
    file: UploadFile = File(..., description="Video file to process"),
    frame_skip: int = Form(1, description="Process every Nth frame (1 = all frames)"),
    start_frame: Optional[int] = Form(None, description="Start processing from this frame"),
    end_frame: Optional[int] = Form(None, description="Stop processing at this frame"),
    min_confidence: Optional[float] = Form(None, description="Minimum confidence threshold")
):
    """
    Process an uploaded video file and return summary statistics (SYNCHRONOUS)
    
    ⚠️ WARNING: For large files, use /process/video/upload/async instead to avoid timeouts.
    
    Accepts video files via multipart/form-data upload.
    The video is processed synchronously - client waits for completion.
    
    Args:
        file: Video file to process (mp4, avi, mov, etc.) - Max 500MB
        frame_skip: Process every Nth frame (1 = all frames, 2 = every 2nd frame, etc.). Default: 1
        start_frame: Start processing from this frame (optional)
        end_frame: Stop processing at this frame (optional)
        min_confidence: Minimum confidence threshold for detections (optional, overrides default)
    
    Returns:
        Comprehensive statistics and all detections from the video
    """
    import tempfile
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        # Also check filename extension as fallback
        filename = file.filename or ""
        if not any(filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']):
            raise HTTPException(
                status_code=400, 
                detail="File must be a video (mp4, avi, mov, mkv, webm, flv)"
            )
    
    # Check file size (read first chunk to estimate)
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB. Your file is {file_size / (1024*1024):.2f}MB. Use /process/video/upload/async for large files."
        )
    
    # Reset file pointer
    await file.seek(0)
    
    # Save uploaded file temporarily
    temp_path = None
    try:
        # Read video content
        contents = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        
        # Open video with OpenCV
        video_cap = cv2.VideoCapture(temp_path)
        
        if not video_cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open video file. Please ensure it's a valid video format."
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
            video_cap.release()
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
                    
                    # Save annotated frame to MongoDB
                    frame_id = save_annotated_frame_to_mongodb(
                        frame=frame,
                        detections=detections,
                        frame_number=frame_num,
                        timestamp_seconds=timestamp,
                        job_id=None,  # No job_id for synchronous processing
                        use_gridfs=False
                    )
                    
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
                            "class_name": det["class_name"],
                            "frame_id": frame_id  # Reference to annotated frame in MongoDB
                        }
                        
                        all_detections.append(occurrence_data)
                        plate_data[plate_text].append(occurrence_data)
                
                # Print progress every 50 frames
                if processed_frames % 50 == 0:
                    print(f"Processed {processed_frames} frames... ({len(all_detections)} detections so far)")
            
        finally:
            # Restore original confidence threshold
            detector.confidence_threshold = original_threshold
            # Release video capture
            video_cap.release()
        
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
            "path": file.filename or "uploaded_video",
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
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass


@app.post("/process/video/upload/async", response_model=JobSubmitResponse)
async def process_video_upload_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to process"),
    frame_skip: int = Form(1, description="Process every Nth frame (1 = all frames)"),
    start_frame: Optional[int] = Form(None, description="Start processing from this frame"),
    end_frame: Optional[int] = Form(None, description="Stop processing at this frame"),
    min_confidence: Optional[float] = Form(None, description="Minimum confidence threshold")
):
    """
    Process an uploaded video file asynchronously (RECOMMENDED for large files)
    
    This endpoint accepts a video file and returns immediately with a job_id.
    The video is processed in the background. Use the job_id to check status and retrieve results.
    
    Args:
        file: Video file to process (mp4, avi, mov, etc.) - Max 500MB
        frame_skip: Process every Nth frame (1 = all frames, 2 = every 2nd frame, etc.). Default: 1
        start_frame: Start processing from this frame (optional)
        end_frame: Stop processing at this frame (optional)
        min_confidence: Minimum confidence threshold for detections (optional, overrides default)
    
    Returns:
        Job submission response with job_id and status URL
    """
    import tempfile
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        filename = file.filename or ""
        if not any(filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']):
            raise HTTPException(
                status_code=400, 
                detail="File must be a video (mp4, avi, mov, mkv, webm, flv)"
            )
    
    # Read and check file size
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=413,
            detail=f"Error reading file. File may be too large or corrupted: {str(e)}"
        )
    
    file_size = len(contents)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB. Your file is {file_size / (1024*1024):.2f}MB."
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )
    
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        
        # Create job record
        metadata = {
            "filename": file.filename,
            "file_size": file_size,
            "frame_skip": frame_skip,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "min_confidence": min_confidence
        }
        
        try:
            create_job(job_id, "video_processing", metadata)
        except Exception as job_error:
            print(f"Error in create_job: {job_error}")
            import traceback
            traceback.print_exc()
            raise
        
        # Start background task
        background_tasks.add_task(
            process_video_background,
            job_id,
            temp_path,
            frame_skip,
            start_frame,
            end_frame,
            min_confidence
        )
        
        return JobSubmitResponse(
            job_id=job_id,
            status="pending",
            message="Video processing job created. Use the job_id to check status.",
            status_url=f"/jobs/{job_id}"
        )
    except HTTPException:
        # Re-raise HTTPExceptions (they're already properly formatted by exception handler)
        raise
    except Exception as e:
        # Clean up temp file if job creation failed
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        # Ensure all errors return JSON
        raise HTTPException(
            status_code=500,
            detail=f"Error creating job: {str(e)}"
        )


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status_endpoint(job_id: str):
    """
    Get the status of a processing job
    
    Args:
        job_id: The job ID returned from /process/video/upload/async
    
    Returns:
        Job status with progress information
    """
    job = get_job_status_data(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress", 0.0),
        message=job.get("message", ""),
        created_at=datetime.fromisoformat(job["created_at"]),
        started_at=datetime.fromisoformat(job["started_at"]) if job.get("started_at") else None,
        completed_at=datetime.fromisoformat(job["completed_at"]) if job.get("completed_at") else None,
        error=job.get("error"),
        result=job.get("result")
    )


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get the result of a completed job
    
    Args:
        job_id: The job ID returned from /process/video/upload/async
    
    Returns:
        Complete processing result (same format as synchronous endpoint)
    """
    job = get_job_status_data(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    if job["status"] == "pending" or job["status"] == "processing":
        raise HTTPException(
            status_code=202,
            detail=f"Job is still {job['status']}. Progress: {job.get('progress', 0.0)*100:.1f}%"
        )
    
    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Job failed: {job.get('error', 'Unknown error')}"
        )
    
    if job["status"] == "completed":
        result = job.get("result")
        if result:
            return result
        else:
            # Try to get from MongoDB
            if mongo_db is not None:
                try:
                    result_doc = mongo_db["video_results"].find_one({"job_id": job_id})
                    if result_doc:
                        return result_doc["result"]
                except Exception as e:
                    print(f"⚠ Error getting result from MongoDB: {e}")
            
            raise HTTPException(
                status_code=404,
                detail="Result not found"
            )
    
    raise HTTPException(
        status_code=500,
        detail=f"Unknown job status: {job['status']}"
    )


@app.get("/frames/{frame_id}")
async def get_annotated_frame(frame_id: str):
    """
    Get an annotated frame from MongoDB by frame ID
    
    Args:
        frame_id: MongoDB document ID of the annotated frame
    
    Returns:
        Annotated frame image (base64 encoded) or GridFS file
    """
    if mongo_db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB not available"
        )
    
    try:
        from bson import ObjectId
        
        # Try to find frame document
        frame_doc = mongo_db["annotated_frames"].find_one({"_id": ObjectId(frame_id)})
        
        if not frame_doc:
            raise HTTPException(
                status_code=404,
                detail=f"Frame {frame_id} not found"
            )
        
        storage_type = frame_doc.get("storage_type", "base64")
        
        if storage_type == "gridfs":
            # Retrieve from GridFS
            from gridfs import GridFS
            fs = GridFS(mongo_db)
            
            gridfs_id = frame_doc.get("gridfs_file_id")
            if not gridfs_id:
                raise HTTPException(
                    status_code=500,
                    detail="GridFS file ID not found in frame document"
                )
            
            gridfs_file = fs.get(ObjectId(gridfs_id))
            frame_bytes = gridfs_file.read()
            frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
            
            return {
                "frame_id": frame_id,
                "frame_number": frame_doc.get("frame_number"),
                "timestamp_seconds": frame_doc.get("timestamp_seconds"),
                "job_id": frame_doc.get("job_id"),
                "detection_count": frame_doc.get("detection_count"),
                "storage_type": "gridfs",
                "image_base64": frame_base64,
                "image_format": "PNG"
            }
        else:
            # Return base64 from document
            frame_base64 = frame_doc.get("image_base64")
            if not frame_base64:
                raise HTTPException(
                    status_code=500,
                    detail="Image data not found in frame document"
                )
            
            return {
                "frame_id": frame_id,
                "frame_number": frame_doc.get("frame_number"),
                "timestamp_seconds": frame_doc.get("timestamp_seconds"),
                "job_id": frame_doc.get("job_id"),
                "detection_count": frame_doc.get("detection_count"),
                "storage_type": "base64",
                "image_base64": frame_base64,
                "image_format": frame_doc.get("image_format", "PNG")
            }
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving frame: {str(e)}"
        )


@app.get("/frames/job/{job_id}")
async def get_job_frames(job_id: str, limit: int = 100, skip: int = 0):
    """
    Get all annotated frames for a specific job
    
    Args:
        job_id: Job ID
        limit: Maximum number of frames to return (default: 100)
        skip: Number of frames to skip (for pagination)
    
    Returns:
        List of frame metadata (without image data for performance)
    """
    if mongo_db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB not available"
        )
    
    try:
        frames = list(mongo_db["annotated_frames"].find(
            {"job_id": job_id}
        ).sort("frame_number", 1).skip(skip).limit(limit))
        
        # Remove image data for list view (can fetch individual frames with /frames/{frame_id})
        for frame in frames:
            frame["_id"] = str(frame["_id"])
            if "image_base64" in frame:
                del frame["image_base64"]
            if "gridfs_file_id" in frame:
                frame["gridfs_file_id"] = str(frame["gridfs_file_id"])
        
        total = mongo_db["annotated_frames"].count_documents({"job_id": job_id})
        
        return {
            "job_id": job_id,
            "total_frames": total,
            "returned": len(frames),
            "skip": skip,
            "limit": limit,
            "frames": frames
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving frames: {str(e)}"
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


class DetectionListItem(BaseModel):
    """Detection item for list endpoint"""
    id: str
    plate: str
    confidence: float
    timestamp: str
    location: Optional[str] = None
    camera: Optional[str] = None
    image: Optional[str] = None
    violation: Optional[str] = None


class ViolationListItem(BaseModel):
    """Violation item for list endpoint"""
    id: str
    plate: str
    type: str
    location: Optional[str] = None
    timestamp: str
    speed: Optional[str] = None
    duration: Optional[str] = None
    status: str
    image: Optional[str] = None


class JobListItem(BaseModel):
    """Job item for list endpoint"""
    id: str
    job_id: str
    job_type: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    video_name: Optional[str] = None
    detections: Optional[int] = None
    violations: Optional[int] = None


@app.get("/api/detections", response_model=List[DetectionListItem])
async def get_detections(
    limit: int = 100,
    skip: int = 0,
    plate: Optional[str] = None,
    camera: Optional[str] = None
):
    """
    Get list of detections from MongoDB
    
    Args:
        limit: Maximum number of detections to return (default: 100)
        skip: Number of detections to skip (for pagination)
        plate: Filter by license plate number (optional)
        camera: Filter by camera ID (optional)
    
    Returns:
        List of detection items
    """
    if mongo_db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB not available"
        )
    
    try:
        collection = mongo_db["detections"]
        
        # Build query
        query = {}
        if plate:
            query["plate_text"] = {"$regex": plate, "$options": "i"}  # Case-insensitive search
        if camera:
            query["camera"] = camera
        
        # Query MongoDB
        cursor = collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
        detections = list(cursor)
        
        # Transform to response format
        result = []
        for doc in detections:
            # Convert MongoDB document to response format
            detection_item = DetectionListItem(
                id=str(doc.get("_id", "")),
                plate=doc.get("plate_text", doc.get("plate", "")),
                confidence=doc.get("confidence", 0.0),
                timestamp=doc.get("created_at", doc.get("timestamp", datetime.utcnow())).isoformat() if isinstance(doc.get("created_at", doc.get("timestamp")), datetime) else str(doc.get("created_at", doc.get("timestamp", datetime.utcnow().isoformat()))),
                location=doc.get("location"),
                camera=doc.get("camera"),
                image=doc.get("image", "/generic-license-plate.png"),  # Default image or from doc
                violation=doc.get("violation")
            )
            result.append(detection_item)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving detections: {str(e)}"
        )


@app.get("/api/violations", response_model=List[ViolationListItem])
async def get_violations(
    limit: int = 100,
    skip: int = 0,
    plate: Optional[str] = None,
    status: Optional[str] = None,
    violation_type: Optional[str] = None
):
    """
    Get list of violations from MongoDB
    
    Args:
        limit: Maximum number of violations to return (default: 100)
        skip: Number of violations to skip (for pagination)
        plate: Filter by license plate number (optional)
        status: Filter by violation status (pending, reviewed, etc.) (optional)
        violation_type: Filter by violation type (optional)
    
    Returns:
        List of violation items
    """
    if mongo_db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB not available"
        )
    
    try:
        collection = mongo_db["violations"]
        
        # Build query
        query = {}
        if plate:
            query["plate"] = {"$regex": plate, "$options": "i"}  # Case-insensitive search
        if status:
            query["status"] = status
        if violation_type:
            query["type"] = {"$regex": violation_type, "$options": "i"}
        
        # Query MongoDB
        cursor = collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        violations = list(cursor)
        
        # Transform to response format
        result = []
        for doc in violations:
            # Convert MongoDB document to response format
            violation_item = ViolationListItem(
                id=str(doc.get("_id", "")),
                plate=doc.get("plate", ""),
                type=doc.get("type", doc.get("violation_type", "Unknown")),
                location=doc.get("location"),
                timestamp=doc.get("timestamp", doc.get("created_at", datetime.utcnow())).isoformat() if isinstance(doc.get("timestamp", doc.get("created_at")), datetime) else str(doc.get("timestamp", doc.get("created_at", datetime.utcnow().isoformat()))),
                speed=doc.get("speed"),
                duration=doc.get("duration"),
                status=doc.get("status", "pending"),
                image=doc.get("image", "/generic-license-plate.png")  # Default image or from doc
            )
            result.append(violation_item)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving violations: {str(e)}"
        )


@app.get("/api/jobs", response_model=List[JobListItem])
async def get_jobs(
    limit: int = 100,
    skip: int = 0,
    status: Optional[str] = None,
    job_type: Optional[str] = None
):
    """
    Get list of jobs from MongoDB
    
    Args:
        limit: Maximum number of jobs to return (default: 100)
        skip: Number of jobs to skip (for pagination)
        status: Filter by job status (pending, processing, completed, failed) (optional)
        job_type: Filter by job type (optional)
    
    Returns:
        List of job items
    """
    if mongo_db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB not available"
        )
    
    try:
        collection = mongo_db["jobs"]
        
        # Build query
        query = {}
        if status:
            query["status"] = status
        if job_type:
            query["job_type"] = job_type
        
        # Query MongoDB - sort by created_at descending (newest first)
        cursor = collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
        jobs = list(cursor)
        
        # Transform to response format
        result = []
        for doc in jobs:
            # Extract video name from metadata if available
            metadata = doc.get("metadata", {})
            video_name = metadata.get("filename") if isinstance(metadata, dict) else None
            
            # Extract detections and violations count from result if available
            result_data = doc.get("result", {})
            detections_count = None
            violations_count = None
            
            if isinstance(result_data, dict):
                # Try to get from statistics
                stats = result_data.get("statistics", {})
                if stats:
                    detections_count = stats.get("total_detections")
                
                # Try to get violations count (if stored in result)
                # This would need to be calculated or stored separately
                # For now, we'll leave it as None unless it's in the result
            
            # Convert timestamps
            created_at = doc.get("created_at")
            if isinstance(created_at, datetime):
                created_at_str = created_at.isoformat()
            elif isinstance(created_at, str):
                created_at_str = created_at
            else:
                created_at_str = datetime.utcnow().isoformat()
            
            started_at = doc.get("started_at")
            started_at_str = None
            if started_at:
                if isinstance(started_at, datetime):
                    started_at_str = started_at.isoformat()
                elif isinstance(started_at, str):
                    started_at_str = started_at
            
            completed_at = doc.get("completed_at")
            completed_at_str = None
            if completed_at:
                if isinstance(completed_at, datetime):
                    completed_at_str = completed_at.isoformat()
                elif isinstance(completed_at, str):
                    completed_at_str = completed_at
            
            job_item = JobListItem(
                id=str(doc.get("_id", "")),
                job_id=doc.get("job_id", ""),
                job_type=doc.get("job_type", "unknown"),
                status=doc.get("status", "pending"),
                progress=doc.get("progress", 0.0),
                message=doc.get("message", ""),
                created_at=created_at_str,
                started_at=started_at_str,
                completed_at=completed_at_str,
                video_name=video_name,
                detections=detections_count,
                violations=violations_count
            )
            result.append(job_item)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving jobs: {str(e)}"
        )


