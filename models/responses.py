"""
Response models for API endpoints
All Pydantic models for response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


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


class StabilityMetrics(BaseModel):
    """Stability metrics for plate detections"""
    is_stable: bool = Field(description="Whether the plate is considered stable")
    stability_score: float = Field(description="Overall stability score (0-1)")
    bbox_variance: float = Field(description="Variance in bounding box position")
    confidence_variance: float = Field(description="Variance in confidence scores")
    position_stability: float = Field(description="Position stability score (0-1)")
    confidence_stability: float = Field(description="Confidence stability score (0-1)")


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
    stability: Optional[StabilityMetrics] = Field(None, description="Stability analysis metrics")


class DeduplicationStats(BaseModel):
    """Statistics about deduplication process"""
    total_detections: int = Field(description="Total detections before deduplication")
    unique_detections: int = Field(description="Unique detections after deduplication")
    duplicate_detections: int = Field(description="Number of duplicates removed")
    deduplication_rate: float = Field(description="Percentage of duplicates removed")
    kept_strategy: str = Field(description="Strategy used to keep detections")
    config: Dict[str, Any] = Field(description="Deduplication configuration used")


class VideoProcessingStats(BaseModel):
    """Summary statistics for video processing"""
    total_frames: int
    processed_frames: int
    frames_with_detections: int
    total_detections: int = Field(description="Total detections before deduplication")
    unique_detections: Optional[int] = Field(None, description="Unique detections after deduplication")
    duplicate_detections: Optional[int] = Field(None, description="Number of duplicates removed")
    deduplication_rate: Optional[float] = Field(None, description="Percentage of duplicates removed")
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
    deduplication: Optional[DeduplicationStats] = Field(None, description="Deduplication statistics")
    plate_summaries: List[PlateSummary]
    all_detections: List[Dict[str, Any]]  # Detailed list of unique detections
    duplicate_detections: Optional[List[Dict[str, Any]]] = Field(None, description="List of duplicate detections")
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


class GeminiDetection(BaseModel):
    """Gemini object detection result"""
    label: str
    box_2d: List[int]  # [x1, y1, x2, y2]
    box_2d_normalized: List[float]  # [ymin, xmin, ymax, xmax] normalized to 0-1000
    confidence: float


class GeminiDetectResponse(BaseModel):
    """Response model for Gemini object detection endpoint"""
    detected: bool
    count: int
    detections: List[GeminiDetection]
    message: str
    image_shape: Optional[Dict[str, int]] = None
    visualization: Optional[str] = Field(None, description="Base64 encoded annotated image")


class GeminiSegmentation(BaseModel):
    """Gemini segmentation result"""
    label: str
    box_2d: List[int]  # [x1, y1, x2, y2]
    box_2d_normalized: List[float]  # [ymin, xmin, ymax, xmax] normalized to 0-1000
    confidence: float
    # Note: mask is not included in response (too large), but available via visualization


class GeminiSegmentResponse(BaseModel):
    """Response model for Gemini segmentation endpoint"""
    detected: bool
    count: int
    segmentations: List[GeminiSegmentation]
    message: str
    image_shape: Optional[Dict[str, int]] = None
    visualization: Optional[str] = Field(None, description="Base64 encoded annotated image with masks")
