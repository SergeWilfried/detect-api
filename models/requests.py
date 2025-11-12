"""
Request models for API endpoints
All Pydantic models for request validation
"""
from pydantic import BaseModel, Field
from typing import Optional


class DetectRequest(BaseModel):
    """Request model for detection endpoint"""
    data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL of the image to process")
    include_visualization: bool = Field(False, description="Include annotated image in response")


class GeminiDetectRequest(BaseModel):
    """Request model for Gemini object detection endpoint"""
    data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL of the image to process")
    prompt: str = Field(
        "Detect all of the prominent items in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.",
        description="Detection prompt"
    )
    include_visualization: bool = Field(False, description="Include annotated image in response")


class GeminiSegmentRequest(BaseModel):
    """Request model for Gemini segmentation endpoint"""
    data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL of the image to process")
    prompt: str = Field(
        "Give the segmentation masks for the wooden and glass items. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels.",
        description="Segmentation prompt"
    )
    include_visualization: bool = Field(False, description="Include annotated image with masks in response")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Transparency for mask overlay (0.0 to 1.0)")
