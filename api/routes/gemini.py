"""Gemini AI endpoints for object detection and segmentation"""
import io
import base64
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from api.dependencies import get_gemini_service
from models.requests import GeminiDetectRequest, GeminiSegmentRequest
from models.responses import (
    GeminiDetectResponse,
    GeminiSegmentResponse,
    GeminiDetection,
    GeminiSegmentation
)

router = APIRouter(prefix="/gemini", tags=["gemini"])


@router.post("/detect", response_model=GeminiDetectResponse)
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

    # Detect objects
    detections = service.detect_objects(image, request.prompt)

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
        GeminiDetection(
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


@router.post("/segment", response_model=GeminiSegmentResponse)
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

    # Segment objects
    segmentations = service.segment_objects(image, request.prompt)

    # Get visualization if requested
    visualization = None
    if request.include_visualization and segmentations:
        annotated_image = service.visualize_segmentations(image, segmentations, alpha=request.alpha)

        # Convert to base64
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        visualization = base64.b64encode(img_bytes).decode('utf-8')

    # Convert to response model (without mask data for size)
    segmentation_items = [
        GeminiSegmentation(
            label=seg["label"],
            box_2d=seg["box_2d"],
            box_2d_normalized=seg["box_2d_normalized"],
            confidence=seg.get("confidence", 1.0)
        )
        for seg in segmentations
    ]

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


@router.post("/detect/upload")
async def gemini_detect_upload(
    file: UploadFile = File(..., description="Image file to process"),
    prompt: Optional[str] = Form(None, description="Custom detection prompt"),
    include_visualization: bool = Form(False, description="Include annotated image in response")
):
    """
    Object detection from uploaded file using Gemini 2.0+
    """
    service = get_gemini_service()

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read and load image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Use default prompt if not provided
    if not prompt:
        prompt = "Detect all of the prominent items in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."

    # Detect objects
    detections = service.detect_objects(image, prompt)

    # Get visualization if requested
    visualization = None
    if include_visualization and detections:
        annotated_image = service.visualize_detections(image, detections)

        # Convert to base64
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        visualization = base64.b64encode(img_bytes).decode('utf-8')

    # Convert to response model
    detection_items = [
        GeminiDetection(
            label=d["label"],
            box_2d=d["box_2d"],
            box_2d_normalized=d["box_2d_normalized"],
            confidence=d.get("confidence", 1.0)
        )
        for d in detections
    ]

    return {
        "detected": len(detections) > 0,
        "count": len(detections),
        "detections": [d.dict() for d in detection_items],
        "message": f"Found {len(detections)} detection(s)" if detections else "No objects detected",
        "image_shape": {
            "width": image.size[0],
            "height": image.size[1]
        },
        "visualization": visualization
    }


@router.post("/segment/upload")
async def gemini_segment_upload(
    file: UploadFile = File(..., description="Image file to process"),
    prompt: Optional[str] = Form(None, description="Custom segmentation prompt"),
    include_visualization: bool = Form(False, description="Include annotated image in response"),
    alpha: float = Form(0.5, description="Transparency for mask overlay (0.0 to 1.0)")
):
    """
    Object segmentation from uploaded file using Gemini 2.5+
    """
    service = get_gemini_service()

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read and load image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Use default prompt if not provided
    if not prompt:
        prompt = "Give the segmentation masks for the wooden and glass items. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels."

    # Segment objects
    segmentations = service.segment_objects(image, prompt)

    # Get visualization if requested
    visualization = None
    if include_visualization and segmentations:
        annotated_image = service.visualize_segmentations(image, segmentations, alpha=alpha)

        # Convert to base64
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        visualization = base64.b64encode(img_bytes).decode('utf-8')

    # Convert to response model
    segmentation_items = [
        GeminiSegmentation(
            label=seg["label"],
            box_2d=seg["box_2d"],
            box_2d_normalized=seg["box_2d_normalized"],
            confidence=seg.get("confidence", 1.0)
        )
        for seg in segmentations
    ]

    return {
        "detected": len(segmentations) > 0,
        "count": len(segmentations),
        "segmentations": [s.dict() for s in segmentation_items],
        "message": f"Found {len(segmentations)} segmentation(s)" if segmentations else "No objects segmented",
        "image_shape": {
            "width": image.size[0],
            "height": image.size[1]
        },
        "visualization": visualization
    }
