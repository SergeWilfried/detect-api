"""License plate detection endpoints"""
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
    """License plate detection from base64 or URL"""
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
    """License plate detection from uploaded file"""
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
        image_shape={
            "height": image_np.shape[0],
            "width": image_np.shape[1]
        },
        visualization=visualization
    )

