"""Video processing endpoints"""
import os
import uuid
import tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException

from api.dependencies import get_detector, get_storage_service
from models.responses import JobSubmitResponse
from services.video_service import process_video_background
from core.config import settings

router = APIRouter(prefix="/process/video", tags=["video"])


@router.post("/upload/async", response_model=JobSubmitResponse)
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
    storage = get_storage_service()
    detector = get_detector()

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

    if file_size > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size / (1024*1024):.0f}MB. Your file is {file_size / (1024*1024):.2f}MB."
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
            storage.create_job(job_id, "video_processing", metadata)
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
            detector,
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
        # Re-raise HTTPExceptions (they're already properly formatted)
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
