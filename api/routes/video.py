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
from utils.upload_progress import UploadProgressTracker, create_storage_progress_callback

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

    # Generate job ID first for progress tracking
    job_id = str(uuid.uuid4())
    temp_path = None

    try:
        # Initialize progress tracker with storage callback
        progress_callback = create_storage_progress_callback(storage)
        progress_tracker = UploadProgressTracker(
            job_id=job_id,
            progress_callback=progress_callback,
            report_interval_bytes=5 * 1024 * 1024  # Report every 5MB
        )

        # Create initial job record for upload tracking
        initial_metadata = {
            "filename": file.filename,
            "status": "uploading"
        }
        storage.create_job(job_id, "video_upload", initial_metadata)

        # Stream file in chunks (memory-efficient for large files)
        file_size = 0
        chunk_size = settings.chunk_size

        # Save file temporarily using chunked streaming
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_path = temp_file.name

            # Read and write file in chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break

                # Check size limit before writing
                if file_size + len(chunk) > settings.max_file_size:
                    temp_file.close()
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    storage.update_job_status(
                        job_id,
                        "failed",
                        error=f"File too large. Maximum: {settings.max_file_size / (1024*1024):.0f}MB"
                    )
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {settings.max_file_size / (1024*1024):.0f}MB. Current size exceeds limit."
                    )

                temp_file.write(chunk)
                file_size += len(chunk)

                # Update progress tracker
                progress_tracker.update(len(chunk))

        # Validate file size
        if file_size == 0:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            storage.update_job_status(job_id, "failed", error="Uploaded file is empty")
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        # Finish progress tracking
        progress_tracker.finish()
        upload_summary = progress_tracker.get_summary()
        print(f"✓ File uploaded successfully: {upload_summary['size_mb']}MB in {upload_summary['elapsed_seconds']}s ({upload_summary['average_speed_mbps']:.2f}MB/s)")

        # Update job record with processing metadata
        metadata = {
            "filename": file.filename,
            "file_size": file_size,
            "file_size_mb": upload_summary['size_mb'],
            "upload_time_seconds": upload_summary['elapsed_seconds'],
            "upload_speed_mbps": upload_summary['average_speed_mbps'],
            "frame_skip": frame_skip,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "min_confidence": min_confidence,
            "upload_method": "chunked_streaming"
        }

        # Update job type from 'video_upload' to 'video_processing'
        storage.update_job_status(
            job_id,
            "pending",
            message="Upload complete. Processing will begin shortly.",
            progress=0.0
        )

        # Update metadata
        if storage.mongo_db is not None:
            try:
                storage.mongo_db["jobs"].update_one(
                    {"job_id": job_id},
                    {"$set": {"job_type": "video_processing", "metadata": metadata}}
                )
            except Exception as e:
                print(f"⚠ Warning: Could not update job metadata: {e}")

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
            message=f"Video processing job created ({file_size / (1024*1024):.2f}MB uploaded). Use the job_id to check status.",
            status_url=f"/jobs/{job_id}"
        )

    except HTTPException:
        # Re-raise HTTPExceptions (they're already properly formatted)
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as cleanup_error:
                print(f"⚠ Warning: Could not delete temp file {temp_path}: {cleanup_error}")
        raise

    except Exception as e:
        # Clean up temp file if job creation failed
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as cleanup_error:
                print(f"⚠ Warning: Could not delete temp file {temp_path}: {cleanup_error}")

        # Ensure all errors return JSON
        import traceback
        print(f"⚠ Error processing upload for job {job_id}:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video upload: {str(e)}"
        )
