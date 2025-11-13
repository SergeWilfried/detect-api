"""Video processing endpoints"""
import os
import uuid
import tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Body

from api.dependencies import get_detector, get_storage_service
from models.responses import JobSubmitResponse
from services.video_service import process_video_background
from services.s3_service import s3_service
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


@router.post("/upload/s3/presigned-url")
async def get_s3_upload_url(
    filename: str = Body(..., description="Original filename"),
    file_size: Optional[int] = Body(None, description="Expected file size in bytes"),
    content_type: str = Body("video/mp4", description="MIME type of the video")
):
    """
    Generate pre-signed URL for direct S3 upload (RECOMMENDED for large files)

    This endpoint generates a pre-signed POST URL that allows clients to upload
    videos directly to S3, bypassing the API server. This is more efficient for
    large files (150-250MB) and reduces server load.

    **Flow:**
    1. Client calls this endpoint to get upload URL
    2. Client uploads directly to S3 using the provided URL and fields
    3. Client calls `/process/video/s3` with job_id to start processing

    Args:
        filename: Original filename (e.g., "traffic.mp4")
        file_size: Expected file size in bytes (optional, for validation)
        content_type: MIME type (default: "video/mp4")

    Returns:
        Upload URL, fields, and job_id for tracking

    Example:
        ```python
        # Step 1: Get pre-signed URL
        response = requests.post("/process/video/upload/s3/presigned-url", json={
            "filename": "video.mp4",
            "file_size": 200000000
        })
        data = response.json()

        # Step 2: Upload to S3
        with open("video.mp4", "rb") as f:
            files = {"file": f}
            requests.post(data["upload_url"], data=data["upload_fields"], files=files)

        # Step 3: Process video
        requests.post("/process/video/s3", json={"s3_key": data["s3_key"]})
        ```
    """
    if not settings.enable_s3:
        raise HTTPException(
            status_code=503,
            detail="S3 upload is not enabled. Use /upload/async endpoint instead."
        )

    if not s3_service.enabled:
        raise HTTPException(
            status_code=503,
            detail="S3 service is not properly configured. Check AWS credentials and bucket settings."
        )

    try:
        upload_data = s3_service.generate_upload_url(
            filename=filename,
            file_size=file_size,
            content_type=content_type
        )

        # Create job record for tracking
        storage = get_storage_service()
        metadata = {
            "filename": filename,
            "file_size": file_size,
            "s3_key": upload_data["s3_key"],
            "upload_method": "s3_presigned",
            "status": "awaiting_upload"
        }
        storage.create_job(upload_data["job_id"], "s3_upload", metadata)

        return {
            **upload_data,
            "status_url": f"/jobs/{upload_data['job_id']}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate S3 upload URL: {str(e)}"
        )


@router.post("/s3", response_model=JobSubmitResponse)
async def process_video_from_s3(
    background_tasks: BackgroundTasks,
    s3_key: str = Body(..., description="S3 object key of uploaded video"),
    frame_skip: int = Body(1, description="Process every Nth frame"),
    start_frame: Optional[int] = Body(None, description="Start processing from this frame"),
    end_frame: Optional[int] = Body(None, description="Stop processing at this frame"),
    min_confidence: Optional[float] = Body(None, description="Minimum confidence threshold"),
    delete_after_processing: bool = Body(False, description="Delete video from S3 after processing")
):
    """
    Process a video that was uploaded directly to S3

    This endpoint processes videos that were uploaded to S3 using the pre-signed URL
    from `/upload/s3/presigned-url`. The video is downloaded from S3, processed,
    and optionally deleted from S3 after processing.

    Args:
        s3_key: S3 object key (from presigned-url response)
        frame_skip: Process every Nth frame (1 = all frames, 2 = every 2nd frame, etc.)
        start_frame: Start processing from this frame (optional)
        end_frame: Stop processing at this frame (optional)
        min_confidence: Minimum confidence threshold for detections (optional)
        delete_after_processing: Delete video from S3 after processing (default: False)

    Returns:
        Job submission response with job_id and status URL
    """
    if not settings.enable_s3:
        raise HTTPException(
            status_code=503,
            detail="S3 processing is not enabled."
        )

    if not s3_service.enabled:
        raise HTTPException(
            status_code=503,
            detail="S3 service is not properly configured."
        )

    storage = get_storage_service()
    detector = get_detector()
    job_id = str(uuid.uuid4())
    temp_path = None

    try:
        # Get S3 file metadata
        file_metadata = s3_service.get_file_metadata(s3_key)

        # Download video from S3 to temp file
        temp_path = s3_service.download_to_temp(s3_key)

        # Create job record
        metadata = {
            "s3_key": s3_key,
            "file_size": file_metadata["size"],
            "file_size_mb": file_metadata["size_mb"],
            "frame_skip": frame_skip,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "min_confidence": min_confidence,
            "upload_method": "s3_direct",
            "delete_after_processing": delete_after_processing
        }

        storage.create_job(job_id, "video_processing_s3", metadata)

        # Start background processing
        def process_and_cleanup():
            """Process video and optionally delete from S3"""
            try:
                # Process video
                process_video_background(
                    job_id,
                    temp_path,
                    detector,
                    frame_skip,
                    start_frame,
                    end_frame,
                    min_confidence
                )

                # Delete from S3 if requested
                if delete_after_processing:
                    try:
                        s3_service.delete_file(s3_key)
                        print(f"✓ Deleted {s3_key} from S3 after processing")
                    except Exception as del_error:
                        print(f"⚠ Warning: Could not delete {s3_key} from S3: {del_error}")

            except Exception as proc_error:
                print(f"⚠ Error in S3 video processing: {proc_error}")
                raise

        background_tasks.add_task(process_and_cleanup)

        return JobSubmitResponse(
            job_id=job_id,
            status="pending",
            message=f"Video processing job created from S3 ({file_metadata['size_mb']}MB). Use the job_id to check status.",
            status_url=f"/jobs/{job_id}"
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        # Clean up temp file if download succeeded but job creation failed
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as cleanup_error:
                print(f"⚠ Warning: Could not delete temp file {temp_path}: {cleanup_error}")

        import traceback
        print(f"⚠ Error processing S3 video:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video from S3: {str(e)}"
        )
