"""Job management endpoints"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Any

from api.dependencies import get_storage_service
from models.responses import JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status_endpoint(job_id: str):
    """
    Get the status of a processing job

    Args:
        job_id: The job ID returned from /process/video/upload/async

    Returns:
        Job status with progress information
    """
    storage = get_storage_service()
    job = storage.get_job_status(job_id)

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


@router.get("/{job_id}/result")
async def get_job_result(job_id: str) -> Dict[str, Any]:
    """
    Get the result of a completed job

    Args:
        job_id: The job ID returned from /process/video/upload/async

    Returns:
        Complete processing result (same format as synchronous endpoint)
    """
    storage = get_storage_service()
    job = storage.get_job_status(job_id)

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
            if storage.mongo_db is not None:
                try:
                    result_doc = storage.mongo_db["video_results"].find_one({"job_id": job_id})
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


@router.get("/{job_id}/frames")
async def get_job_frames(job_id: str, limit: int = 100, offset: int = 0):
    """
    Get annotated frames for a job

    Args:
        job_id: The job ID
        limit: Maximum number of frames to return
        offset: Number of frames to skip

    Returns:
        List of frame metadata (without full image data for performance)
    """
    storage = get_storage_service()

    if storage.mongo_db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB not available"
        )

    try:
        # Query frames for this job
        frames = list(storage.mongo_db["annotated_frames"].find(
            {"job_id": job_id},
            {"image_base64": 0}  # Exclude large image data
        ).sort("frame_number", 1).skip(offset).limit(limit))

        # Convert ObjectId to string
        for frame in frames:
            frame["_id"] = str(frame["_id"])
            if "gridfs_file_id" in frame:
                frame["gridfs_file_id"] = str(frame["gridfs_file_id"])

        return {
            "job_id": job_id,
            "count": len(frames),
            "offset": offset,
            "limit": limit,
            "frames": frames
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving frames: {str(e)}"
        )


@router.get("/frame/{frame_id}")
async def get_annotated_frame(frame_id: str):
    """
    Get an annotated frame by ID (includes full image data)

    Args:
        frame_id: MongoDB document ID of the annotated frame

    Returns:
        Frame data with base64 image
    """
    storage = get_storage_service()

    if storage.mongo_db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB not available"
        )

    try:
        from bson import ObjectId

        frame = storage.mongo_db["annotated_frames"].find_one({"_id": ObjectId(frame_id)})

        if not frame:
            raise HTTPException(
                status_code=404,
                detail=f"Frame {frame_id} not found"
            )

        # Convert ObjectId to string
        frame["_id"] = str(frame["_id"])

        # Handle GridFS if used
        if frame.get("storage_type") == "gridfs":
            from gridfs import GridFS
            fs = GridFS(storage.mongo_db)
            gridfs_file = fs.get(ObjectId(frame["gridfs_file_id"]))
            frame["image_base64"] = gridfs_file.read()

        return frame

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving frame: {str(e)}"
        )
