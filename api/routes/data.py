"""Data retrieval endpoints for detections and violations"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any

from api.dependencies import get_storage_service

router = APIRouter(tags=["data"])


@router.get("/detections")
async def list_detections(
    job_id: Optional[str] = Query(None, description="Filter by job ID"),
    plate_text: Optional[str] = Query(None, description="Filter by license plate text"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of detections to return"),
    offset: int = Query(0, ge=0, description="Number of detections to skip")
):
    """
    List all detections from video processing results
    
    Args:
        job_id: Optional job ID filter
        plate_text: Optional license plate text filter (partial match)
        limit: Maximum number of detections to return (1-1000)
        offset: Number of detections to skip
        
    Returns:
        List of detections with pagination info
    """
    storage = get_storage_service()
    
    if storage.mongo_db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB not available"
        )
    
    try:
        # Query individual detections collection directly (more efficient)
        query = {}
        if job_id:
            query["job_id"] = job_id
        
        if plate_text:
            # Case-insensitive partial match for plate text
            query["plate_text"] = {"$regex": plate_text, "$options": "i"}
        
        # Get total count
        total = storage.mongo_db["detections"].count_documents(query)
        
        # Query with pagination
        detections_cursor = storage.mongo_db["detections"].find(query).sort("created_at", -1).skip(offset).limit(limit)
        paginated_detections = list(detections_cursor)
        
        # Convert ObjectId to string
        for det in paginated_detections:
            det["_id"] = str(det["_id"])
        
        return {
            "detections": paginated_detections,
            "total": total,
            "limit": limit,
            "offset": offset,
            "count": len(paginated_detections)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving detections: {str(e)}"
        )


@router.get("/violations")
async def list_violations(
    job_id: Optional[str] = Query(None, description="Filter by job ID"),
    plate_text: Optional[str] = Query(None, description="Filter by license plate text"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of violations to return"),
    offset: int = Query(0, ge=0, description="Number of violations to skip")
):
    """
    List all violations (detections that may indicate violations)
    
    Currently returns all detections. Can be extended with custom violation logic.
    
    Args:
        job_id: Optional job ID filter
        plate_text: Optional license plate text filter (partial match)
        min_confidence: Optional minimum confidence threshold
        limit: Maximum number of violations to return (1-1000)
        offset: Number of violations to skip
        
    Returns:
        List of violations with pagination info
    """
    storage = get_storage_service()
    
    if storage.mongo_db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB not available"
        )
    
    try:
        # Query individual detections collection directly (more efficient)
        query = {}
        if job_id:
            query["job_id"] = job_id
        
        if plate_text:
            # Case-insensitive partial match for plate text
            query["plate_text"] = {"$regex": plate_text, "$options": "i"}
        
        if min_confidence is not None:
            query["confidence"] = {"$gte": min_confidence}
        
        # Get total count
        total = storage.mongo_db["detections"].count_documents(query)
        
        # Query with pagination
        detections_cursor = storage.mongo_db["detections"].find(query).sort("created_at", -1).skip(offset).limit(limit)
        paginated_violations = list(detections_cursor)
        
        # Convert ObjectId to string and add violation metadata
        for violation in paginated_violations:
            violation["_id"] = str(violation["_id"])
            # Add violation metadata (can be extended with custom logic)
            violation["violation_type"] = "license_plate_detected"  # Default
        
        return {
            "violations": paginated_violations,
            "total": total,
            "limit": limit,
            "offset": offset,
            "count": len(paginated_violations)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving violations: {str(e)}"
        )

