"""Health check and info endpoints"""
from fastapi import APIRouter
from api.dependencies import get_storage_service
from core.config import settings

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    storage = get_storage_service()

    health_status = {
        "status": "healthy",
        "service": "license-plate-detection-api",
        "version": settings.api_version,
        "databases": {
            "mongodb": "connected" if storage.mongo_client is not None else "disconnected",
            "redis": "connected" if storage.redis_client is not None else "disconnected"
        }
    }

    # Test database connections
    if storage.mongo_client is not None:
        try:
            storage.mongo_client.admin.command('ping')
            health_status["databases"]["mongodb"] = "connected"
        except:
            health_status["databases"]["mongodb"] = "error"

    if storage.redis_client is not None:
        try:
            storage.redis_client.ping()
            health_status["databases"]["redis"] = "connected"
        except:
            health_status["databases"]["redis"] = "error"

    return health_status

@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "endpoints": {
            "/health": "Health check",
            "/detect": "POST - Detect license plates (base64 or URL)",
            "/detect/upload": "POST - Detect license plates (file upload)",
            "/process/video/upload/async": "POST - Process video (async)",
            "/jobs/{job_id}": "GET - Get job status",
            "/gemini/detect": "POST - Gemini object detection",
            "/docs": "API documentation"
        },
        "databases": {
            "mongodb": {
                "connected": get_storage_service().mongo_client is not None
            },
            "redis": {
                "connected": get_storage_service().redis_client is not None
            }
        }
    }
