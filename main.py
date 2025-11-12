# Prevent OpenCV from trying to load GUI libraries (libGL.so.1)
import os
os.environ['OPENCV_DISABLE_LIBGL'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import settings
from api.dependencies import get_detector, get_storage_service

# Import route modules
from api.routes import health, detection, video, jobs, gemini


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("=" * 60)
    print(f"Initializing {settings.api_title}...")
    print("=" * 60)

    # Initialize services
    storage = get_storage_service()
    storage.connect_mongodb()
    storage.connect_redis()

    # Initialize detector
    try:
        get_detector()
        print("✓ Detector initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize detector: {e}")
        print("   Some endpoints may not work properly")

    print("=" * 60)
    print(f"✓ {settings.api_title} v{settings.api_version} is ready!")
    print(f"  - Docs: http://localhost:{settings.port}/docs")
    print(f"  - Health: http://localhost:{settings.port}/health")
    print("=" * 60)

    yield

    # Shutdown
    print("\nShutting down...")
    storage.close_connections()
    print("✓ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Ensure validation errors return JSON"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "status_code": 422}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch all exceptions and return JSON"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "status_code": 500}
    )


# Include routers
app.include_router(health.router)
app.include_router(detection.router)
app.include_router(video.router)
app.include_router(jobs.router)
app.include_router(gemini.router)
