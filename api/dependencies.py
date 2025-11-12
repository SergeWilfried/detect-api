"""
FastAPI dependencies
Provides singleton instances of services for dependency injection
"""
from typing import Optional
from pathlib import Path
from detection_service import LicensePlateDetector
from services.gemini_service import GeminiImageService
from services.storage_service import storage_service
from core.config import settings


# Global instances
_detector: Optional[LicensePlateDetector] = None
_gemini_service: Optional[GeminiImageService] = None


def get_detector() -> LicensePlateDetector:
    """Get or create the detector instance (singleton pattern)"""
    global _detector
    if _detector is None:
        # Get model path from settings
        model_path = None
        if settings.model_path_resolved:
            model_path = str(settings.model_path_resolved)
            print(f"Using model from: {model_path}")
        else:
            print("⚠ Custom model not found, will use default YOLO model")

        # Create detector
        _detector = LicensePlateDetector(
            model_path=model_path,
            confidence_threshold=settings.confidence_threshold,
            ocr_engine=settings.ocr_engine,
            gemini_api_key=settings.gemini_api_key
        )
    return _detector


def get_gemini_service() -> GeminiImageService:
    """Get or create the Gemini service instance (singleton pattern)"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiImageService(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model
        )
    return _gemini_service


def get_storage_service():
    """Get the storage service instance"""
    return storage_service
