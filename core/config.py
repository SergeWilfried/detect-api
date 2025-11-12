"""
Configuration management using Pydantic Settings
Centralizes all environment variables and configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Settings
    api_title: str = "License Plate Detection API"
    api_description: str = "License plate detection using YOLOv11 - Based on Medium article implementation"
    api_version: str = "1.0.0"

    # CORS Settings
    cors_origins: str = "*"  # Comma-separated list of origins

    # YOLO Model Settings
    yolo_model_path: Optional[str] = None
    confidence_threshold: float = 0.25

    # OCR Settings
    ocr_engine: str = "easyocr"  # 'easyocr' or 'gemini'
    ocr_languages: List[str] = ["en"]

    # Gemini API Settings
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"

    # Database Settings
    mongodb_uri: Optional[str] = None
    mongodb_db_name: str = "detect_api"

    # Redis Settings
    redis_url: Optional[str] = None

    # File Settings
    max_file_size: int = 500 * 1024 * 1024  # 500MB in bytes
    model_path: str = "./models/license_plate_detector.pt"
    video_path: str = "./files/2.mp4"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000

    # OpenCV Settings
    opencv_disable_libgl: str = "1"
    qt_qpa_platform: str = "offscreen"

    # Deduplication Settings
    enable_deduplication: bool = True
    dedup_iou_threshold: float = 0.7
    dedup_max_frame_gap: int = 5
    dedup_max_distance: float = 50.0
    dedup_keep_strategy: str = "highest_confidence"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Allow extra env vars without validation errors
    }

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def model_path_resolved(self) -> Optional[Path]:
        """Get resolved absolute path to model file if it exists"""
        if self.yolo_model_path:
            return Path(self.yolo_model_path).resolve()

        model_path = Path(self.model_path).resolve()
        if model_path.exists():
            return model_path

        return None

    @property
    def video_path_resolved(self) -> Optional[Path]:
        """Get resolved absolute path to video file if it exists"""
        video_path = Path(self.video_path).resolve()
        if video_path.exists():
            return video_path
        return None


# Global settings instance
settings = Settings()
