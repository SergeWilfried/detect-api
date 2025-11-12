"""
Custom exceptions for the application
Provides specific exception types for different error scenarios
"""


class DetectionError(Exception):
    """Base exception for detection-related errors"""
    pass


class VideoProcessingError(DetectionError):
    """Raised when video processing fails"""
    pass


class OCRError(DetectionError):
    """Raised when OCR processing fails"""
    pass


class ModelLoadError(DetectionError):
    """Raised when model loading fails"""
    pass


class ImageLoadError(DetectionError):
    """Raised when image loading fails"""
    pass


class DatabaseError(Exception):
    """Base exception for database-related errors"""
    pass


class StorageError(Exception):
    """Raised when storage operations fail"""
    pass


class JobNotFoundError(Exception):
    """Raised when a job is not found"""
    pass


class InvalidFileError(Exception):
    """Raised when an uploaded file is invalid"""
    pass
