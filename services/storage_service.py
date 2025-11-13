"""
Storage service for MongoDB and Redis operations
Handles all database interactions
"""
import os
import json
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

# Database imports
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from core.config import settings
from core.exceptions import DatabaseError, StorageError


class StorageService:
    """Service for handling MongoDB and Redis storage operations"""

    def __init__(self):
        self.mongo_client: Optional[Any] = None
        self.mongo_db: Optional[Any] = None
        self.redis_client: Optional[Any] = None

    def connect_mongodb(self):
        """Connect to MongoDB"""
        if not PYMONGO_AVAILABLE:
            print("⚠ Warning: pymongo not installed. MongoDB features will be disabled.")
            return None, None

        if self.mongo_client is not None:
            return self.mongo_client, self.mongo_db

        mongo_uri = settings.mongodb_uri
        if not mongo_uri:
            print("⚠ Warning: MONGODB_URI not set. MongoDB features disabled.")
            return None, None

        try:
            self.mongo_client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            # Test connection
            self.mongo_client.admin.command('ping')
            self.mongo_db = self.mongo_client[settings.mongodb_db_name]
            print(f"✓ Connected to MongoDB: {settings.mongodb_db_name}")
            return self.mongo_client, self.mongo_db
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"⚠ Warning: Could not connect to MongoDB: {e}")
            self.mongo_client = None
            self.mongo_db = None
            return None, None

    def connect_redis(self):
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            print("⚠ Warning: redis not installed. Redis features will be disabled.")
            return None

        if self.redis_client is not None:
            return self.redis_client

        redis_url = settings.redis_url
        if not redis_url:
            print("⚠ Warning: REDIS_URL not set. Redis features disabled.")
            return None

        try:
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            print("✓ Connected to Redis")
            return self.redis_client
        except Exception as e:
            print(f"⚠ Warning: Could not connect to Redis: {e}")
            self.redis_client = None
            return None

    def close_connections(self):
        """Close database connections"""
        if self.mongo_client is not None:
            try:
                self.mongo_client.close()
                print("✓ MongoDB connection closed")
            except Exception as e:
                print(f"⚠ Error closing MongoDB connection: {e}")

        if self.redis_client is not None:
            try:
                self.redis_client.close()
                print("✓ Redis connection closed")
            except Exception as e:
                print(f"⚠ Error closing Redis connection: {e}")

    # MongoDB operations
    def save_detection(self, detection_data: Dict[str, Any], collection_name: str = "detections") -> Optional[str]:
        """Save detection result to MongoDB"""
        if self.mongo_db is None:
            return None

        try:
            collection = self.mongo_db[collection_name]
            detection_data["created_at"] = datetime.utcnow()
            result = collection.insert_one(detection_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"⚠ Error saving to MongoDB: {e}")
            return None

    def save_annotated_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        frame_number: int,
        timestamp_seconds: float,
        job_id: Optional[str] = None,
        use_gridfs: bool = False,
        annotated_base64: Optional[str] = None
    ) -> Optional[str]:
        """
        Save annotated frame to MongoDB

        Args:
            frame: Original frame as numpy array (BGR format from OpenCV)
            detections: List of detection dictionaries
            frame_number: Frame number in video
            timestamp_seconds: Timestamp in seconds
            job_id: Optional job ID for grouping frames
            use_gridfs: If True, use GridFS for storage (for large images > 16MB)
            annotated_base64: Pre-generated base64 image (if None, will be generated)

        Returns:
            MongoDB document ID or GridFS file ID, or None if failed
        """
        if self.mongo_db is None:
            print(f"⚠ Warning: MongoDB not connected, cannot save annotated frame {frame_number}")
            return None

        if annotated_base64 is None:
            print(f"⚠ Warning: No annotated_base64 provided for frame {frame_number}")
            return None

        try:
            # Prepare frame document
            frame_doc = {
                "frame_number": frame_number,
                "timestamp_seconds": timestamp_seconds,
                "job_id": job_id,
                "detection_count": len(detections) if detections else 0,
                "created_at": datetime.utcnow()
            }

            # Add detection metadata if available
            if detections:
                frame_doc["detections"] = [
                    {
                        "plate_text": d.get("plate_text", ""),
                        "confidence": d.get("confidence", 0.0),
                        "bbox": d.get("bbox", {})
                    }
                    for d in detections
                ]

            if use_gridfs:
                # Use GridFS for large images
                from gridfs import GridFS
                fs = GridFS(self.mongo_db)

                # Decode base64 to bytes
                frame_bytes = base64.b64decode(annotated_base64)

                # Store in GridFS
                file_id = fs.put(
                    frame_bytes,
                    filename=f"frame_{frame_number}_{job_id or 'unknown'}.png",
                    content_type="image/png",
                    frame_number=frame_number,
                    timestamp=timestamp_seconds,
                    job_id=job_id
                )

                frame_doc["gridfs_file_id"] = str(file_id)
                frame_doc["storage_type"] = "gridfs"

                # Save metadata
                result = self.mongo_db["annotated_frames"].insert_one(frame_doc)
                print(f"✓ Saved annotated frame {frame_number} to GridFS (ID: {result.inserted_id})")
                return str(result.inserted_id)
            else:
                # Use base64 in document (works for images < 16MB)
                frame_doc["image_base64"] = annotated_base64
                frame_doc["storage_type"] = "base64"
                frame_doc["image_format"] = "PNG"

                # Save to MongoDB
                result = self.mongo_db["annotated_frames"].insert_one(frame_doc)
                print(f"✓ Saved annotated frame {frame_number} to MongoDB (ID: {result.inserted_id})")
                return str(result.inserted_id)

        except Exception as e:
            import traceback
            print(f"⚠ Error saving annotated frame {frame_number} to MongoDB: {e}")
            traceback.print_exc()
            return None

    # Redis operations
    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if self.redis_client is None:
            return None

        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"⚠ Error reading from Redis: {e}")
            return None

    def set_to_cache(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if self.redis_client is None:
            return False

        try:
            json_value = json.dumps(value, default=str) if not isinstance(value, str) else value
            if expire_seconds:
                self.redis_client.setex(key, expire_seconds, json_value)
            else:
                self.redis_client.set(key, json_value)
            return True
        except Exception as e:
            print(f"⚠ Error writing to Redis: {e}")
            return False

    # Job management
    def create_job(self, job_id: str, job_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new job record"""
        job_data = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "pending",
            "progress": 0.0,
            "message": "Job queued",
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata
        }

        # Save to MongoDB
        if self.mongo_db is not None:
            try:
                self.mongo_db["jobs"].insert_one(job_data.copy())
            except Exception as e:
                print(f"⚠ Error saving job to MongoDB: {e}")

        # Save to Redis for quick access (expires in 24 hours)
        self.set_to_cache(f"job:{job_id}", job_data, expire_seconds=86400)

        return job_data

    def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None
    ):
        """Update job status"""
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }

        if progress is not None:
            update_data["progress"] = progress

        if message is not None:
            update_data["message"] = message

        if error is not None:
            update_data["error"] = error

        if status == "processing" and "started_at" not in update_data:
            update_data["started_at"] = datetime.utcnow().isoformat()

        if status in ["completed", "failed"]:
            update_data["completed_at"] = datetime.utcnow().isoformat()
            if result:
                update_data["result"] = result

        # Update MongoDB
        if self.mongo_db is not None:
            try:
                self.mongo_db["jobs"].update_one(
                    {"job_id": job_id},
                    {"$set": update_data}
                )
            except Exception as e:
                print(f"⚠ Error updating job in MongoDB: {e}")

        # Update Redis
        current_job = self.get_from_cache(f"job:{job_id}")
        if current_job:
            current_job.update(update_data)
            self.set_to_cache(f"job:{job_id}", current_job, expire_seconds=86400)

        return update_data

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status data from Redis or MongoDB"""
        # Try Redis first (faster)
        job = self.get_from_cache(f"job:{job_id}")

        if not job and self.mongo_db is not None:
            # Fallback to MongoDB
            try:
                job_doc = self.mongo_db["jobs"].find_one({"job_id": job_id})
                if job_doc:
                    job = dict(job_doc)
                    # Remove MongoDB _id
                    job.pop("_id", None)
                    # Cache in Redis
                    self.set_to_cache(f"job:{job_id}", job, expire_seconds=86400)
            except Exception as e:
                print(f"⚠ Error getting job from MongoDB: {e}")

        return job

    def save_video_result(self, job_id: str, result: Dict[str, Any]) -> Optional[str]:
        """Save video processing result to MongoDB"""
        if self.mongo_db is None:
            return None

        try:
            result_doc = {
                "job_id": job_id,
                "result": result,
                "created_at": datetime.utcnow()
            }
            inserted = self.mongo_db["video_results"].insert_one(result_doc)
            return str(inserted.inserted_id)
        except Exception as e:
            print(f"⚠ Error saving video result to MongoDB: {e}")
            return None


# Global storage service instance
storage_service = StorageService()
