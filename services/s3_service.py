"""
AWS S3 service for video upload and storage
Handles pre-signed URLs, file uploads, and downloads
"""
import os
import uuid
import tempfile
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from core.config import settings


class S3Service:
    """Service for AWS S3 operations"""

    def __init__(self):
        self.s3_client = None
        self.bucket_name = settings.s3_bucket_name
        self.enabled = settings.enable_s3

        if self.enabled and BOTO3_AVAILABLE:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize S3 client with credentials"""
        if not BOTO3_AVAILABLE:
            print("⚠ Warning: boto3 not installed. S3 features will be disabled.")
            self.enabled = False
            return

        if not settings.s3_bucket_name:
            print("⚠ Warning: S3_BUCKET_NAME not set. S3 features disabled.")
            self.enabled = False
            return

        try:
            # Configure S3 client with credentials and region
            config = Config(
                region_name=settings.aws_region,
                signature_version='s3v4',
                retries={
                    'max_attempts': 3,
                    'mode': 'standard'
                }
            )

            # Create S3 client
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    config=config
                )
            else:
                # Use IAM role or environment credentials
                self.s3_client = boto3.client('s3', config=config)

            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✓ Connected to S3 bucket: {self.bucket_name}")

        except NoCredentialsError:
            print("⚠ Warning: AWS credentials not found. S3 features disabled.")
            self.enabled = False
            self.s3_client = None
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            print(f"⚠ Warning: Could not connect to S3 bucket '{self.bucket_name}': {error_code}")
            self.enabled = False
            self.s3_client = None
        except Exception as e:
            print(f"⚠ Warning: S3 initialization failed: {e}")
            self.enabled = False
            self.s3_client = None

    def generate_upload_url(
        self,
        filename: str,
        file_size: Optional[int] = None,
        content_type: str = "video/mp4"
    ) -> Dict[str, Any]:
        """
        Generate pre-signed URL for direct client upload to S3

        Args:
            filename: Original filename
            file_size: Expected file size in bytes (optional)
            content_type: MIME type of the file

        Returns:
            Dict with upload_url, s3_key, job_id, and expiration info
        """
        if not self.enabled or not self.s3_client:
            raise Exception("S3 is not enabled or configured")

        # Generate unique S3 key
        job_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_ext = os.path.splitext(filename)[1] or '.mp4'
        s3_key = f"{settings.s3_video_prefix}{timestamp}_{job_id}{file_ext}"

        try:
            # Generate pre-signed POST URL for upload
            conditions = [
                {"bucket": self.bucket_name},
                {"key": s3_key},
                {"Content-Type": content_type}
            ]

            # Add size limit if provided
            if file_size:
                conditions.append(["content-length-range", 0, settings.max_file_size])

            # Generate pre-signed POST
            presigned_post = self.s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=s3_key,
                Fields={
                    "Content-Type": content_type,
                    "x-amz-meta-original-filename": filename,
                    "x-amz-meta-job-id": job_id
                },
                Conditions=conditions,
                ExpiresIn=settings.s3_upload_expiration
            )

            return {
                "job_id": job_id,
                "upload_url": presigned_post['url'],
                "upload_fields": presigned_post['fields'],
                "s3_key": s3_key,
                "bucket": self.bucket_name,
                "expires_in": settings.s3_upload_expiration,
                "max_file_size": settings.max_file_size,
                "instructions": {
                    "method": "POST",
                    "note": "Use multipart/form-data. Include all upload_fields, then add 'file' field with video content."
                }
            }

        except ClientError as e:
            raise Exception(f"Failed to generate pre-signed URL: {e}")

    def generate_download_url(self, s3_key: str, expires_in: Optional[int] = None) -> str:
        """
        Generate pre-signed URL for downloading a file from S3

        Args:
            s3_key: S3 object key
            expires_in: Expiration time in seconds (default: 24 hours)

        Returns:
            Pre-signed download URL
        """
        if not self.enabled or not self.s3_client:
            raise Exception("S3 is not enabled or configured")

        if expires_in is None:
            expires_in = settings.s3_download_expiration

        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expires_in
            )
            return url

        except ClientError as e:
            raise Exception(f"Failed to generate download URL: {e}")

    def download_to_temp(self, s3_key: str) -> str:
        """
        Download S3 file to temporary local file

        Args:
            s3_key: S3 object key

        Returns:
            Path to temporary file
        """
        if not self.enabled or not self.s3_client:
            raise Exception("S3 is not enabled or configured")

        try:
            # Get file extension
            file_ext = os.path.splitext(s3_key)[1] or '.mp4'

            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            temp_path = temp_file.name
            temp_file.close()

            # Download from S3
            print(f"📥 Downloading {s3_key} from S3...")
            self.s3_client.download_file(self.bucket_name, s3_key, temp_path)

            file_size = os.path.getsize(temp_path)
            print(f"✓ Downloaded {file_size / (1024*1024):.2f}MB to {temp_path}")

            return temp_path

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            raise Exception(f"Failed to download from S3 (Error: {error_code}): {e}")

    def upload_file(self, file_path: str, s3_key: Optional[str] = None) -> str:
        """
        Upload a local file to S3

        Args:
            file_path: Path to local file
            s3_key: S3 object key (if None, generates one)

        Returns:
            S3 key of uploaded file
        """
        if not self.enabled or not self.s3_client:
            raise Exception("S3 is not enabled or configured")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate S3 key if not provided
        if s3_key is None:
            filename = os.path.basename(file_path)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            job_id = str(uuid.uuid4())
            file_ext = os.path.splitext(filename)[1] or '.mp4'
            s3_key = f"{settings.s3_video_prefix}{timestamp}_{job_id}{file_ext}"

        try:
            file_size = os.path.getsize(file_path)
            print(f"📤 Uploading {file_size / (1024*1024):.2f}MB to S3: {s3_key}")

            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'video/mp4'}
            )

            print(f"✓ Uploaded to S3: {s3_key}")
            return s3_key

        except ClientError as e:
            raise Exception(f"Failed to upload to S3: {e}")

    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3

        Args:
            s3_key: S3 object key

        Returns:
            True if successful
        """
        if not self.enabled or not self.s3_client:
            raise Exception("S3 is not enabled or configured")

        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            print(f"✓ Deleted from S3: {s3_key}")
            return True

        except ClientError as e:
            print(f"⚠ Warning: Failed to delete {s3_key} from S3: {e}")
            return False

    def get_file_metadata(self, s3_key: str) -> Dict[str, Any]:
        """
        Get metadata for an S3 object

        Args:
            s3_key: S3 object key

        Returns:
            Dict with file metadata
        """
        if not self.enabled or not self.s3_client:
            raise Exception("S3 is not enabled or configured")

        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)

            return {
                "s3_key": s3_key,
                "size": response.get('ContentLength', 0),
                "size_mb": round(response.get('ContentLength', 0) / (1024*1024), 2),
                "content_type": response.get('ContentType'),
                "last_modified": response.get('LastModified'),
                "metadata": response.get('Metadata', {})
            }

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                raise FileNotFoundError(f"File not found in S3: {s3_key}")
            raise Exception(f"Failed to get file metadata: {e}")


# Global S3 service instance
s3_service = S3Service()
