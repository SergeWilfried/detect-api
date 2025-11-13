"""
Upload progress tracking utilities
Provides real-time upload progress tracking for large files
"""
from typing import Optional, Callable
import time


class UploadProgressTracker:
    """Tracks upload progress and reports to storage service"""

    def __init__(
        self,
        job_id: str,
        total_size: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        report_interval_bytes: int = 5 * 1024 * 1024  # Report every 5MB
    ):
        """
        Initialize upload progress tracker

        Args:
            job_id: Unique job identifier
            total_size: Total expected file size (if known)
            progress_callback: Optional callback function(job_id, bytes_uploaded, total_size, percentage)
            report_interval_bytes: Report progress every N bytes (default: 5MB)
        """
        self.job_id = job_id
        self.total_size = total_size
        self.bytes_uploaded = 0
        self.progress_callback = progress_callback
        self.report_interval_bytes = report_interval_bytes
        self.last_reported_bytes = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time

    def update(self, chunk_size: int) -> bool:
        """
        Update progress with newly uploaded chunk

        Args:
            chunk_size: Size of the chunk just uploaded

        Returns:
            True if progress was reported, False otherwise
        """
        self.bytes_uploaded += chunk_size

        # Check if we should report progress
        if self.bytes_uploaded - self.last_reported_bytes >= self.report_interval_bytes:
            return self._report_progress()

        return False

    def _report_progress(self) -> bool:
        """Report current progress via callback"""
        if self.progress_callback is None:
            return False

        current_time = time.time()
        elapsed_time = current_time - self.start_time
        time_since_last_report = current_time - self.last_report_time

        # Calculate upload speed
        upload_speed_mbps = 0.0
        if time_since_last_report > 0:
            bytes_since_last_report = self.bytes_uploaded - self.last_reported_bytes
            upload_speed_mbps = (bytes_since_last_report / (1024 * 1024)) / time_since_last_report

        # Calculate percentage if total size is known
        percentage = None
        eta_seconds = None
        if self.total_size and self.total_size > 0:
            percentage = (self.bytes_uploaded / self.total_size) * 100
            if upload_speed_mbps > 0:
                remaining_mb = (self.total_size - self.bytes_uploaded) / (1024 * 1024)
                eta_seconds = remaining_mb / upload_speed_mbps

        try:
            self.progress_callback(
                job_id=self.job_id,
                bytes_uploaded=self.bytes_uploaded,
                total_size=self.total_size,
                percentage=percentage,
                upload_speed_mbps=upload_speed_mbps,
                eta_seconds=eta_seconds,
                elapsed_seconds=elapsed_time
            )
            self.last_reported_bytes = self.bytes_uploaded
            self.last_report_time = current_time
            return True
        except Exception as e:
            print(f"⚠ Error reporting progress: {e}")
            return False

    def finish(self):
        """Report final progress when upload is complete"""
        # Force final report
        self.last_reported_bytes = 0
        self._report_progress()

    def get_summary(self) -> dict:
        """Get upload summary statistics"""
        elapsed_time = time.time() - self.start_time
        avg_speed_mbps = 0.0
        if elapsed_time > 0:
            avg_speed_mbps = (self.bytes_uploaded / (1024 * 1024)) / elapsed_time

        return {
            "job_id": self.job_id,
            "bytes_uploaded": self.bytes_uploaded,
            "size_mb": round(self.bytes_uploaded / (1024 * 1024), 2),
            "total_size": self.total_size,
            "elapsed_seconds": round(elapsed_time, 2),
            "average_speed_mbps": round(avg_speed_mbps, 2),
            "completed": True
        }


def create_storage_progress_callback(storage_service):
    """
    Create a progress callback that updates job status in storage service

    Args:
        storage_service: StorageService instance

    Returns:
        Callback function compatible with UploadProgressTracker
    """
    def callback(job_id, bytes_uploaded, total_size, percentage, upload_speed_mbps, eta_seconds, elapsed_seconds):
        """Update job status with upload progress"""
        size_mb = bytes_uploaded / (1024 * 1024)

        message_parts = [f"Uploading: {size_mb:.1f}MB"]

        if percentage is not None:
            message_parts.append(f"({percentage:.1f}%)")

        if upload_speed_mbps is not None:
            message_parts.append(f"at {upload_speed_mbps:.1f}MB/s")

        if eta_seconds is not None and eta_seconds > 0:
            if eta_seconds < 60:
                message_parts.append(f"- ETA: {eta_seconds:.0f}s")
            else:
                message_parts.append(f"- ETA: {eta_seconds/60:.1f}m")

        message = " ".join(message_parts)

        # Update job status
        storage_service.update_job_status(
            job_id=job_id,
            status="uploading",
            progress=percentage / 100.0 if percentage else None,
            message=message
        )

        print(f"📤 Upload progress [{job_id}]: {message}")

    return callback
