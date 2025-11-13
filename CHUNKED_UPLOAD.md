# Chunked Upload Implementation

## Overview

The video upload endpoint now uses **chunked streaming** to handle large video files (150-250MB) efficiently without loading the entire file into memory.

## Features

### 1. Memory-Efficient Streaming
- Files are read and written in **1MB chunks** (configurable)
- Maximum memory usage: ~1MB regardless of file size
- Prevents memory overflow on large uploads

### 2. Real-Time Progress Tracking
- Progress updates every **5MB** uploaded
- Tracks upload speed (MB/s)
- Estimates time remaining (ETA)
- Updates job status in real-time

### 3. Enhanced Error Handling
- Size validation during upload (not after)
- Automatic temp file cleanup on errors
- Detailed error messages with context

## Configuration

Environment variables in `.env`:

```env
# Upload Settings
MAX_FILE_SIZE=524288000          # 500MB max (in bytes)
CHUNK_SIZE=1048576               # 1MB chunks (in bytes)
UPLOAD_TIMEOUT_SECONDS=600       # 10 minutes timeout
```

## API Usage

### Endpoint
```
POST /process/video/upload/async
```

### Request
```bash
curl -X POST "http://localhost:8000/process/video/upload/async" \
  -F "file=@large_video.mp4" \
  -F "frame_skip=2" \
  -F "min_confidence=0.3"
```

### Response
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Video processing job created (187.45MB uploaded). Use the job_id to check status.",
  "status_url": "/jobs/550e8400-e29b-41d4-a716-446655440000"
}
```

### Check Progress
```bash
curl "http://localhost:8000/jobs/550e8400-e29b-41d4-a716-446655440000"
```

### Progress Responses

**During Upload:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploading",
  "progress": 0.53,
  "message": "Uploading: 100.0MB (53%) at 12.5MB/s - ETA: 7s"
}
```

**Upload Complete, Processing:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0.25,
  "message": "Processed 500/2000 frames (5 unique plates, 47 total detections)"
}
```

## Architecture

### Upload Flow

1. **Client uploads video** → Multipart form data
2. **Server generates job_id** → Immediate response
3. **Chunked streaming begins** → 1MB chunks
4. **Progress tracking** → Updates every 5MB
   - Calculates upload speed
   - Estimates ETA
   - Updates job status in MongoDB/Redis
5. **Upload completes** → File saved to temp location
6. **Background processing starts** → Video analysis
7. **Temp file cleanup** → After processing

### Components

#### 1. UploadProgressTracker (`utils/upload_progress.py`)
- Tracks bytes uploaded
- Calculates speed and ETA
- Calls progress callback
- Provides upload summary

#### 2. Storage Callback
- Updates job status in MongoDB/Redis
- Stores progress metadata
- Enables real-time status queries

#### 3. Video Endpoint (`api/routes/video.py`)
- Streams file in chunks
- Validates size during upload
- Handles errors gracefully
- Cleans up temp files

## Performance Metrics

### Memory Usage
- **Before**: Up to 500MB (entire file in memory)
- **After**: ~1MB (single chunk size)
- **Improvement**: 99.8% reduction for 500MB files

### Upload Speed (Example: 200MB file)
```
Upload progress [job-123]: Uploading: 50.0MB (25%) at 15.2MB/s - ETA: 10s
Upload progress [job-123]: Uploading: 100.0MB (50%) at 14.8MB/s - ETA: 7s
Upload progress [job-123]: Uploading: 150.0MB (75%) at 15.1MB/s - ETA: 3s
✓ File uploaded successfully: 200.00MB in 13.5s (14.81MB/s)
```

## Error Handling

### File Too Large
```json
{
  "detail": "File too large. Maximum size is 500MB. Current size exceeds limit.",
  "status_code": 413
}
```

### Empty File
```json
{
  "detail": "Uploaded file is empty",
  "status_code": 400
}
```

### Invalid File Type
```json
{
  "detail": "File must be a video (mp4, avi, mov, mkv, webm, flv)",
  "status_code": 400
}
```

## Testing

### Test with Large File
```bash
# Generate 200MB test video
ffmpeg -f lavfi -i testsrc=duration=60:size=1920x1080:rate=30 \
  -c:v libx264 -preset fast test_200mb.mp4

# Upload
curl -X POST "http://localhost:8000/process/video/upload/async" \
  -F "file=@test_200mb.mp4" \
  -F "frame_skip=5"
```

### Monitor Progress
```bash
# Poll job status
watch -n 1 'curl -s "http://localhost:8000/jobs/YOUR_JOB_ID" | jq'
```

## Future Enhancements

### 1. Resumable Uploads (tus protocol)
- Allow upload resume after network failure
- Better for mobile/unstable connections

### 2. Cloud Storage Integration
- Direct upload to S3/Azure/GCP
- Pre-signed URLs
- Reduced server load

### 3. WebSocket Progress Updates
- Real-time progress without polling
- Push notifications to client

### 4. Multi-part Upload
- Parallel chunk uploads
- Faster for large files
- Better bandwidth utilization

## Troubleshooting

### Timeout During Upload
Increase timeout in config:
```env
UPLOAD_TIMEOUT_SECONDS=1200  # 20 minutes
```

### Slow Upload Speed
Check chunk size configuration:
```env
CHUNK_SIZE=2097152  # Try 2MB chunks
```

### Memory Issues
Reduce chunk size:
```env
CHUNK_SIZE=524288  # 512KB chunks
```

## Related Files

- [api/routes/video.py](api/routes/video.py) - Video upload endpoint
- [utils/upload_progress.py](utils/upload_progress.py) - Progress tracking
- [core/config.py](core/config.py) - Configuration settings
- [services/storage_service.py](services/storage_service.py) - Job status storage
