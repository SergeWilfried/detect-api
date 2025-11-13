# AWS S3 Integration for Large Video Uploads

## Overview

AWS S3 integration provides a **production-ready solution** for handling large video files (150-250MB+) by offloading upload bandwidth from your API server. Clients upload directly to S3 using pre-signed URLs, and the API processes videos from S3.

## Benefits

### 1. **Reduced API Server Load**
- Clients upload directly to S3, not through your API
- API server only handles metadata and processing
- Better scalability for concurrent uploads

### 2. **Better Performance**
- AWS S3's global infrastructure
- No file size through API server memory
- Faster uploads with AWS's CDN

### 3. **Cost Efficiency**
- Reduced bandwidth costs on API server
- S3's competitive storage pricing
- Pay only for what you use

### 4. **Reliability**
- S3's 99.999999999% durability
- Built-in redundancy
- Automatic retry mechanisms

## Architecture

### Upload Flow (Pre-signed URL Method)

```
┌─────────┐                 ┌─────────┐                 ┌─────────┐
│ Client  │────────────────>│   API   │                 │   S3    │
└─────────┘                 └─────────┘                 └─────────┘
     │                            │                           │
     │ 1. Request presigned URL   │                           │
     │───────────────────────────>│                           │
     │                            │                           │
     │ 2. Generate presigned URL  │                           │
     │                            │──────────────────────────>│
     │                            │                           │
     │ 3. Return URL + job_id     │                           │
     │<───────────────────────────│                           │
     │                            │                           │
     │ 4. Upload directly to S3   │                           │
     │────────────────────────────────────────────────────────>│
     │                            │                           │
     │ 5. Request processing      │                           │
     │───────────────────────────>│                           │
     │                            │ 6. Download video         │
     │                            │<──────────────────────────│
     │                            │                           │
     │                            │ 7. Process video          │
     │                            │    (license plate detect) │
     │                            │                           │
     │ 8. Poll job status         │                           │
     │───────────────────────────>│                           │
     │                            │                           │
```

## Setup

### 1. AWS S3 Bucket Configuration

#### Create S3 Bucket
```bash
aws s3 mb s3://your-video-bucket --region us-east-1
```

#### Configure CORS (required for browser uploads)
Create `cors-config.json`:
```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET", "POST", "PUT"],
    "AllowedOrigins": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000
  }
]
```

Apply CORS configuration:
```bash
aws s3api put-bucket-cors \
  --bucket your-video-bucket \
  --cors-configuration file://cors-config.json
```

#### Configure Lifecycle Policy (optional - auto-delete old videos)
Create `lifecycle-policy.json`:
```json
{
  "Rules": [
    {
      "Id": "DeleteOldVideos",
      "Status": "Enabled",
      "Prefix": "videos/",
      "Expiration": {
        "Days": 7
      }
    }
  ]
}
```

Apply lifecycle policy:
```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket your-video-bucket \
  --lifecycle-configuration file://lifecycle-policy.json
```

### 2. AWS IAM Credentials

#### Create IAM User
```bash
aws iam create-user --user-name detect-api-uploader
```

#### Create IAM Policy
Create `s3-upload-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:HeadObject"
      ],
      "Resource": "arn:aws:s3:::your-video-bucket/videos/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::your-video-bucket"
    }
  ]
}
```

Apply policy:
```bash
aws iam put-user-policy \
  --user-name detect-api-uploader \
  --policy-name S3VideoUploadPolicy \
  --policy-document file://s3-upload-policy.json
```

#### Generate Access Keys
```bash
aws iam create-access-key --user-name detect-api-uploader
```

Save the `AccessKeyId` and `SecretAccessKey` from the output.

### 3. Environment Configuration

Add to `.env`:
```env
# AWS S3 Settings
ENABLE_S3=true
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-video-bucket
S3_UPLOAD_EXPIRATION=3600          # 1 hour
S3_DOWNLOAD_EXPIRATION=86400       # 24 hours
S3_VIDEO_PREFIX=videos/
```

### 4. Install Dependencies

```bash
pip install boto3>=1.34.0
```

## API Usage

### Method 1: Pre-signed URL Upload (RECOMMENDED)

#### Step 1: Request Pre-signed URL
```bash
curl -X POST "http://localhost:8000/process/video/upload/s3/presigned-url" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "traffic_video.mp4",
    "file_size": 187000000,
    "content_type": "video/mp4"
  }'
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "upload_url": "https://your-video-bucket.s3.amazonaws.com/",
  "upload_fields": {
    "key": "videos/20250113_143022_550e8400.mp4",
    "Content-Type": "video/mp4",
    "x-amz-meta-original-filename": "traffic_video.mp4",
    "x-amz-meta-job-id": "550e8400-e29b-41d4-a716-446655440000",
    "policy": "eyJleH...",
    "x-amz-algorithm": "AWS4-HMAC-SHA256",
    "x-amz-credential": "AKIA.../20250113/us-east-1/s3/aws4_request",
    "x-amz-date": "20250113T143022Z",
    "x-amz-signature": "abc123..."
  },
  "s3_key": "videos/20250113_143022_550e8400.mp4",
  "bucket": "your-video-bucket",
  "expires_in": 3600,
  "max_file_size": 524288000,
  "status_url": "/jobs/550e8400-e29b-41d4-a716-446655440000"
}
```

#### Step 2: Upload to S3

**Using curl:**
```bash
curl -X POST "https://your-video-bucket.s3.amazonaws.com/" \
  -F "key=videos/20250113_143022_550e8400.mp4" \
  -F "Content-Type=video/mp4" \
  -F "x-amz-meta-original-filename=traffic_video.mp4" \
  -F "x-amz-meta-job-id=550e8400-e29b-41d4-a716-446655440000" \
  -F "policy=eyJleH..." \
  -F "x-amz-algorithm=AWS4-HMAC-SHA256" \
  -F "x-amz-credential=AKIA.../20250113/us-east-1/s3/aws4_request" \
  -F "x-amz-date=20250113T143022Z" \
  -F "x-amz-signature=abc123..." \
  -F "file=@traffic_video.mp4"
```

**Using Python:**
```python
import requests

# Step 1: Get presigned URL
response = requests.post("http://localhost:8000/process/video/upload/s3/presigned-url", json={
    "filename": "traffic_video.mp4",
    "file_size": 187000000
})
data = response.json()

# Step 2: Upload to S3
with open("traffic_video.mp4", "rb") as f:
    files = {"file": f}
    upload_response = requests.post(
        data["upload_url"],
        data=data["upload_fields"],
        files=files
    )

print(f"Upload status: {upload_response.status_code}")
print(f"S3 Key: {data['s3_key']}")
```

**Using JavaScript (Browser):**
```javascript
// Step 1: Get presigned URL
const presignedResponse = await fetch('/process/video/upload/s3/presigned-url', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    filename: 'traffic_video.mp4',
    file_size: videoFile.size
  })
});
const { upload_url, upload_fields, s3_key } = await presignedResponse.json();

// Step 2: Upload to S3
const formData = new FormData();
Object.entries(upload_fields).forEach(([key, value]) => {
  formData.append(key, value);
});
formData.append('file', videoFile);

const uploadResponse = await fetch(upload_url, {
  method: 'POST',
  body: formData
});

console.log('Uploaded to S3:', s3_key);
```

#### Step 3: Process Video
```bash
curl -X POST "http://localhost:8000/process/video/s3" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_key": "videos/20250113_143022_550e8400.mp4",
    "frame_skip": 2,
    "min_confidence": 0.3,
    "delete_after_processing": true
  }'
```

**Response:**
```json
{
  "job_id": "7a2f5c90-e29b-41d4-a716-446655440001",
  "status": "pending",
  "message": "Video processing job created from S3 (187.00MB). Use the job_id to check status.",
  "status_url": "/jobs/7a2f5c90-e29b-41d4-a716-446655440001"
}
```

#### Step 4: Monitor Progress
```bash
curl "http://localhost:8000/jobs/7a2f5c90-e29b-41d4-a716-446655440001"
```

### Method 2: Hybrid Upload (Direct S3 + Chunked)

You can still use the chunked upload endpoint ([CHUNKED_UPLOAD.md](CHUNKED_UPLOAD.md)) if S3 is not configured. The API automatically handles both methods.

## Comparison: Chunked vs S3 Upload

| Feature | Chunked Upload | S3 Pre-signed URL |
|---------|---------------|-------------------|
| **Server Load** | High (all data through API) | Low (metadata only) |
| **Scalability** | Limited by server bandwidth | High (AWS infrastructure) |
| **Upload Speed** | Depends on server location | Fast (AWS global network) |
| **Cost** | Higher bandwidth costs | Lower (S3 storage pricing) |
| **Setup Complexity** | Simple (no external deps) | Moderate (AWS setup) |
| **Best For** | Small files, dev/testing | Large files, production |
| **Max File Size** | 500MB (configurable) | 5TB (S3 limit) |
| **Progress Tracking** | Real-time via API | Client-side only |

## Performance Metrics

### Chunked Upload (through API)
- **200MB video**: ~13.5s @ 14.8MB/s
- **Server memory**: ~1MB (chunked)
- **Server bandwidth**: Full file size

### S3 Pre-signed Upload
- **200MB video**: ~8-12s @ 16-25MB/s (varies by location)
- **Server memory**: ~0 (no file data)
- **Server bandwidth**: Minimal (metadata only)

## Cost Analysis

### Example: 1000 videos/month @ 200MB each

**Chunked Upload:**
- Bandwidth: 200GB upload + 200GB download = 400GB
- Cost: ~$35/mo (AWS EC2 bandwidth @ $0.09/GB)

**S3 Pre-signed:**
- S3 Storage: 200GB @ $0.023/GB = $4.60/mo
- S3 Requests: 1000 uploads + downloads @ $0.005/1000 = $0.01/mo
- Bandwidth (S3→API): 200GB @ $0.09/GB = $18/mo
- **Total**: ~$22.61/mo

**Savings**: ~35% reduction

## Security Best Practices

### 1. Limit Pre-signed URL Expiration
```env
S3_UPLOAD_EXPIRATION=3600  # 1 hour max
```

### 2. Restrict S3 Bucket Access
- Use IAM policies with minimal permissions
- Enable bucket versioning
- Enable server-side encryption

### 3. Validate File Types
```python
# Server-side validation after upload
content_type = s3_service.get_file_metadata(s3_key)["content_type"]
if not content_type.startswith("video/"):
    raise HTTPException(400, "Invalid file type")
```

### 4. Implement File Size Limits
```json
{
  "conditions": [
    ["content-length-range", 0, 524288000]  # 500MB max
  ]
}
```

### 5. Enable S3 Bucket Logging
```bash
aws s3api put-bucket-logging \
  --bucket your-video-bucket \
  --bucket-logging-status file://logging-config.json
```

## Troubleshooting

### Issue: "S3 service is not properly configured"
**Solution**: Check environment variables:
```bash
echo $ENABLE_S3
echo $S3_BUCKET_NAME
echo $AWS_ACCESS_KEY_ID
```

### Issue: "Access Denied" during upload
**Solution**: Verify IAM permissions and CORS configuration

### Issue: Upload succeeds but processing fails
**Solution**: Check S3 bucket name and key in job metadata:
```bash
curl "http://localhost:8000/jobs/{job_id}"
```

### Issue: Slow S3 downloads
**Solution**:
- Use S3 Transfer Acceleration
- Choose region closer to API server
- Increase server bandwidth

## Advanced Configuration

### S3 Transfer Acceleration
Enable faster uploads from distant locations:

```bash
aws s3api put-bucket-accelerate-configuration \
  --bucket your-video-bucket \
  --accelerate-configuration Status=Enabled
```

Update endpoint in code:
```python
# services/s3_service.py
config = Config(
    s3={'use_accelerate_endpoint': True}
)
```

### Multi-part Upload (for files > 100MB)
S3 automatically uses multi-part for large files via boto3.

### CloudFront Integration
Add CDN for faster downloads:
```python
# Generate CloudFront signed URL instead of S3
cloudfront_url = generate_cloudfront_url(s3_key)
```

## Monitoring

### CloudWatch Metrics
Monitor S3 usage:
- **NumberOfObjects**: Track uploaded videos
- **BucketSizeBytes**: Storage usage
- **AllRequests**: API calls to S3

### Custom Metrics
Track in your application:
```python
# Log S3 upload metrics
logger.info("s3_upload", {
    "job_id": job_id,
    "file_size_mb": file_size_mb,
    "upload_time_seconds": upload_time
})
```

## Testing

### Test S3 Connection
```bash
curl -X POST "http://localhost:8000/process/video/upload/s3/presigned-url" \
  -H "Content-Type: application/json" \
  -d '{"filename": "test.mp4", "file_size": 1000000}'
```

### Test Upload Flow
```bash
# Run complete test script
python test_s3_upload.py
```

## Migration from Chunked to S3

### Phase 1: Deploy S3 Integration
1. Set `ENABLE_S3=false` initially
2. Deploy code with S3 service
3. Test with `ENABLE_S3=true` in staging

### Phase 2: Gradual Rollout
1. Enable S3 for new uploads
2. Keep chunked upload as fallback
3. Monitor metrics

### Phase 3: Full Migration
1. Update client applications
2. Deprecate chunked endpoint
3. Remove old code

## Related Files

- [api/routes/video.py](api/routes/video.py#L204-L412) - S3 endpoints
- [services/s3_service.py](services/s3_service.py) - S3 service implementation
- [core/config.py](core/config.py#L41-L49) - S3 configuration
- [CHUNKED_UPLOAD.md](CHUNKED_UPLOAD.md) - Alternative upload method

## Support

For issues or questions:
- Check [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- Review API logs: `/var/log/detect-api.log`
- Test S3 connection: `aws s3 ls s3://your-video-bucket/`
