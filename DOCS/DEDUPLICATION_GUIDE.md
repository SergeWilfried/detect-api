# Deduplication Feature Guide

## Overview

The deduplication feature intelligently removes duplicate license plate detections from video processing results. When processing videos frame-by-frame, the same license plate often appears in multiple consecutive frames, creating redundant detections. This feature identifies and removes these duplicates while preserving the highest quality detection.

## Why Deduplication?

Without deduplication:
- A single license plate in a 30 fps video could generate 30+ detections per second
- Results become bloated with redundant data
- Difficult to determine unique plates vs repeated detections
- Storage and bandwidth waste

With deduplication:
- Keeps only the best detection from each duplicate group
- Provides accurate counts of unique detections
- Maintains full visibility of duplicates for audit purposes
- Includes stability metrics for quality assessment

## How It Works

### Detection-Based Deduplication

The system uses a smart detection-based approach that considers:

1. **Plate Text Matching**: Detections with the same plate text are candidates for deduplication
2. **Frame Proximity**: Only detections within N frames of each other are compared
3. **Spatial Overlap**: Uses IoU (Intersection over Union) to measure bounding box overlap
4. **Distance Fallback**: Also checks center-to-center distance between bounding boxes

Two detections are considered duplicates if:
- They have the same plate text (or both are unknown)
- They are within `max_frame_gap` frames of each other (default: 5)
- EITHER:
  - Their bounding boxes overlap by at least `iou_threshold` (default: 70%)
  - OR their centers are within `max_distance` pixels (default: 50)

### Keep Strategy

When duplicates are found, the system keeps the best detection using one of these strategies:
- **highest_confidence** (default): Keep the detection with the highest confidence score
- **first**: Keep the first detection in the sequence
- **last**: Keep the last detection in the sequence

## Configuration

### Environment Variables

Add these to your `.env` file to customize deduplication behavior:

```bash
# Enable/disable deduplication
ENABLE_DEDUPLICATION=true

# IoU threshold for bbox overlap (0.0 to 1.0)
# Higher = stricter overlap required
DEDUP_IOU_THRESHOLD=0.7

# Maximum frames apart to consider as duplicates
# Smaller = faster processing, but may miss duplicates
DEDUP_MAX_FRAME_GAP=5

# Maximum pixel distance between centers
# Larger = more lenient spatial matching
DEDUP_MAX_DISTANCE=50.0

# Strategy for keeping detections
# Options: highest_confidence, first, last
DEDUP_KEEP_STRATEGY=highest_confidence
```

### Code Configuration

In `core/config.py`:

```python
class Settings(BaseSettings):
    # Deduplication Settings
    enable_deduplication: bool = True
    dedup_iou_threshold: float = 0.7
    dedup_max_frame_gap: int = 5
    dedup_max_distance: float = 50.0
    dedup_keep_strategy: str = "highest_confidence"
```

## Response Structure

### Statistics

The video processing response includes comprehensive deduplication statistics:

```json
{
  "statistics": {
    "total_detections": 1247,          // Before deduplication
    "unique_detections": 423,          // After deduplication
    "duplicate_detections": 824,       // Duplicates removed
    "deduplication_rate": 66.1,        // Percentage removed
    ...
  },
  "deduplication": {
    "total_detections": 1247,
    "unique_detections": 423,
    "duplicate_detections": 824,
    "deduplication_rate": 66.1,
    "kept_strategy": "highest_confidence",
    "config": {
      "iou_threshold": 0.7,
      "max_frame_gap": 5,
      "max_distance": 50.0
    }
  }
}
```

### Detection Lists

```json
{
  "all_detections": [/* Unique detections only */],
  "duplicate_detections": [/* Removed duplicates */]
}
```

### Stability Metrics

Each plate summary includes stability analysis:

```json
{
  "plate_summaries": [
    {
      "plate_text": "ABC123",
      "total_occurrences": 15,
      "stability": {
        "is_stable": true,
        "stability_score": 0.892,          // 0-1, higher = more stable
        "bbox_variance": 12.34,            // Position variance in pixels
        "confidence_variance": 0.002,       // Confidence variance
        "position_stability": 0.877,        // 0-1, position consistency
        "confidence_stability": 0.908       // 0-1, confidence consistency
      },
      ...
    }
  ]
}
```

## Usage Examples

### Default Configuration

```python
# Processing runs with default settings from environment
# Deduplication is enabled automatically
result = await process_video_background(
    job_id="job123",
    video_path="video.mp4",
    detector=detector,
    frame_skip=1,
    start_frame=None,
    end_frame=None,
    min_confidence=0.25
)
```

### Disabling Deduplication

```bash
# In .env
ENABLE_DEDUPLICATION=false
```

```python
# Or modify settings at runtime (not recommended)
from core.config import settings
settings.enable_deduplication = False
```

### Custom Thresholds

```bash
# More aggressive deduplication (remove more duplicates)
DEDUP_IOU_THRESHOLD=0.5      # Lower overlap requirement
DEDUP_MAX_FRAME_GAP=10       # Larger frame window
DEDUP_MAX_DISTANCE=100.0     # Larger distance tolerance

# More conservative deduplication (keep more detections)
DEDUP_IOU_THRESHOLD=0.9      # Higher overlap requirement
DEDUP_MAX_FRAME_GAP=2        # Smaller frame window
DEDUP_MAX_DISTANCE=25.0      # Smaller distance tolerance
```

## Performance Considerations

### Computational Complexity

- **Time Complexity**: O(n²) worst case, but optimized with early stopping
- **Space Complexity**: O(n) for storing detections
- **Real-world Performance**: Minimal overhead (typically <5% of total processing time)

### Optimization Features

1. **Frame-based sorting**: Detections are sorted by frame number for efficient comparison
2. **Early stopping**: Comparison stops when frame gap exceeds threshold
3. **Sliding window**: Only nearby frames are compared, not all detections

### Tuning for Performance

**For faster processing** (may miss some duplicates):
- Reduce `dedup_max_frame_gap` (e.g., 3)
- Reduce `dedup_max_distance` (e.g., 30.0)

**For thorough deduplication** (slower but more accurate):
- Increase `dedup_max_frame_gap` (e.g., 10)
- Increase `dedup_max_distance` (e.g., 100.0)

## Quality Assessment

### Stability Score

The stability score (0-1) indicates detection quality:

- **0.8-1.0**: Excellent - Very stable plate across frames
- **0.6-0.8**: Good - Reasonably stable, minor variance
- **0.4-0.6**: Fair - Moderate variance, may be moving vehicle
- **0.0-0.4**: Poor - High variance, possibly false detections

### Using Stability for Filtering

```python
# Filter for high-quality detections only
high_quality_plates = [
    plate for plate in result['plate_summaries']
    if plate['stability']['stability_score'] > 0.7
]
```

## API Response Model

### DeduplicationStats

```python
class DeduplicationStats(BaseModel):
    total_detections: int          # Total before deduplication
    unique_detections: int         # After deduplication
    duplicate_detections: int      # Number removed
    deduplication_rate: float      # Percentage removed
    kept_strategy: str             # Strategy used
    config: Dict[str, Any]         # Configuration applied
```

### StabilityMetrics

```python
class StabilityMetrics(BaseModel):
    is_stable: bool                # Stable threshold met (>0.6)
    stability_score: float         # Overall score (0-1)
    bbox_variance: float           # Position variance (pixels)
    confidence_variance: float     # Confidence variance
    position_stability: float      # Position stability (0-1)
    confidence_stability: float    # Confidence stability (0-1)
```

## Troubleshooting

### Too Many Duplicates Removed

**Symptoms**: Very few detections remain, `deduplication_rate` > 80%

**Solutions**:
- Increase `dedup_iou_threshold` to 0.8 or 0.9
- Decrease `dedup_max_frame_gap` to 3 or 4
- Decrease `dedup_max_distance` to 30 or 40

### Not Enough Duplicates Removed

**Symptoms**: Still seeing obvious duplicates, `deduplication_rate` < 20%

**Solutions**:
- Decrease `dedup_iou_threshold` to 0.5 or 0.6
- Increase `dedup_max_frame_gap` to 8 or 10
- Increase `dedup_max_distance` to 75 or 100

### Performance Issues

**Symptoms**: Processing takes much longer than before

**Solutions**:
- Reduce `dedup_max_frame_gap` to 3
- Increase `frame_skip` in video processing (process fewer frames)
- Disable deduplication for real-time applications

## Technical Details

### IoU Calculation

Intersection over Union measures bounding box overlap:

```
IoU = Area of Intersection / Area of Union

Where:
- Intersection = overlapping region of two bboxes
- Union = total area covered by both bboxes
```

**Example**:
- IoU = 1.0: Perfect overlap (identical boxes)
- IoU = 0.7: 70% overlap (default threshold)
- IoU = 0.0: No overlap

### Distance Calculation

Euclidean distance between bounding box centers:

```
distance = sqrt((x1 - x2)² + (y1 - y2)²)
```

This provides a fallback when IoU is low but boxes are close (e.g., slight shifts between frames).

## Best Practices

1. **Start with defaults**: The default configuration works well for most scenarios
2. **Monitor statistics**: Check `deduplication_rate` to ensure reasonable removal (40-70%)
3. **Use stability scores**: Filter results by stability for high-confidence plates
4. **Keep duplicates available**: The `duplicate_detections` list allows post-processing if needed
5. **Test with sample videos**: Adjust thresholds based on your specific video characteristics

## Future Enhancements

Potential improvements for future versions:

- Per-request deduplication parameters (override config)
- Adaptive thresholds based on video characteristics
- Machine learning-based duplicate detection
- Real-time deduplication for streaming video
- Duplicate grouping visualization
- Historical comparison for repeated vehicles

## References

- **IoU**: [Intersection over Union on Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)
- **Object Tracking**: Common technique in video analysis to handle temporal redundancy
- **YOLOv11**: Detection model used for license plate detection
