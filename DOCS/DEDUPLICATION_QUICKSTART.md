# Deduplication Quick Start

## What is it?

Removes duplicate license plate detections from video processing results while keeping the best quality detection.

## Quick Setup

Add to your `.env` file:

```bash
# Enable deduplication (default: true)
ENABLE_DEDUPLICATION=true

# IoU overlap threshold 0-1 (default: 0.7)
DEDUP_IOU_THRESHOLD=0.7

# Max frames apart to consider duplicates (default: 5)
DEDUP_MAX_FRAME_GAP=5

# Max pixel distance between centers (default: 50.0)
DEDUP_MAX_DISTANCE=50.0

# Keep strategy: highest_confidence, first, last (default: highest_confidence)
DEDUP_KEEP_STRATEGY=highest_confidence
```

## Common Scenarios

### Default (Balanced)
```bash
DEDUP_IOU_THRESHOLD=0.7
DEDUP_MAX_FRAME_GAP=5
DEDUP_MAX_DISTANCE=50.0
```
Good for most videos. Removes ~40-70% duplicates.

### Aggressive (Remove More)
```bash
DEDUP_IOU_THRESHOLD=0.5
DEDUP_MAX_FRAME_GAP=10
DEDUP_MAX_DISTANCE=100.0
```
Use when you have many duplicate detections.

### Conservative (Keep More)
```bash
DEDUP_IOU_THRESHOLD=0.9
DEDUP_MAX_FRAME_GAP=2
DEDUP_MAX_DISTANCE=25.0
```
Use when you want to preserve more detections.

### Disabled
```bash
ENABLE_DEDUPLICATION=false
```
No deduplication - all detections returned.

## Response Structure

```json
{
  "statistics": {
    "total_detections": 1000,        // Before deduplication
    "unique_detections": 340,        // After deduplication
    "duplicate_detections": 660,     // Removed
    "deduplication_rate": 66.0       // Percentage removed
  },
  "deduplication": {
    "total_detections": 1000,
    "unique_detections": 340,
    "duplicate_detections": 660,
    "deduplication_rate": 66.0,
    "kept_strategy": "highest_confidence",
    "config": {
      "iou_threshold": 0.7,
      "max_frame_gap": 5,
      "max_distance": 50.0
    }
  },
  "all_detections": [/* 340 unique detections */],
  "duplicate_detections": [/* 660 duplicates */]
}
```

## Stability Scores

Each plate includes a stability score (0-1):

```json
{
  "plate_summaries": [
    {
      "plate_text": "ABC123",
      "stability": {
        "is_stable": true,           // Score > 0.6
        "stability_score": 0.892,    // 0-1 (higher = better)
        "bbox_variance": 12.34,      // Position variance
        "confidence_variance": 0.002 // Confidence variance
      }
    }
  ]
}
```

**Interpreting Scores:**
- 0.8-1.0: Excellent quality
- 0.6-0.8: Good quality
- 0.4-0.6: Fair quality
- 0.0-0.4: Poor quality

## Troubleshooting

### Too many removed?
Increase `DEDUP_IOU_THRESHOLD` to 0.9

### Not enough removed?
Decrease `DEDUP_IOU_THRESHOLD` to 0.5

### Processing too slow?
Decrease `DEDUP_MAX_FRAME_GAP` to 3

## Learn More

See [DEDUPLICATION_GUIDE.md](DEDUPLICATION_GUIDE.md) for complete documentation.
