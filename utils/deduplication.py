"""
Frame and detection deduplication utilities
Implements detection-based deduplication to filter duplicate detections
"""
from typing import List, Dict, Any, Tuple
import math


def calculate_bbox_overlap(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        bbox1: First bounding box {x1, y1, x2, y2, width, height}
        bbox2: Second bounding box {x1, y1, x2, y2, width, height}

    Returns:
        IoU value between 0.0 and 1.0
    """
    # Get coordinates
    x1_min, y1_min = bbox1["x1"], bbox1["y1"]
    x1_max, y1_max = bbox1["x2"], bbox1["y2"]
    x2_min, y2_min = bbox2["x1"], bbox2["y1"]
    x2_max, y2_max = bbox2["x2"], bbox2["y2"]

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_bbox_distance(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
    """
    Calculate Euclidean distance between bbox centers

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        Distance in pixels
    """
    # Calculate centers
    center1_x = (bbox1["x1"] + bbox1["x2"]) / 2
    center1_y = (bbox1["y1"] + bbox1["y2"]) / 2
    center2_x = (bbox2["x1"] + bbox2["x2"]) / 2
    center2_y = (bbox2["y1"] + bbox2["y2"]) / 2

    # Euclidean distance
    distance = math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    return distance


def is_duplicate_detection(
    det1: Dict[str, Any],
    det2: Dict[str, Any],
    iou_threshold: float = 0.7,
    max_frame_gap: int = 5,
    max_distance: float = 50.0
) -> bool:
    """
    Determine if two detections are duplicates

    Criteria for duplicates:
    - Same plate text (or both unknown)
    - Close frame numbers (within max_frame_gap)
    - High bbox overlap (IoU > iou_threshold) OR small distance

    Args:
        det1: First detection
        det2: Second detection
        iou_threshold: Minimum IoU to consider as duplicate (default: 0.7)
        max_frame_gap: Maximum frames apart to consider (default: 5)
        max_distance: Maximum pixel distance between centers (default: 50)

    Returns:
        True if detections are likely duplicates
    """
    # Check plate text match
    plate1 = det1.get("plate_text", "").strip()
    plate2 = det2.get("plate_text", "").strip()

    # If both have text and it's different, not a duplicate
    if plate1 and plate2 and plate1 != plate2:
        return False

    # Check frame proximity
    frame_gap = abs(det1["frame_number"] - det2["frame_number"])
    if frame_gap > max_frame_gap:
        return False

    # Check bbox overlap
    iou = calculate_bbox_overlap(det1["bbox"], det2["bbox"])
    if iou >= iou_threshold:
        return True

    # Check bbox distance as fallback
    distance = calculate_bbox_distance(det1["bbox"], det2["bbox"])
    if distance <= max_distance:
        return True

    return False


def deduplicate_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.7,
    max_frame_gap: int = 5,
    max_distance: float = 50.0,
    keep_strategy: str = "highest_confidence"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Remove duplicate detections from a list

    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for duplicates
        max_frame_gap: Max frames apart to consider duplicates
        max_distance: Max pixel distance for duplicates
        keep_strategy: Which detection to keep when duplicates found
                      - "highest_confidence": Keep detection with highest confidence
                      - "first": Keep first detection
                      - "last": Keep last detection

    Returns:
        Tuple of:
        - unique_detections: List of unique detections
        - duplicate_detections: List of detected duplicates
        - stats: Statistics about deduplication
    """
    if not detections:
        return [], [], {
            "total_detections": 0,
            "unique_detections": 0,
            "duplicate_detections": 0,
            "deduplication_rate": 0.0
        }

    # Sort by frame number for efficient comparison
    sorted_detections = sorted(detections, key=lambda x: x["frame_number"])

    unique_detections = []
    duplicate_detections = []
    marked_as_duplicate = set()

    for i, det1 in enumerate(sorted_detections):
        if i in marked_as_duplicate:
            continue

        # Find all duplicates of this detection
        duplicates_of_current = [det1]

        # Only check nearby frames (optimization)
        for j in range(i + 1, len(sorted_detections)):
            det2 = sorted_detections[j]

            # Stop if frames are too far apart
            if det2["frame_number"] - det1["frame_number"] > max_frame_gap:
                break

            if j not in marked_as_duplicate:
                if is_duplicate_detection(det1, det2, iou_threshold, max_frame_gap, max_distance):
                    duplicates_of_current.append(det2)
                    marked_as_duplicate.add(j)

        # Choose which detection to keep based on strategy
        if keep_strategy == "highest_confidence":
            kept_detection = max(duplicates_of_current, key=lambda x: x["confidence"])
        elif keep_strategy == "first":
            kept_detection = duplicates_of_current[0]
        elif keep_strategy == "last":
            kept_detection = duplicates_of_current[-1]
        else:
            kept_detection = duplicates_of_current[0]

        unique_detections.append(kept_detection)

        # Add the rest as duplicates
        for dup in duplicates_of_current:
            if dup is not kept_detection:
                duplicate_detections.append(dup)

    # Calculate statistics
    total = len(detections)
    unique = len(unique_detections)
    duplicates = len(duplicate_detections)
    dedup_rate = (duplicates / total * 100) if total > 0 else 0.0

    stats = {
        "total_detections": total,
        "unique_detections": unique,
        "duplicate_detections": duplicates,
        "deduplication_rate": round(dedup_rate, 2),
        "kept_strategy": keep_strategy,
        "config": {
            "iou_threshold": iou_threshold,
            "max_frame_gap": max_frame_gap,
            "max_distance": max_distance
        }
    }

    return unique_detections, duplicate_detections, stats


def analyze_plate_stability(occurrences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze stability of plate detections (for quality scoring)

    Args:
        occurrences: List of occurrences for a single plate

    Returns:
        Stability metrics
    """
    if not occurrences:
        return {
            "is_stable": False,
            "stability_score": 0.0,
            "bbox_variance": 0.0,
            "confidence_variance": 0.0
        }

    # Calculate bbox center variance
    centers_x = [(occ["bbox"]["x1"] + occ["bbox"]["x2"]) / 2 for occ in occurrences]
    centers_y = [(occ["bbox"]["y1"] + occ["bbox"]["y2"]) / 2 for occ in occurrences]

    mean_x = sum(centers_x) / len(centers_x)
    mean_y = sum(centers_y) / len(centers_y)

    variance_x = sum((x - mean_x)**2 for x in centers_x) / len(centers_x)
    variance_y = sum((y - mean_y)**2 for y in centers_y) / len(centers_y)
    bbox_variance = math.sqrt(variance_x + variance_y)

    # Calculate confidence variance
    confidences = [occ["confidence"] for occ in occurrences]
    mean_conf = sum(confidences) / len(confidences)
    conf_variance = sum((c - mean_conf)**2 for c in confidences) / len(confidences)

    # Stability score (lower variance = higher stability)
    # Normalize to 0-1 scale
    position_stability = max(0.0, 1.0 - (bbox_variance / 100.0))  # Assume 100px variance = unstable
    confidence_stability = max(0.0, 1.0 - (conf_variance * 10))  # Scale variance

    stability_score = (position_stability + confidence_stability) / 2
    is_stable = stability_score > 0.6  # Threshold for "stable"

    return {
        "is_stable": is_stable,
        "stability_score": round(stability_score, 3),
        "bbox_variance": round(bbox_variance, 2),
        "confidence_variance": round(conf_variance, 3),
        "position_stability": round(position_stability, 3),
        "confidence_stability": round(confidence_stability, 3)
    }
