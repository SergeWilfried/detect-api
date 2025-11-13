"""
Video processing service
Handles video processing logic and background tasks
"""
import cv2
import time
import os
from typing import Optional, Dict, Any, List
from collections import defaultdict
from detection_service import LicensePlateDetector
from services.storage_service import storage_service
from utils.deduplication import deduplicate_detections, analyze_plate_stability
from core.config import settings


def calculate_speed(occurrences, fps, pixels_per_meter=10.0, speed_limit_kph=60):
    """
    Calculate estimated speed from plate occurrences
    
    Args:
        occurrences: List of detection occurrences for same plate
        fps: Video frames per second
        pixels_per_meter: Calibration factor (pixels per real meter)
        speed_limit_kph: Speed limit for violation detection
    
    Returns:
        dict with speed_kph, speed_mph, is_violation, etc.
    """
    if len(occurrences) < 2:
        return None
    
    # Sort by frame number
    sorted_occ = sorted(occurrences, key=lambda x: x['frame_number'])
    
    speeds = []
    for i in range(1, len(sorted_occ)):
        prev = sorted_occ[i-1]
        curr = sorted_occ[i]
        
        # Calculate center of bounding boxes
        prev_center_y = (prev['bbox']['y1'] + prev['bbox']['y2']) / 2
        curr_center_y = (curr['bbox']['y1'] + curr['bbox']['y2']) / 2
        
        # Distance in pixels (vertical movement - car approaching/leaving)
        pixel_distance = abs(curr_center_y - prev_center_y)
        
        # Time elapsed
        time_elapsed = (curr['frame_number'] - prev['frame_number']) / fps
        
        if time_elapsed > 0:
            # Convert to real-world speed
            meters_traveled = pixel_distance / pixels_per_meter
            speed_mps = meters_traveled / time_elapsed
            speed_kph = speed_mps * 3.6  # m/s to km/h
            speeds.append(speed_kph)
    
    if not speeds:
        return None
    
    avg_speed_kph = sum(speeds) / len(speeds)
    avg_speed_mph = avg_speed_kph * 0.621371
    
    return {
        "estimated_speed_kph": round(avg_speed_kph, 2),
        "estimated_speed_mph": round(avg_speed_mph, 2),
        "is_violation": avg_speed_kph > speed_limit_kph,
        "speed_limit_kph": speed_limit_kph,
        "confidence": "low",  # Indicate this is a simple estimation
        "calibration_used": pixels_per_meter
    }


def process_video_background(
    job_id: str,
    video_path: str,
    detector: LicensePlateDetector,
    frame_skip: int,
    start_frame: Optional[int],
    end_frame: Optional[int],
    min_confidence: Optional[float]
):
    """
    Background task to process video (runs in background thread)

    Args:
        job_id: Unique job identifier
        video_path: Path to video file
        detector: LicensePlateDetector instance
        frame_skip: Process every Nth frame
        start_frame: Starting frame number (inclusive)
        end_frame: Ending frame number (inclusive)
        min_confidence: Minimum confidence threshold override
    """
    start_time = time.time()
    try:
        storage_service.update_job_status(job_id, "processing", progress=0.0, message="Opening video file...")

        # Open video with OpenCV
        video_cap = cv2.VideoCapture(video_path)

        if not video_cap.isOpened():
            storage_service.update_job_status(job_id, "failed", error="Could not open video file")
            return

        # Get video properties
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = total_frames / fps if fps > 0 else 0

        # Determine frame range
        start_frame_num = start_frame if start_frame is not None else 0
        end_frame_num = end_frame if end_frame is not None else total_frames - 1

        if start_frame_num < 0 or end_frame_num >= total_frames or start_frame_num > end_frame_num:
            video_cap.release()
            storage_service.update_job_status(job_id, "failed", error=f"Invalid frame range: {start_frame_num} to {end_frame_num}")
            return

        # Override confidence threshold if provided
        original_threshold = detector.confidence_threshold
        if min_confidence is not None:
            detector.confidence_threshold = min_confidence

        try:
            # Storage for all detections
            all_detections = []
            plate_data = defaultdict(list)

            processed_frames = 0
            frames_with_detections = 0
            frames_to_process = len(range(start_frame_num, end_frame_num + 1, frame_skip))

            # Process frames
            for idx, frame_num in enumerate(range(start_frame_num, end_frame_num + 1, frame_skip)):
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = video_cap.read()

                if not ret:
                    break

                # Detect license plates in this frame
                detections = detector.detect_license_plates(frame)

                processed_frames += 1

                if detections:
                    frames_with_detections += 1
                    timestamp = frame_num / fps if fps > 0 else 0

                    # Save annotated frame to MongoDB
                    annotated_base64 = detector.get_visualization(frame, detections)
                    frame_id = storage_service.save_annotated_frame(
                        frame=frame,
                        detections=detections,
                        frame_number=frame_num,
                        timestamp_seconds=timestamp,
                        job_id=job_id,
                        use_gridfs=False,
                        annotated_base64=annotated_base64
                    )

                    for det in detections:
                        plate_text = det.get("plate_text", "").strip()
                        if not plate_text:
                            plate_text = f"UNKNOWN_{len(all_detections)}"

                        occurrence_data = {
                            "frame_number": frame_num,
                            "timestamp_seconds": timestamp,
                            "confidence": det["confidence"],
                            "ocr_confidence": det.get("ocr_confidence"),
                            "bbox": det["bbox"],
                            "plate_text": plate_text,
                            "class_name": det["class_name"],
                            "frame_id": frame_id
                        }

                        # Save individual detection to database
                        detection_doc = {
                            "job_id": job_id,
                            "frame_number": frame_num,
                            "timestamp_seconds": timestamp,
                            "confidence": det["confidence"],
                            "ocr_confidence": det.get("ocr_confidence"),
                            "bbox": det["bbox"],
                            "plate_text": plate_text,
                            "class_name": det["class_name"],
                            "frame_id": frame_id,
                            "video_info": {
                                "total_frames": total_frames,
                                "fps": fps,
                                "resolution": {"width": width, "height": height}
                            }
                        }
                        detection_id = storage_service.save_detection(detection_doc, collection_name="detections")
                        if detection_id:
                            occurrence_data["detection_id"] = detection_id

                        all_detections.append(occurrence_data)
                        plate_data[plate_text].append(occurrence_data)

                # Update progress every 10 frames or every 1%
                if processed_frames % 10 == 0 or (idx + 1) % max(1, frames_to_process // 100) == 0:
                    progress = (idx + 1) / frames_to_process
                    unique_plates_count = len(set(det.get("plate_text", "").strip() for det in all_detections if det.get("plate_text", "").strip()))
                    storage_service.update_job_status(
                        job_id,
                        "processing",
                        progress=progress,
                        message=f"Processed {processed_frames}/{frames_to_process} frames ({unique_plates_count} unique plates, {len(all_detections)} total detections)"
                    )
                    # Small sleep to prevent blocking
                    time.sleep(0.01)

        finally:
            detector.confidence_threshold = original_threshold
            video_cap.release()

        # === DEDUPLICATION STEP ===
        # Remove duplicate detections using smart detection-based filtering
        if settings.enable_deduplication:
            unique_detections, duplicate_detections, dedup_stats = deduplicate_detections(
                all_detections,
                iou_threshold=settings.dedup_iou_threshold,
                max_frame_gap=settings.dedup_max_frame_gap,
                max_distance=settings.dedup_max_distance,
                keep_strategy=settings.dedup_keep_strategy
            )
        else:
            # No deduplication - all detections are unique
            unique_detections = all_detections
            duplicate_detections = []
            dedup_stats = {
                "total_detections": len(all_detections),
                "unique_detections": len(all_detections),
                "duplicate_detections": 0,
                "deduplication_rate": 0.0,
                "kept_strategy": "none",
                "config": {"enabled": False}
            }

        print(f"Deduplication: {len(all_detections)} → {len(unique_detections)} detections "
              f"({dedup_stats['duplicate_detections']} duplicates removed, "
              f"{dedup_stats['deduplication_rate']:.1f}% dedup rate)")

        # Rebuild plate_data with deduplicated detections
        plate_data_deduplicated = defaultdict(list)
        for det in unique_detections:
            plate_text = det.get("plate_text", "").strip()
            if not plate_text:
                plate_text = f"UNKNOWN_{det['frame_number']}"
            plate_data_deduplicated[plate_text].append(det)

        # Calculate statistics (using deduplicated data)
        processing_time = time.time() - start_time
        total_detections = len(all_detections)  # Original count
        unique_detections_count = len(unique_detections)  # After deduplication
        unique_plates = len(plate_data_deduplicated)
        detection_rate = total_detections / video_duration if video_duration > 0 else 0
        avg_fps = processed_frames / processing_time if processing_time > 0 else None

        # Generate plate summaries (using deduplicated data)
        plate_summaries = []
        for plate_text, occurrences in plate_data_deduplicated.items():
            confidences = [occ["confidence"] for occ in occurrences]
            ocr_confidences = [occ["ocr_confidence"] for occ in occurrences if occ["ocr_confidence"] is not None]
            frame_numbers = [occ["frame_number"] for occ in occurrences]

            # Analyze plate stability
            stability_metrics = analyze_plate_stability(occurrences)

            # Calculate speed if enabled
            speed_data = None
            is_violation = False
            if settings.enable_speed_detection:
                speed_data = calculate_speed(
                    occurrences,
                    fps,
                    pixels_per_meter=settings.pixels_per_meter,
                    speed_limit_kph=settings.speed_limit_kph
                )
                if speed_data:
                    is_violation = speed_data['is_violation']

            plate_summaries.append({
                "plate_text": plate_text,
                "total_occurrences": len(occurrences),
                "first_seen_frame": min(frame_numbers),
                "last_seen_frame": max(frame_numbers),
                "first_seen_timestamp": min(occ["timestamp_seconds"] for occ in occurrences),
                "last_seen_timestamp": max(occ["timestamp_seconds"] for occ in occurrences),
                "average_confidence": sum(confidences) / len(confidences),
                "average_ocr_confidence": sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None,
                "frames_with_detection": sorted(frame_numbers),
                "occurrences": occurrences,
                "stability": stability_metrics,  # NEW: Stability analysis
                "speed_analysis": speed_data,  # NEW: Speed analysis
                "is_violation": is_violation  # NEW: Violation flag
            })

        # Sort by total occurrences
        plate_summaries.sort(key=lambda x: x["total_occurrences"], reverse=True)

        # Prepare result
        result = {
            "success": True,
            "message": f"Processed {processed_frames} frames. Found {unique_plates} unique license plate(s) with {unique_detections_count} unique detection(s) ({total_detections} total, {dedup_stats['duplicate_detections']} duplicates removed).",
            "video_info": {
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "resolution": {"width": width, "height": height},
                "duration_seconds": round(video_duration, 2)
            },
            "statistics": {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "frames_with_detections": frames_with_detections,
                "total_detections": total_detections,  # Original count
                "unique_detections": unique_detections_count,  # After deduplication
                "duplicate_detections": dedup_stats['duplicate_detections'],
                "deduplication_rate": dedup_stats['deduplication_rate'],
                "unique_plates": unique_plates,
                "video_duration_seconds": round(video_duration, 2),
                "processing_time_seconds": round(processing_time, 2),
                "average_fps": round(avg_fps, 2) if avg_fps else None,
                "detection_rate": round(detection_rate, 2)
            },
            "deduplication": dedup_stats,  # NEW: Full deduplication statistics
            "plate_summaries": plate_summaries,
            "all_detections": unique_detections,  # Return only unique detections
            "duplicate_detections": duplicate_detections,  # NEW: Separate list of duplicates
            "processing_parameters": {
                "frame_skip": frame_skip,
                "start_frame": start_frame_num,
                "end_frame": end_frame_num,
                "confidence_threshold": min_confidence if min_confidence is not None else detector.confidence_threshold,
                "deduplication_enabled": settings.enable_deduplication,
                "deduplication_config": dedup_stats['config']
            }
        }

        # Save result to MongoDB
        storage_service.save_video_result(job_id, result)

        # Update job status
        storage_service.update_job_status(job_id, "completed", progress=1.0, message="Processing completed", result=result)

    except Exception as e:
        error_msg = str(e)
        print(f"⚠ Error processing video job {job_id}: {error_msg}")
        storage_service.update_job_status(job_id, "failed", error=error_msg)
    finally:
        # Clean up temporary file
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except:
                pass
