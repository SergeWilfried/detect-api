"""
Test script for video detection endpoint
Run this after starting the API server to test video frame processing
"""
import requests
import json
import base64
from pathlib import Path


def test_video_endpoint(frame_number=None):
    """Test the video detection endpoint"""
    url = "http://localhost:8000/detect/video"
    
    params = {}
    if frame_number is not None:
        params["frame_number"] = frame_number
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        
        print("[OK] Successfully processed frame")
        print(f"  Message: {result['message']}")
        print(f"  Detections: {result['count']}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        
        if result['detections']:
            print("\n  Detection details:")
            for i, det in enumerate(result['detections'], 1):
                print(f"    {i}. {det['class_name']}: {det['confidence']:.4f}")
                bbox = det['bbox']
                print(f"       BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) -> ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
        
        # Save visualization if available
        if result.get('visualization'):
            img_data = base64.b64decode(result['visualization'])
            output_path = f"frame_{frame_number or 'next'}_detection.png"
            with open(output_path, "wb") as f:
                f.write(img_data)
            print(f"\n  Visualization saved to: {output_path}")
        
        return result
        
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to API")
        print("[INFO] Make sure the API is running: uvicorn main:app --reload")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP Error: {e}")
        if e.response.status_code == 404:
            print(f"[INFO] Video file not found. Make sure {VIDEO_PATH} exists")
        return None
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return None


if __name__ == "__main__":
    VIDEO_PATH = "./files/deneme.mp4"
    
    print("Testing Video Detection Endpoint")
    print("=" * 50)
    
    # Check if video exists
    if not Path(VIDEO_PATH).exists():
        print(f"[WARNING] Video file not found at {VIDEO_PATH}")
        print("   The endpoint will return an error if the file doesn't exist")
    else:
        print(f"[OK] Video file found: {VIDEO_PATH}\n")
    
    # Test 1: Process first frame
    print("Test 1: Processing frame 0")
    print("-" * 50)
    test_video_endpoint(frame_number=0)
    
    print("\n" + "=" * 50 + "\n")
    
    # Test 2: Process frame 100 (if video is long enough)
    print("Test 2: Processing frame 100")
    print("-" * 50)
    test_video_endpoint(frame_number=100)
    
    print("\n" + "=" * 50 + "\n")
    
    # Test 3: Process next frame (sequential)
    print("Test 3: Processing next frame")
    print("-" * 50)
    test_video_endpoint()

