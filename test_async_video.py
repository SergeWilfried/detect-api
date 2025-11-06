#!/usr/bin/env python3
"""Test script for async video processing endpoint"""
import requests
import time
import json

# Test the async video upload endpoint
print("Testing async video upload endpoint...")
print("=" * 50)

# Upload video file
video_file = "files/2.mp4"
print(f"\n1. Uploading video: {video_file}")

try:
    with open(video_file, 'rb') as f:
        files = {'file': (video_file, f, 'video/mp4')}
        data = {
            'frame_skip': 10,
            'min_confidence': 0.25
        }
        
        response = requests.post(
            'http://localhost:8000/process/video/upload/async',
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            job_data = response.json()
            print(f"[OK] Job created successfully!")
            print(f"  Job ID: {job_data['job_id']}")
            print(f"  Status: {job_data['status']}")
            print(f"  Message: {job_data['message']}")
            print(f"  Status URL: {job_data['status_url']}")
            
            job_id = job_data['job_id']
            
            # Poll for status
            print(f"\n2. Polling job status...")
            max_polls = 60  # Max 60 polls (2 minutes)
            poll_count = 0
            status = None
            
            while poll_count < max_polls:
                status_response = requests.get(f'http://localhost:8000/jobs/{job_id}')
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    progress = status.get('progress', 0) * 100
                    print(f"  Status: {status['status']} | Progress: {progress:.1f}% | {status.get('message', '')}")
                    
                    if status['status'] == 'completed':
                        print(f"\n[OK] Job completed!")
                        break
                    elif status['status'] == 'failed':
                        print(f"\n[ERROR] Job failed: {status.get('error', 'Unknown error')}")
                        break
                else:
                    print(f"  Error getting status: {status_response.status_code}")
                    break
                
                time.sleep(2)  # Poll every 2 seconds
                poll_count += 1
            
            # Get result if completed
            if status and status['status'] == 'completed':
                print(f"\n3. Retrieving results...")
                result_response = requests.get(f'http://localhost:8000/jobs/{job_id}/result')
                
                if result_response.status_code == 200:
                    result = result_response.json()
                    print(f"[OK] Results retrieved!")
                    print(f"  Processed frames: {result['statistics']['processed_frames']}")
                    print(f"  Total detections: {result['statistics']['total_detections']}")
                    print(f"  Unique plates: {result['statistics']['unique_plates']}")
                    print(f"  Processing time: {result['statistics']['processing_time_seconds']:.2f}s")
                    
                    if result['plate_summaries']:
                        print(f"\n  Plate summaries:")
                        for plate in result['plate_summaries'][:5]:  # Show first 5
                            print(f"    - {plate['plate_text']}: {plate['total_occurrences']} occurrences")
                else:
                    print(f"[ERROR] Error getting result: {result_response.status_code}")
                    print(f"  {result_response.text}")
        else:
            print(f"[ERROR] Error creating job: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"  Error detail: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"  Response: {response.text}")
            
except FileNotFoundError:
    print(f"[ERROR] Video file not found: {video_file}")
except Exception as e:
    print(f"[ERROR] Error: {e}")

print("\n" + "=" * 50)
print("Test completed!")

