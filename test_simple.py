import requests
import json

# Test the endpoint with minimal data
print("Testing async endpoint...")

try:
    # Create a small test file
    with open("test_video.mp4", "wb") as f:
        f.write(b"fake video data")
    
    with open("test_video.mp4", "rb") as f:
        files = {'file': ('test.mp4', f, 'video/mp4')}
        data = {'frame_skip': 10}
        
        response = requests.post(
            'http://localhost:8000/process/video/upload/async',
            files=files,
            data=data,
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("SUCCESS!")
            print(json.dumps(response.json(), indent=2))
        else:
            try:
                error = response.json()
                print("Error details:")
                print(json.dumps(error, indent=2))
            except:
                print(f"Raw response: {response.text}")
                
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()

