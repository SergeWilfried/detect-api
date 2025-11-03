"""
Example usage of the License Plate Detection API
"""
import requests
import base64
import json
from pathlib import Path


def detect_from_url(image_url: str, include_viz: bool = True):
    """Example: Detect license plates from image URL"""
    url = "http://localhost:8000/detect"
    
    payload = {
        "image_url": image_url,
        "include_visualization": include_viz
    }
    
    response = requests.post(url, json=payload)
    return response.json()


def detect_from_base64(image_path: str, include_viz: bool = True):
    """Example: Detect license plates from local image file (base64)"""
    url = "http://localhost:8000/detect"
    
    # Read and encode image
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    payload = {
        "data": f"data:image/jpeg;base64,{encoded_string}",
        "include_visualization": include_viz
    }
    
    response = requests.post(url, json=payload)
    return response.json()


def detect_from_upload(image_path: str, include_viz: bool = True):
    """Example: Detect license plates from file upload"""
    url = "http://localhost:8000/detect/upload"
    
    files = {
        'file': open(image_path, 'rb')
    }
    
    data = {
        'include_visualization': include_viz
    }
    
    response = requests.post(url, files=files, data=data)
    files['file'].close()
    return response.json()


def save_visualization(response: dict, output_path: str = "visualization.png"):
    """Save the visualization image from response"""
    if response.get("visualization"):
        img_data = base64.b64decode(response["visualization"])
        with open(output_path, "wb") as f:
            f.write(img_data)
        print(f"Visualization saved to {output_path}")
    else:
        print("No visualization in response")


if __name__ == "__main__":
    # Make sure the API is running on localhost:8000
    
    # Example 1: Detect from URL
    print("Example 1: Detecting from URL...")
    try:
        result = detect_from_url(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Car_with_license_plate.jpg/800px-Car_with_license_plate.jpg",
            include_viz=True
        )
        print(json.dumps(result, indent=2))
        
        if result.get("visualization"):
            save_visualization(result, "detection_url.png")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure the API is running and the URL is accessible")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Detect from local file (base64)
    print("Example 2: Detecting from local file (base64)...")
    # Replace with your image path
    image_path = "files/deneme.mp4"  # Update this path
    if Path(image_path).exists():
        try:
            result = detect_from_base64(image_path, include_viz=True)
            print(json.dumps(result, indent=2))
            save_visualization(result, "detection_base64.png")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Image file not found: {image_path}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Detect from file upload
    print("Example 3: Detecting from file upload...")
    # Replace with your image path
    image_path = "files/test_image.jpg"  # Update this path
    if Path(image_path).exists():
        try:
            result = detect_from_upload(image_path, include_viz=True)
            print(json.dumps(result, indent=2))
            save_visualization(result, "detection_upload.png")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Image file not found: {image_path}")

