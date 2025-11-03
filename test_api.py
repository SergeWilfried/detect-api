"""
Simple test script for the License Plate Detection API
Run this after starting the API server
"""
import requests
import json


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_root():
    """Test root endpoint"""
    print("Testing / endpoint...")
    response = requests.get("http://localhost:8000/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


if __name__ == "__main__":
    try:
        test_health()
        test_root()
        print("✅ Basic API tests passed!")
        print("\n💡 To test detection, use example_usage.py or visit http://localhost:8000/docs")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("💡 Make sure the API is running: uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")

