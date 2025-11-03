# License Plate Detection API

License plate detection API using YOLOv11, based on the implementation from the Medium article: [License Plate Recognition using YOLO and Custom Dataset Visualization](https://medium.com/@1032211306/license-plate-recognition-using-yolo-and-custom-dataset-visualization-886994c85d61)

## Features

- 🚗 License plate detection using YOLOv11
- 📸 Multiple input methods: base64, image URL, or file upload
- 🎨 Optional visualization with bounding boxes
- ⚡ Fast inference with ultralytics
- 📊 Detailed detection results with confidence scores
- 🔧 Easy to extend with custom trained models

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The YOLOv11 model will be downloaded automatically on first run.

## Usage

### Start the API server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Detect from Base64 or URL
```bash
POST /detect
Content-Type: application/json

{
  "image_url": "https://example.com/car.jpg",
  "include_visualization": true
}
```

Or with base64:
```json
{
  "data": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "include_visualization": true
}
```

#### 3. Detect from File Upload
```bash
POST /detect/upload
Content-Type: multipart/form-data

file: [image file]
include_visualization: true
```

#### 4. Detect from Test Video File
```bash
GET /detect/video?frame_number=0

# Process specific frame
GET /detect/video?frame_number=100

# Process next frame (no parameter)
GET /detect/video
```
Note: This endpoint uses the video file at `./files/deneme.mp4` and optional model at `./models/license_plate_detector.pt` loaded at startup.

### Response Format

```json
{
  "detected": true,
  "count": 2,
  "detections": [
    {
      "class_name": "car",
      "confidence": 0.8567,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 350.2,
        "y2": 280.1,
        "width": 249.7,
        "height": 79.8
      }
    }
  ],
  "message": "Found 2 detection(s)",
  "confidence": 0.8567,
  "image_shape": {
    "height": 720,
    "width": 1280
  },
  "visualization": "base64_encoded_image..."
}
```

## Using a Custom Trained Model

To use your own license plate detection model trained on custom data:

1. Train your model using YOLOv11 (see the Medium article for training details)
2. Set the environment variable:
```bash
export YOLO_MODEL_PATH=/path/to/your/model.pt
export CONFIDENCE_THRESHOLD=0.5
```

3. Restart the API

## Test Resources

The API automatically loads test resources on startup if they exist:

- **Model**: `./models/license_plate_detector.pt` - Custom trained license plate model
- **Video**: `./files/deneme.mp4` - Test video file for frame-by-frame detection

If these files are not found, the API will still work using the default pretrained model. Check the root endpoint `/` to see which resources are loaded.

## Environment Variables

- `YOLO_MODEL_PATH`: Path to custom YOLO model file (optional, overrides default path)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections (default: 0.25)

## Training Your Own Model

For best results with license plates, you should:

1. **Prepare your dataset**: Collect and annotate license plate images
2. **Format annotations**: Use YOLO format (class x_center y_center width height)
3. **Train the model**: Use ultralytics training scripts
4. **Export the model**: Save as `.pt` file
5. **Update code**: Filter detections by class name "license_plate"

Example training command:
```bash
yolo train data=license_plates.yaml model=yolo11n.pt epochs=100
```

## Project Structure

```
.
├── main.py                 # FastAPI application
├── detection_service.py    # YOLO detection service
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Notes

- The current implementation uses YOLOv11 pretrained model which detects general objects
- For production license plate detection, train a custom model on license plate datasets
- The visualization feature draws bounding boxes on detected objects
- All image formats supported by PIL/OpenCV are accepted

## License

This project implements concepts from the Medium article mentioned above.

