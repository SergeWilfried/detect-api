# License Plate Detection API

License plate detection API using YOLOv11, based on the implementation from the Medium article: [License Plate Recognition using YOLO and Custom Dataset Visualization](https://medium.com/@1032211306/license-plate-recognition-using-yolo-and-custom-dataset-visualization-886994c85d61)

## Features

- 🚗 License plate detection using YOLOv11
- 📝 **License plate number extraction using EasyOCR or Gemini**
- 🤖 **Gemini 2.0+ Enhanced Object Detection** - Advanced object detection with bounding boxes
- 🎯 **Gemini 2.5+ Segmentation** - Pixel-level segmentation masks
- 📸 Multiple input methods: base64, image URL, or file upload
- 🎨 Optional visualization with bounding boxes and extracted text
- ⚡ Fast inference with ultralytics
- 📊 Detailed detection results with confidence scores and OCR results
- 🎬 **Full video processing with comprehensive statistics**
- 📈 **Plate occurrence tracking and timeline analysis**
- 🔍 **Smart deduplication** - Removes duplicate detections across frames
- 📊 **Stability analysis** - Quality scoring for detected plates
- 🔧 Easy to extend with custom trained models

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (optional):
```bash
export GEMINI_API_KEY=your_api_key_here  # Required for Gemini features
export YOLO_MODEL_PATH=/path/to/model.pt  # Optional: custom YOLO model
export CONFIDENCE_THRESHOLD=0.25  # Optional: detection confidence threshold
export OCR_ENGINE=easyocr  # Optional: 'easyocr' or 'gemini'

# Deduplication settings (optional)
export ENABLE_DEDUPLICATION=true  # Enable/disable deduplication
export DEDUP_IOU_THRESHOLD=0.7  # Bbox overlap threshold (0-1)
export DEDUP_MAX_FRAME_GAP=5  # Max frames apart to consider duplicates
export DEDUP_MAX_DISTANCE=50.0  # Max pixel distance between centers
export DEDUP_KEEP_STRATEGY=highest_confidence  # highest_confidence, first, or last
```

3. The YOLOv11 model will be downloaded automatically on first run.

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

#### 5. Process Entire Video with Statistics
```bash
POST /process/video?frame_skip=5&start_frame=0&end_frame=500&min_confidence=0.3

# Process every 5th frame (faster processing)
POST /process/video?frame_skip=5

# Process all frames with custom confidence threshold
POST /process/video?frame_skip=1&min_confidence=0.5

# Process specific frame range
POST /process/video?frame_skip=3&start_frame=0&end_frame=200
```

**Parameters:**
- `frame_skip` (int, default: 1): Process every Nth frame (1 = all frames, 5 = every 5th frame)
- `start_frame` (int, optional): Start processing from this frame number
- `end_frame` (int, optional): Stop processing at this frame number
- `min_confidence` (float, optional): Override default confidence threshold

#### 6. Gemini Object Detection (Gemini 2.0+)
```bash
POST /gemini/detect
Content-Type: application/json

{
  "image_url": "https://example.com/car.jpg",
  "prompt": "Detect all license plates in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.",
  "include_visualization": true
}
```

Or with base64:
```json
{
  "data": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "prompt": "Detect all cars and license plates",
  "include_visualization": true
}
```

**File Upload:**
```bash
POST /gemini/detect/upload
Content-Type: multipart/form-data

file: [image file]
prompt: "Detect all green objects"
include_visualization: true
```

**Response:**
```json
{
  "detected": true,
  "count": 2,
  "detections": [
    {
      "label": "license plate",
      "box_2d": [100, 200, 350, 280],
      "box_2d_normalized": [200.0, 100.0, 280.0, 350.0],
      "confidence": 0.95
    }
  ],
  "message": "Found 2 detection(s)",
  "image_shape": {"width": 1280, "height": 720},
  "visualization": "base64_encoded_image..."
}
```

#### 7. Gemini Segmentation (Gemini 2.5+)
```bash
POST /gemini/segment
Content-Type: application/json

{
  "image_url": "https://example.com/car.jpg",
  "prompt": "Give the segmentation masks for all license plates in the image. Output a JSON list...",
  "include_visualization": true,
  "alpha": 0.5
}
```

**File Upload:**
```bash
POST /gemini/segment/upload
Content-Type: multipart/form-data

file: [image file]
prompt: "Segment all wooden and glass items"
include_visualization: true
alpha: 0.6
```

**Response:**
```json
{
  "detected": true,
  "count": 2,
  "segmentations": [
    {
      "label": "license plate",
      "box_2d": [100, 200, 350, 280],
      "box_2d_normalized": [200.0, 100.0, 280.0, 350.0],
      "confidence": 0.95,
      "mask_base64": "base64_encoded_mask_image..."
    }
  ],
  "message": "Found 2 segmentation(s)",
  "image_shape": {"width": 1280, "height": 720},
  "visualization": "base64_encoded_image_with_overlays..."
}
```

**Returns:** Comprehensive statistics including:
- Video information (frames, FPS, resolution, duration)
- Processing statistics (processed frames, detection counts, processing time)
- **Deduplication statistics** (unique vs duplicate detections, removal rate)
- Plate summaries (unique plates, occurrence counts, timestamps, confidence scores)
- **Stability metrics** (quality scoring for detected plates)
- All individual detections with frame numbers and timestamps
- Separate list of duplicate detections for audit purposes

**Example Response:**
```json
{
  "success": true,
  "message": "Processed 126 frames. Found 3 unique license plate(s) with 18 unique detection(s) (45 total, 27 duplicates removed).",
  "video_info": {
    "path": "./files/deneme.mp4",
    "total_frames": 631,
    "fps": 30.0,
    "resolution": {"width": 1920, "height": 1080},
    "duration_seconds": 21.03
  },
  "statistics": {
    "total_frames": 631,
    "processed_frames": 126,
    "frames_with_detections": 45,
    "total_detections": 45,
    "unique_detections": 18,
    "duplicate_detections": 27,
    "deduplication_rate": 60.0,
    "unique_plates": 3,
    "video_duration_seconds": 21.03,
    "processing_time_seconds": 15.32,
    "average_fps": 8.22,
    "detection_rate": 2.14
  },
  "deduplication": {
    "total_detections": 45,
    "unique_detections": 18,
    "duplicate_detections": 27,
    "deduplication_rate": 60.0,
    "kept_strategy": "highest_confidence",
    "config": {
      "iou_threshold": 0.7,
      "max_frame_gap": 5,
      "max_distance": 50.0
    }
  },
  "plate_summaries": [
    {
      "plate_text": "N-894JV",
      "total_occurrences": 6,
      "first_seen_frame": 100,
      "last_seen_frame": 250,
      "first_seen_timestamp": 3.33,
      "last_seen_timestamp": 8.33,
      "average_confidence": 0.7567,
      "average_ocr_confidence": 0.5534,
      "frames_with_detection": [100, 125, 150, 175, 200, 225],
      "occurrences": [...],
      "stability": {
        "is_stable": true,
        "stability_score": 0.892,
        "bbox_variance": 12.34,
        "confidence_variance": 0.002,
        "position_stability": 0.877,
        "confidence_stability": 0.908
      }
    }
  ],
  "all_detections": [/* 18 unique detections */],
  "duplicate_detections": [/* 27 duplicate detections */],
  "processing_parameters": {
    "frame_skip": 5,
    "start_frame": 0,
    "end_frame": 500,
    "confidence_threshold": 0.25,
    "deduplication_enabled": true,
    "deduplication_config": {
      "iou_threshold": 0.7,
      "max_frame_gap": 5,
      "max_distance": 50.0
    }
  }
}
```

### Response Format (Single Frame/Image)

```json
{
  "detected": true,
  "count": 2,
  "detections": [
    {
      "class_name": "License_Plate",
      "confidence": 0.8567,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 350.2,
        "y2": 280.1,
        "width": 249.7,
        "height": 79.8
      },
      "plate_text": "ABC123",
      "ocr_confidence": 0.9234
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

### Core Settings
- `GEMINI_API_KEY`: Google Gemini API key (required for Gemini detection/segmentation features)
- `YOLO_MODEL_PATH`: Path to custom YOLO model file (optional, overrides default path)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections (default: 0.25)
- `OCR_ENGINE`: OCR engine to use - 'easyocr' or 'gemini' (default: 'easyocr')

### Deduplication Settings
- `ENABLE_DEDUPLICATION`: Enable/disable deduplication (default: true)
- `DEDUP_IOU_THRESHOLD`: Bbox overlap threshold 0-1 (default: 0.7)
- `DEDUP_MAX_FRAME_GAP`: Max frames apart to consider duplicates (default: 5)
- `DEDUP_MAX_DISTANCE`: Max pixel distance between centers (default: 50.0)
- `DEDUP_KEEP_STRATEGY`: Strategy for keeping detections - 'highest_confidence', 'first', or 'last' (default: 'highest_confidence')

See [DEDUPLICATION_QUICKSTART.md](DEDUPLICATION_QUICKSTART.md) for detailed configuration guide.

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

## OCR Features

- **EasyOCR Integration**: Automatically extracts license plate numbers from detected regions
- **Gemini OCR**: Alternative OCR using Google Gemini Vision API (legacy support)
- **Text Preprocessing**: Automatically enhances license plate regions for better OCR accuracy
- **Confidence Scores**: Returns OCR confidence for each extracted plate number
- **Visualization**: Shows extracted text on annotated images

## Gemini Image Understanding Features

### Object Detection (Gemini 2.0+)
- **Enhanced Detection**: Uses Gemini 2.0+ models with improved accuracy for object detection
- **Custom Prompts**: Specify what objects to detect using natural language prompts
- **Bounding Boxes**: Returns normalized coordinates (0-1000) and absolute pixel coordinates
- **Flexible Input**: Supports images from URLs, base64, or file uploads

### Segmentation (Gemini 2.5+)
- **Pixel-Level Masks**: Get precise segmentation masks for detected objects
- **Custom Segmentation**: Use prompts to specify which objects to segment
- **Mask Visualization**: Overlay masks on original images with customizable transparency
- **Base64 Masks**: Receive individual mask images as base64-encoded PNG files

**Example Prompts:**
- `"Detect all license plates in the image"`
- `"Show bounding boxes of all green objects in this image"`
- `"Segment all cars and trucks"`
- `"Give the segmentation masks for wooden and glass items"`

## Notes

- The current implementation uses YOLOv11 pretrained model which detects general objects
- For production license plate detection, train a custom model on license plate datasets
- EasyOCR will download its models on first run (this may take a few minutes)
- OCR processing adds some overhead - expect 1-2 seconds per detection for text extraction
- The visualization feature draws bounding boxes and extracted text on detected objects
- All image formats supported by PIL/OpenCV are accepted

## License

This project implements concepts from the Medium article mentioned above.

