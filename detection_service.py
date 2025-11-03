"""
License Plate Detection Service using YOLO
Based on the Medium article: License Plate Recognition using YOLO and Custom Dataset Visualization
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ultralytics import YOLO
from PIL import Image
import io
import requests
import base64
import os


class LicensePlateDetector:
    """License Plate Detection using YOLOv11"""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25):
        """
        Initialize the license plate detector
        
        Args:
            model_path: Path to custom YOLO model. If None, uses pretrained YOLOv11 model
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                print(f"Loaded custom model from {model_path}")
            except Exception as e:
                print(f"Error loading custom model {model_path}: {e}")
                print("Falling back to pretrained model")
                self.model = YOLO("yolo11n.pt")
                print("Loaded YOLOv11 pretrained model (fallback)")
        else:
            # Use YOLOv11 pretrained model (will be downloaded automatically)
            # For license plates, we'll use a general object detection model
            # In production, you should train a custom model on license plate dataset
            try:
                self.model = YOLO("yolo11n.pt")  # Nano model for faster inference
                print("Loaded YOLOv11 pretrained model")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        
        # Class names for COCO dataset (YOLOv11 default)
        # For license plate detection, you would train a custom model
        # with class "license_plate" or similar
        self.class_names = self.model.names
    
    def load_image_from_url(self, url: str) -> Optional[np.ndarray]:
        """Load image from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            # Convert PIL to OpenCV format
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error loading image from URL: {e}")
            return None
    
    def load_image_from_base64(self, data: str) -> Optional[np.ndarray]:
        """Load image from base64 encoded string"""
        try:
            # Remove data URL prefix if present
            if ',' in data:
                data = data.split(',')[1]
            
            image_data = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_data))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error loading image from base64: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for detection
        Can include resizing, normalization, etc.
        """
        # YOLO handles preprocessing internally, but we can add custom preprocessing here
        return image
    
    def detect_license_plates(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect license plates in the image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with bounding boxes, confidence, etc.
        """
        if image is None:
            return []
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Run YOLO inference
        results = self.model(processed_image, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                # For license plate detection, you would filter by class_name == "license_plate"
                # For now, we return all detections (in production, use a custom trained model)
                detection = {
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1)
                    }
                }
                detections.append(detection)
        
        return detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image
        
        Args:
            image: Original image
            detections: List of detection dictionaries
            
        Returns:
            Image with bounding boxes drawn
        """
        annotated_image = image.copy()
        
        for det in detections:
            bbox = det["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            confidence = det["confidence"]
            class_name = det["class_name"]
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, label_size[1])
            
            # Draw label background
            cv2.rectangle(
                annotated_image,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return annotated_image
    
    def detect_from_url(self, image_url: str) -> Dict[str, Any]:
        """Detect license plates from image URL"""
        image = self.load_image_from_url(image_url)
        if image is None:
            return {
                "detected": False,
                "detections": [],
                "count": 0,
                "error": "Failed to load image from URL"
            }
        
        detections = self.detect_license_plates(image)
        
        return {
            "detected": len(detections) > 0,
            "detections": detections,
            "count": len(detections),
            "image_shape": {
                "height": image.shape[0],
                "width": image.shape[1]
            }
        }
    
    def detect_from_base64(self, image_data: str) -> Dict[str, Any]:
        """Detect license plates from base64 encoded image"""
        image = self.load_image_from_base64(image_data)
        if image is None:
            return {
                "detected": False,
                "detections": [],
                "count": 0,
                "error": "Failed to load image from base64"
            }
        
        detections = self.detect_license_plates(image)
        
        return {
            "detected": len(detections) > 0,
            "detections": detections,
            "count": len(detections),
            "image_shape": {
                "height": image.shape[0],
                "width": image.shape[1]
            }
        }
    
    def get_visualization(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> bytes:
        """
        Get visualization as base64 encoded image
        
        Args:
            image: Original image
            detections: List of detections
            
        Returns:
            Base64 encoded image bytes
        """
        annotated_image = self.visualize_detections(image, detections)
        
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        
        return base64.b64encode(img_bytes).decode('utf-8')

