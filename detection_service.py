"""
License Plate Detection Service using YOLO and OCR (EasyOCR or Gemini)
Based on the Medium article: License Plate Recognition using YOLO and Custom Dataset Visualization
"""
# Prevent OpenCV from trying to load GUI libraries (libGL.so.1)
import os
os.environ.setdefault('OPENCV_DISABLE_LIBGL', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ultralytics import YOLO
from PIL import Image
import io
import requests
import base64
import os
import easyocr

# Optional Gemini import - will only be used if ocr_engine is set to 'gemini'
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# New Gemini API for object detection and segmentation
try:
    from google import genai as new_genai
    from google.genai import types
    NEW_GEMINI_AVAILABLE = True
except ImportError:
    NEW_GEMINI_AVAILABLE = False
    try:
        # Fallback check
        import google.genai as new_genai
        from google.genai import types
        NEW_GEMINI_AVAILABLE = True
    except ImportError:
        NEW_GEMINI_AVAILABLE = False


class LicensePlateDetector:
    """License Plate Detection using YOLOv11 and OCR (EasyOCR or Gemini)"""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25, 
                 enable_ocr: bool = True, ocr_languages: List[str] = ['en'],
                 ocr_engine: str = 'easyocr', gemini_api_key: Optional[str] = None):
        """
        Initialize the license plate detector
        
        Args:
            model_path: Path to custom YOLO model. If None, uses pretrained YOLOv11 model
            confidence_threshold: Minimum confidence score for detections
            enable_ocr: Whether to enable OCR for text extraction
            ocr_languages: List of languages for OCR (default: ['en']) - only used for EasyOCR
            ocr_engine: OCR engine to use - 'easyocr' or 'gemini' (default: 'easyocr')
            gemini_api_key: Google Gemini API key (required if ocr_engine is 'gemini')
        """
        self.confidence_threshold = confidence_threshold
        self.enable_ocr = enable_ocr
        self.ocr_engine = ocr_engine.lower() if enable_ocr else None
        
        if model_path:
            # Check if path exists (handle both relative and absolute paths)
            abs_path = os.path.abspath(model_path) if not os.path.isabs(model_path) else model_path
            if os.path.exists(abs_path) or os.path.exists(model_path):
                try:
                    self.model = YOLO(model_path)
                    print(f"Loaded custom model from {model_path}")
                    print(f"Model class names: {list(self.model.names.values())}")
                except Exception as e:
                    print(f"Error loading custom model {model_path}: {e}")
                    print("Falling back to pretrained model")
                    self.model = YOLO("yolo11n.pt")
                    print("Loaded YOLOv11 pretrained model (fallback)")
            else:
                print(f"Model path does not exist: {model_path} (checked: {abs_path})")
                print("Falling back to pretrained model")
                try:
                    self.model = YOLO("yolo11n.pt")
                    print("Loaded YOLOv11 pretrained model (fallback)")
                except Exception as e:
                    print(f"Error loading fallback model: {e}")
                    raise
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
        print(f"Available classes: {self.class_names}")
        
        # Initialize OCR based on selected engine
        self.ocr_reader = None
        self.gemini_model = None  # Legacy Gemini API
        self.gemini_client = None  # New Gemini API
        
        if self.enable_ocr:
            if self.ocr_engine == 'gemini':
                api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
                if not api_key:
                    print("Warning: GEMINI_API_KEY not found. Set environment variable or pass gemini_api_key parameter.")
                    print("Falling back to EasyOCR")
                    self.ocr_engine = 'easyocr'
                else:
                    # Try to use new Gemini API first (better text extraction)
                    if NEW_GEMINI_AVAILABLE:
                        try:
                            self.gemini_client = new_genai.Client(api_key=api_key)
                            print("Gemini OCR (new API) initialized successfully!")
                        except Exception as e:
                            print(f"Warning: Could not initialize new Gemini API: {e}")
                            # Fall back to legacy API or EasyOCR
                            if GEMINI_AVAILABLE:
                                try:
                                    genai.configure(api_key=api_key)
                                    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                                    print("Gemini OCR (legacy API) initialized successfully!")
                                except Exception as e2:
                                    print(f"Warning: Could not initialize legacy Gemini: {e2}")
                                    print("Falling back to EasyOCR")
                                    self.ocr_engine = 'easyocr'
                            else:
                                print("Falling back to EasyOCR")
                                self.ocr_engine = 'easyocr'
                    # Fall back to legacy Gemini API if new API not available
                    elif GEMINI_AVAILABLE:
                        try:
                            genai.configure(api_key=api_key)
                            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                            print("Gemini OCR (legacy API) initialized successfully!")
                        except Exception as e:
                            print(f"Warning: Could not initialize Gemini: {e}")
                            print("Falling back to EasyOCR")
                            self.ocr_engine = 'easyocr'
                    else:
                        print("Warning: google-genai or google-generativeai not installed.")
                        print("Install with: pip install google-genai")
                        print("Falling back to EasyOCR")
                        self.ocr_engine = 'easyocr'
            
            # Initialize EasyOCR if selected or as fallback
            if self.ocr_engine == 'easyocr':
                try:
                    print("Initializing EasyOCR reader...")
                    self.ocr_reader = easyocr.Reader(ocr_languages, gpu=False)
                    print("EasyOCR reader ready!")
                except Exception as e:
                    print(f"Warning: Could not initialize EasyOCR: {e}")
                    print("OCR will be disabled")
                    self.enable_ocr = False
                    self.ocr_reader = None
    
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
    
    def preprocess_for_ocr(self, plate_roi: np.ndarray, for_gemini: bool = False) -> np.ndarray:
        """
        Preprocess license plate ROI for better OCR results
        
        Args:
            plate_roi: Region of interest (license plate crop)
            for_gemini: If True, use lighter preprocessing optimized for Gemini OCR
            
        Returns:
            Preprocessed image (grayscale for EasyOCR, RGB for Gemini)
        """
        # For Gemini, use lighter preprocessing since it handles images well
        if for_gemini:
            return self.preprocess_for_gemini(plate_roi)
        
        # Enhanced preprocessing for EasyOCR
        # Convert to grayscale
        if len(plate_roi.shape) == 3:
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_roi.copy()
        
        height, width = gray.shape
        
        # 1. Upscale if too small (minimum 200x60 for good OCR)
        if width < 200 or height < 60:
            scale = max(200 / width, 60 / height) * 2.0
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            height, width = gray.shape
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 3. Light Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 4. Adaptive thresholding (better than global for uneven lighting)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 5. Morphological operations to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        # 6. Remove small noise particles
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(~processed, connectivity=8)
        min_area = (width * height) * 0.002  # Remove noise < 0.2% of image area
        result = processed.copy()
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                result[labels == i] = 255
        
        return result
    
    def preprocess_for_gemini(self, plate_roi: np.ndarray) -> np.ndarray:
        """
        Light preprocessing optimized for Gemini OCR
        Gemini handles most image conditions well, so we do minimal enhancement
        
        Args:
            plate_roi: Region of interest (license plate crop)
            
        Returns:
            RGB image optimized for Gemini
        """
        # Ensure RGB format
        if len(plate_roi.shape) == 2:
            # Grayscale to RGB
            rgb = cv2.cvtColor(plate_roi, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2RGB)
        
        height, width = rgb.shape[:2]
        
        # Upscale if too small (Gemini works better with larger images)
        if width < 100 or height < 30:
            scale = max(100 / width, 30 / height) * 2.5
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Light enhancement: increase brightness and contrast slightly
        # This helps with dark or low-contrast images without over-processing
        rgb = cv2.convertScaleAbs(rgb, alpha=1.15, beta=15)
        
        # Optional: Apply CLAHE to each channel for better contrast
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return rgb
    
    def extract_text_from_plate(self, image: np.ndarray, bbox: Dict[str, float]) -> Tuple[str, float]:
        """
        Extract text from a license plate region using the configured OCR engine
        
        Args:
            image: Full image
            bbox: Bounding box dictionary with x1, y1, x2, y2
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        if not self.enable_ocr:
            return "", 0.0
        
        # Route to appropriate OCR engine
        # Prefer new Gemini API if available
        if self.ocr_engine == 'gemini' and self.gemini_client is not None:
            return self.extract_text_from_plate_gemini_v2(image, bbox)
        elif self.ocr_engine == 'gemini' and self.gemini_model is not None:
            return self.extract_text_from_plate_gemini(image, bbox)
        elif self.ocr_engine == 'easyocr' and self.ocr_reader is not None:
            return self.extract_text_from_plate_easyocr(image, bbox)
        else:
            return "", 0.0
    
    def extract_text_from_plate_easyocr(self, image: np.ndarray, bbox: Dict[str, float]) -> Tuple[str, float]:
        """
        Extract text from a license plate region using EasyOCR
        
        Args:
            image: Full image
            bbox: Bounding box dictionary with x1, y1, x2, y2
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        if self.ocr_reader is None:
            return "", 0.0
        
        try:
            # Extract ROI (Region of Interest)
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            
            # Ensure coordinates are within image bounds
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            if x2 <= x1 or y2 <= y1:
                return "", 0.0
            
            # Crop the license plate region
            plate_roi = image[y1:y2, x1:x2]
            
            if plate_roi.size == 0:
                return "", 0.0
            
            # Preprocess for OCR (enhanced preprocessing for EasyOCR)
            processed_roi = self.preprocess_for_ocr(plate_roi, for_gemini=False)
            
            # Run OCR
            results = self.ocr_reader.readtext(processed_roi)
            
            if not results:
                return "", 0.0
            
            # Combine all detected text
            texts = []
            confidences = []
            for (bbox_ocr, text, conf) in results:
                # Filter out low confidence detections
                if conf > 0.3:
                    texts.append(text.strip())
                    confidences.append(conf)
            
            if not texts:
                return "", 0.0
            
            # Combine texts and calculate average confidence
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Clean up text (remove spaces, fix common OCR errors)
            combined_text = combined_text.replace(" ", "").upper()
            
            return combined_text, avg_confidence
            
        except Exception as e:
            print(f"Error during EasyOCR: {e}")
            return "", 0.0
    
    def extract_text_from_plate_gemini_v2(self, image: np.ndarray, bbox: Dict[str, float]) -> Tuple[str, float]:
        """
        Extract text from a license plate region using the new Google Gemini API (2.0+)
        
        Args:
            image: Full image
            bbox: Bounding box dictionary with x1, y1, x2, y2
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        if self.gemini_client is None:
            return "", 0.0
        
        try:
            # Extract ROI (Region of Interest)
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            
            # Ensure coordinates are within image bounds
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            if x2 <= x1 or y2 <= y1:
                return "", 0.0
            
            # Crop the license plate region
            plate_roi = image[y1:y2, x1:x2]
            
            if plate_roi.size == 0:
                return "", 0.0
            
            # Preprocess for Gemini OCR (light enhancement optimized for Gemini)
            plate_rgb = self.preprocess_for_gemini(plate_roi)
            plate_pil = Image.fromarray(plate_rgb)
            
            # Prepare prompt for Gemini - more specific prompt for better results
            prompt = """Extract the license plate number from this image. 
Return ONLY the alphanumeric characters on the license plate without any spaces, punctuation, hyphens, or additional explanation.
Do not include any descriptive text, prefixes, or suffixes.
If you cannot clearly read the license plate, return an empty string."""
            
            # Call new Gemini API
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, plate_pil]
            )
            
            # Extract text from response
            if response and response.text:
                extracted_text = response.text.strip()
                # Clean up text (remove spaces, punctuation, convert to uppercase)
                extracted_text = extracted_text.replace(" ", "").replace("-", "").replace(".", "").replace("_", "")
                extracted_text = extracted_text.replace(":", "").replace(";", "").replace(",", "")
                # Remove common prefixes/suffixes that Gemini might add
                extracted_text = extracted_text.replace("LICENSEPLATE", "").replace("PLATENUMBER", "")
                extracted_text = extracted_text.replace("PLATE", "").replace("NUMBER", "")
                extracted_text = extracted_text.replace("LICENSE", "")
                # Remove quotes if present
                extracted_text = extracted_text.strip('"').strip("'")
                extracted_text = extracted_text.upper()
                
                # Use high confidence for Gemini - it's generally reliable
                confidence = 0.90 if extracted_text else 0.0
                
                return extracted_text, confidence
            else:
                return "", 0.0
            
        except Exception as e:
            print(f"Error during Gemini OCR (new API): {e}")
            return "", 0.0
    
    def extract_text_from_plate_gemini(self, image: np.ndarray, bbox: Dict[str, float]) -> Tuple[str, float]:
        """
        Extract text from a license plate region using Google Gemini Vision API
        
        Args:
            image: Full image
            bbox: Bounding box dictionary with x1, y1, x2, y2
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        if self.gemini_model is None:
            return "", 0.0
        
        try:
            # Extract ROI (Region of Interest)
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            
            # Ensure coordinates are within image bounds
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            if x2 <= x1 or y2 <= y1:
                return "", 0.0
            
            # Crop the license plate region
            plate_roi = image[y1:y2, x1:x2]
            
            if plate_roi.size == 0:
                return "", 0.0
            
            # Preprocess for Gemini OCR (light enhancement optimized for Gemini)
            plate_rgb = self.preprocess_for_gemini(plate_roi)
            plate_pil = Image.fromarray(plate_rgb)
            
            # Prepare prompt for Gemini
            prompt = """Extract the license plate number from this image. 
Return ONLY the alphanumeric characters on the license plate without any spaces, punctuation, hyphens, or additional explanation.
Do not include any descriptive text, prefixes, or suffixes.
If you cannot clearly read the license plate, return an empty string."""
            
            # Call Gemini API
            response = self.gemini_model.generate_content([prompt, plate_pil])
            
            # Extract text from response
            if response and response.text:
                extracted_text = response.text.strip()
                # Clean up text (remove spaces, punctuation, convert to uppercase)
                extracted_text = extracted_text.replace(" ", "").replace("-", "").replace(".", "").upper()
                # Filter out common prefixes/suffixes that Gemini might add
                extracted_text = extracted_text.replace("LICENSEPLATE", "").replace("PLATENUMBER", "")
                
                # Gemini doesn't provide confidence scores, so we use a default high value
                # You could potentially analyze the response structure for confidence if available
                confidence = 0.85  # Default confidence for Gemini
                
                return extracted_text, confidence
            else:
                return "", 0.0
            
        except Exception as e:
            print(f"Error during Gemini OCR: {e}")
            return "", 0.0
    
    def detect_license_plates(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect license plates in the image and extract text using OCR
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with bounding boxes, confidence, and extracted text
        """
        if image is None:
            return []
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Run YOLO inference
        results = self.model(processed_image, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        all_detections_before_filter = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                # Store all detections for debugging
                all_detections_before_filter.append({
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": confidence
                })
                
                # Filter for license plate detections only
                # Check if class name contains license plate indicators
                # Also handle variations like "License_Plate", "license_plate", "License Plate", etc.
                class_name_lower = class_name.lower().replace("_", " ").replace("-", " ")
                is_license_plate = (
                    "license" in class_name_lower and "plate" in class_name_lower
                ) or (
                    "License_Plate" == class_name or 
                    "license_plate" == class_name_lower
                )
                
                # If no license plate detected, skip this detection
                if not is_license_plate:
                    continue
                
                bbox_dict = {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1)
                }
                
                # Extract text from license plate if OCR is enabled
                plate_text = ""
                ocr_confidence = 0.0
                if self.enable_ocr:
                    # Extract text from the detected license plate region
                    plate_text, ocr_confidence = self.extract_text_from_plate(image, bbox_dict)
                
                detection = {
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": bbox_dict,
                    "plate_text": plate_text,
                    "ocr_confidence": round(ocr_confidence, 4)
                }
                detections.append(detection)
        
        # Debug output: show what was detected before filtering
        if all_detections_before_filter and len(detections) == 0:
            print(f"DEBUG: Model detected {len(all_detections_before_filter)} objects, but none were license plates:")
            for det in all_detections_before_filter:
                print(f"  - {det['class_name']} (ID: {det['class_id']}, conf: {det['confidence']:.2f})")
        
        return detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image, including extracted text
        
        Args:
            image: Original image
            detections: List of detection dictionaries
            
        Returns:
            Image with bounding boxes and text drawn
        """
        annotated_image = image.copy()
        
        for det in detections:
            bbox = det["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            confidence = det["confidence"]
            class_name = det["class_name"]
            plate_text = det.get("plate_text", "")
            ocr_confidence = det.get("ocr_confidence", 0.0)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label with text if available
            if plate_text:
                label = f"{plate_text} ({confidence:.2f})"
                ocr_label = f"OCR: {ocr_confidence:.2f}"
            else:
                label = f"{class_name}: {confidence:.2f}"
                ocr_label = None
            
            # Draw main label
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
            
            # Draw OCR label below if available
            if ocr_label:
                ocr_label_size, _ = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ocr_label_y = y2 + 20
                cv2.rectangle(
                    annotated_image,
                    (x1, ocr_label_y - ocr_label_size[1] - 3),
                    (x1 + ocr_label_size[0], ocr_label_y + 3),
                    (255, 0, 0),
                    -1
                )
                cv2.putText(
                    annotated_image,
                    ocr_label,
                    (x1, ocr_label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
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


class GeminiImageService:
    """
    Service for using Gemini's enhanced image understanding capabilities:
    - Object Detection (Gemini 2.0+)
    - Segmentation (Gemini 2.5+)
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """
        Initialize Gemini Image Service
        
        Args:
            api_key: Google Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Gemini model to use (default: "gemini-2.5-flash")
        """
        if not NEW_GEMINI_AVAILABLE:
            raise ImportError("google-genai package not installed. Install with: pip install google-genai")
        
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = new_genai.Client(api_key=self.api_key)
        print(f"Gemini Image Service initialized with model: {model}")
    
    def detect_objects(self, image: Image.Image, prompt: str = "Detect all of the prominent items in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.") -> List[Dict[str, Any]]:
        """
        Detect objects in an image using Gemini 2.0+ object detection
        
        Args:
            image: PIL Image object
            prompt: Detection prompt (can be customized for specific objects)
            
        Returns:
            List of detection dictionaries with bounding boxes and labels
        """
        try:
            config = types.GenerateContentConfig(
                response_mime_type="application/json"
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[image, prompt],
                config=config
            )
            
            import json
            bounding_boxes = json.loads(response.text)
            
            # Convert normalized coordinates to absolute coordinates
            width, height = image.size
            detections = []
            
            for item in bounding_boxes:
                if "box_2d" in item:
                    # Coordinates are [ymin, xmin, ymax, xmax] normalized to 0-1000
                    ymin, xmin, ymax, xmax = item["box_2d"]
                    
                    # Convert to absolute coordinates
                    abs_x1 = int(xmin / 1000 * width)
                    abs_y1 = int(ymin / 1000 * height)
                    abs_x2 = int(xmax / 1000 * width)
                    abs_y2 = int(ymax / 1000 * height)
                    
                    detections.append({
                        "label": item.get("label", "unknown"),
                        "box_2d": [abs_x1, abs_y1, abs_x2, abs_y2],
                        "box_2d_normalized": [ymin, xmin, ymax, xmax],
                        "confidence": item.get("confidence", 1.0)
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error during Gemini object detection: {e}")
            return []
    
    def segment_objects(self, image: Image.Image, prompt: str = "Give the segmentation masks for the wooden and glass items. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels.") -> List[Dict[str, Any]]:
        """
        Segment objects in an image using Gemini 2.5+ segmentation
        
        Args:
            image: PIL Image object
            prompt: Segmentation prompt (describes what to segment)
            
        Returns:
            List of segmentation dictionaries with bounding boxes, masks, and labels
        """
        try:
            # Resize image if too large (recommended for better performance)
            original_size = image.size
            img_resized = image.copy()
            if max(image.size) > 1024:
                img_resized.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
            
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # Better results for detection
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, img_resized],
                config=config
            )
            
            import json
            import re
            
            # Parse JSON response (may be wrapped in markdown code blocks)
            response_text = response.text
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            else:
                # Try to find JSON array
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
            
            items = json.loads(response_text)
            
            # Process each segmentation item
            segmentations = []
            img_width, img_height = original_size
            resized_width, resized_height = img_resized.size
            
            for item in items:
                if "box_2d" not in item or "mask" not in item:
                    continue
                
                # Get bounding box coordinates (normalized 0-1000)
                box = item["box_2d"]
                y0_norm, x0_norm, y1_norm, x1_norm = box
                
                # Convert to absolute coordinates (based on original image size)
                y0 = int(y0_norm / 1000 * img_height)
                x0 = int(x0_norm / 1000 * img_width)
                y1 = int(y1_norm / 1000 * img_height)
                x1 = int(x1_norm / 1000 * img_width)
                
                # Skip invalid boxes
                if y0 >= y1 or x0 >= x1:
                    continue
                
                # Process mask
                mask_str = item["mask"]
                if not mask_str.startswith("data:image/png;base64,"):
                    continue
                
                # Remove prefix and decode
                mask_str = mask_str.removeprefix("data:image/png;base64,")
                mask_data = base64.b64decode(mask_str)
                mask_img = Image.open(io.BytesIO(mask_data))
                
                # Resize mask to match bounding box (in original image coordinates)
                box_width = x1 - x0
                box_height = y1 - y0
                mask_resized = mask_img.resize((box_width, box_height), Image.Resampling.BILINEAR)
                
                # Convert mask to numpy array
                mask_array = np.array(mask_resized)
                
                segmentations.append({
                    "label": item.get("label", "unknown"),
                    "box_2d": [x0, y0, x1, y1],
                    "box_2d_normalized": box,
                    "mask": mask_array,
                    "mask_image": mask_img,
                    "confidence": item.get("confidence", 1.0)
                })
            
            return segmentations
            
        except Exception as e:
            print(f"Error during Gemini segmentation: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def load_image_from_file(self, file_path: str) -> Optional[Image.Image]:
        """Load image from file path"""
        try:
            return Image.open(file_path)
        except Exception as e:
            print(f"Error loading image from file: {e}")
            return None
    
    def load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Load image from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"Error loading image from URL: {e}")
            return None
    
    def load_image_from_base64(self, data: str) -> Optional[Image.Image]:
        """Load image from base64 encoded string"""
        try:
            # Remove data URL prefix if present
            if ',' in data:
                data = data.split(',')[1]
            
            image_data = base64.b64decode(data)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            print(f"Error loading image from base64: {e}")
            return None
    
    def visualize_detections(self, image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
        """
        Visualize detections on image
        
        Args:
            image: Original PIL Image
            detections: List of detection dictionaries
            
        Returns:
            PIL Image with bounding boxes drawn
        """
        from PIL import ImageDraw, ImageFont
        
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        for det in detections:
            x1, y1, x2, y2 = det["box_2d"]
            label = det.get("label", "unknown")
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            
            # Draw label
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Get text size
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw label background
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="green")
            draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
        
        return annotated
    
    def visualize_segmentations(self, image: Image.Image, segmentations: List[Dict[str, Any]], 
                                alpha: float = 0.5) -> Image.Image:
        """
        Visualize segmentations on image with mask overlays
        
        Args:
            image: Original PIL Image
            segmentations: List of segmentation dictionaries
            alpha: Transparency for mask overlay (0.0 to 1.0)
            
        Returns:
            PIL Image with segmentation masks overlaid
        """
        from PIL import ImageDraw
        
        annotated = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        colors = [
            (255, 0, 0, int(255 * alpha)),  # Red
            (0, 255, 0, int(255 * alpha)),   # Green
            (0, 0, 255, int(255 * alpha)),   # Blue
            (255, 255, 0, int(255 * alpha)), # Yellow
            (255, 0, 255, int(255 * alpha)), # Magenta
            (0, 255, 255, int(255 * alpha)), # Cyan
        ]
        
        for i, seg in enumerate(segmentations):
            x0, y0, x1, y1 = seg["box_2d"]
            mask_array = seg["mask"]
            label = seg.get("label", "unknown")
            
            color = colors[i % len(colors)]
            
            # Draw mask pixels
            mask_img = Image.fromarray(mask_array)
            for y in range(mask_array.shape[0]):
                for x in range(mask_array.shape[1]):
                    if mask_array[y, x] > 128:  # Threshold for mask
                        overlay_draw.point((x0 + x, y0 + y), fill=color)
            
            # Draw bounding box
            overlay_draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 255), width=2)
        
        # Composite overlay on original image
        result = Image.alpha_composite(annotated, overlay)
        return result.convert("RGB")

