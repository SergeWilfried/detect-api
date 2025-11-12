"""
Gemini Image Service for object detection and segmentation
Extracted from detection_service.py for better modularity
"""
import os
import io
import base64
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image

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

from core.config import settings


class GeminiImageService:
    """
    Service for using Gemini's enhanced image understanding capabilities:
    - Object Detection (Gemini 2.0+)
    - Segmentation (Gemini 2.5+)
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Gemini Image Service

        Args:
            api_key: Google Gemini API key (defaults to settings.gemini_api_key)
            model: Gemini model to use (default: from settings)
        """
        if not NEW_GEMINI_AVAILABLE:
            raise ImportError("google-genai package not installed. Install with: pip install google-genai")

        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set environment variable or pass api_key parameter.")

        self.model = model or settings.gemini_model
        self.client = new_genai.Client(api_key=self.api_key)
        print(f"Gemini Image Service initialized with model: {self.model}")

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
