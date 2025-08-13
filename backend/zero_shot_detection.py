"""
Zero-Shot Object Detection Module
Uses CLIP + SAM for open-vocabulary object detection
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import clip
from transformers import pipeline, AutoProcessor, AutoModel
from dataclasses import dataclass

@dataclass
class Detection:
    """Detection result with zero-shot capabilities"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    description: str
    color: str = "unknown"

class ZeroShotDetector:
    """Zero-shot object detection using CLIP and segmentation models"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clip_model = None
        self.clip_preprocess = None
        self.depth_estimator = None
        self.segmentation_model = None
        self._load_models()
    
    def _load_models(self):
        """Load zero-shot detection models"""
        try:
            # Load CLIP for zero-shot classification
            print("Loading CLIP model for zero-shot detection...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            print("✅ CLIP model loaded")
            
            # Load depth estimation for 3D understanding
            print("Loading depth estimation model...")
            self.depth_estimator = pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=0 if self.device == "cuda" else -1
            )
            print("✅ Depth estimation model loaded")
            
            # Load segmentation model for precise object boundaries
            print("Loading segmentation model...")
            self.segmentation_model = pipeline(
                "image-segmentation",
                model="facebook/detr-resnet-50-panoptic",
                device=0 if self.device == "cuda" else -1
            )
            print("✅ Segmentation model loaded")
            
        except Exception as e:
            print(f"❌ Error loading zero-shot models: {e}")
    
    def detect_objects_zero_shot(self, image: np.ndarray, query_classes: List[str] = None) -> List[Detection]:
        """
        Perform zero-shot object detection
        
        Args:
            image: Input image as numpy array
            query_classes: List of classes to detect (if None, uses common objects)
        
        Returns:
            List of Detection objects
        """
        if not self.clip_model:
            return []
        
        # Default query classes if none provided
        if query_classes is None:
            query_classes = [
                "person", "car", "bicycle", "dog", "cat", "bird", "horse", "sheep", "cow",
                "bottle", "chair", "sofa", "table", "bed", "toilet", "tv", "laptop", "mouse",
                "keyboard", "cell phone", "book", "clock", "vase", "scissors", "teddy bear",
                "red object", "blue object", "green object", "yellow object", "black object", "white object"
            ]
        
        detections = []
        
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Get segmentation masks
            segments = self.segmentation_model(pil_image)
            
            # Process each segment
            for segment in segments:
                mask = np.array(segment['mask'])
                
                # Get bounding box from mask
                coords = np.where(mask > 0)
                if len(coords[0]) == 0:
                    continue
                
                y1, y2 = coords[0].min(), coords[0].max()
                x1, x2 = coords[1].min(), coords[1].max()
                
                # Extract region of interest
                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                # Convert ROI to PIL for CLIP
                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                roi_preprocessed = self.clip_preprocess(roi_pil).unsqueeze(0).to(self.device)
                
                # Prepare text queries
                text_queries = [f"a photo of a {cls}" for cls in query_classes]
                text_tokens = clip.tokenize(text_queries).to(self.device)
                
                # Get CLIP predictions
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(roi_preprocessed)
                    text_features = self.clip_model.encode_text(text_tokens)
                    
                    # Calculate similarities
                    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    best_match_idx = similarities.argmax().item()
                    confidence = similarities[0, best_match_idx].item()
                
                # Only keep high-confidence detections
                if confidence > 0.15:  # Threshold for zero-shot detection
                    class_name = query_classes[best_match_idx]
                    
                    # Analyze color
                    color = self._analyze_color(roi)
                    
                    # Create description
                    description = f"{color} {class_name}" if color != "unknown" else class_name
                    
                    detection = Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_name=class_name,
                        description=description,
                        color=color
                    )
                    detections.append(detection)
            
        except Exception as e:
            print(f"Error in zero-shot detection: {e}")
        
        return detections
    
    def _analyze_color(self, roi: np.ndarray) -> str:
        """Analyze dominant color in region of interest"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            v_mean = np.mean(hsv[:, :, 2])
            
            # Color classification
            if v_mean < 50:
                return "black"
            elif s_mean < 30:
                return "white" if v_mean > 180 else "gray"
            elif h_mean < 10 or h_mean > 170:
                return "red"
            elif 10 <= h_mean < 25:
                return "orange"
            elif 25 <= h_mean < 35:
                return "yellow"
            elif 35 <= h_mean < 80:
                return "green"
            elif 80 <= h_mean < 130:
                return "blue"
            elif 130 <= h_mean < 160:
                return "purple"
            else:
                return "pink"
        except:
            return "unknown"
    
    def detect_with_text_query(self, image: np.ndarray, text_query: str) -> List[Detection]:
        """
        Detect objects based on natural language query
        
        Args:
            image: Input image
            text_query: Natural language description (e.g., "red car", "person sitting")
        
        Returns:
            List of detections matching the query
        """
        # Parse query to extract potential classes and attributes
        query_words = text_query.lower().split()
        
        # Color keywords
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray"]
        query_colors = [word for word in query_words if word in colors]
        
        # Object keywords (expand based on query)
        potential_objects = [
            word for word in query_words 
            if word not in colors and len(word) > 2
        ]
        
        # Add common variations
        expanded_classes = potential_objects.copy()
        for obj in potential_objects:
            expanded_classes.extend([f"{obj}s", f"small {obj}", f"large {obj}"])
        
        # Perform detection
        detections = self.detect_objects_zero_shot(image, expanded_classes)
        
        # Filter based on query
        filtered_detections = []
        for detection in detections:
            # Check if detection matches query criteria
            matches_color = not query_colors or detection.color in query_colors
            matches_object = any(obj in detection.class_name.lower() for obj in potential_objects)
            
            if matches_color and (matches_object or not potential_objects):
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def get_depth_information(self, image: np.ndarray) -> np.ndarray:
        """Get depth map for 3D understanding"""
        try:
            if self.depth_estimator:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                depth_result = self.depth_estimator(pil_image)
                depth_map = np.array(depth_result['depth'])
                return depth_map
        except Exception as e:
            print(f"Depth estimation error: {e}")
        
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
    def analyze_scene_composition(self, image: np.ndarray) -> Dict:
        """Analyze overall scene composition and relationships"""
        detections = self.detect_objects_zero_shot(image)
        depth_map = self.get_depth_information(image)
        
        # Analyze spatial relationships
        relationships = []
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i+1:], i+1):
                # Calculate relative positions
                x1_center = (det1.bbox[0] + det1.bbox[2]) / 2
                y1_center = (det1.bbox[1] + det1.bbox[3]) / 2
                x2_center = (det2.bbox[0] + det2.bbox[2]) / 2
                y2_center = (det2.bbox[1] + det2.bbox[3]) / 2
                
                # Determine spatial relationship
                if abs(x1_center - x2_center) < 50:  # Vertically aligned
                    if y1_center < y2_center:
                        relationships.append(f"{det1.class_name} above {det2.class_name}")
                    else:
                        relationships.append(f"{det1.class_name} below {det2.class_name}")
                elif abs(y1_center - y2_center) < 50:  # Horizontally aligned
                    if x1_center < x2_center:
                        relationships.append(f"{det1.class_name} left of {det2.class_name}")
                    else:
                        relationships.append(f"{det1.class_name} right of {det2.class_name}")
        
        return {
            'objects': [
                {
                    'class': det.class_name,
                    'description': det.description,
                    'confidence': det.confidence,
                    'bbox': det.bbox,
                    'color': det.color
                }
                for det in detections
            ],
            'relationships': relationships,
            'object_count': len(detections),
            'dominant_colors': list(set(det.color for det in detections if det.color != "unknown"))
        }

# Global zero-shot detector instance
zero_shot_detector = None

def get_zero_shot_detector() -> ZeroShotDetector:
    """Get or create zero-shot detector instance"""
    global zero_shot_detector
    if zero_shot_detector is None:
        zero_shot_detector = ZeroShotDetector()
    return zero_shot_detector
