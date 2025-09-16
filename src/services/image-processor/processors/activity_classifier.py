"""
Activity Classification Processor - Identifies urban activities and events from street imagery.
Classifies activities like construction, events, dining, protests, etc. for pricing impact analysis.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

from ....shared.config.settings import settings
from ....shared.utils.logging_utils import get_logger, log_performance
from ....shared.utils.error_handling import ImageProcessingError


logger = get_logger(__name__)


@dataclass
class DetectedActivity:
    """Data class for detected activity information."""
    activity_type: str
    intensity_score: float  # 0-10 scale
    pricing_impact: str  # 'increase', 'decrease', 'neutral'
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    description: str = ""


class ActivityClassifier:
    """
    Advanced activity classification for urban street scenes.
    Identifies various activities and their potential impact on ride demand and pricing.
    """
    
    def __init__(self):
        """Initialize the activity classifier with models and configurations."""
        self.device = torch.device('cuda' if torch.cuda.is_available() and settings.image_processing.gpu_enabled else 'cpu')
        self.confidence_threshold = settings.image_processing.confidence_threshold
        
        # Load activity classification model
        self.activity_model = self._load_activity_model()
        
        # Load scene classification model
        self.scene_model = self._load_scene_model()
        
        # Activity definitions with pricing impact
        self.activity_definitions = {
            "construction": {
                "keywords": ["crane", "excavator", "construction_site", "barriers", "hard_hat"],
                "pricing_impact": "increase",
                "base_intensity": 7.0,
                "description": "Construction activity detected"
            },
            "outdoor_dining": {
                "keywords": ["tables", "chairs", "umbrellas", "restaurant", "cafe"],
                "pricing_impact": "neutral",
                "base_intensity": 3.0,
                "description": "Outdoor dining area"
            },
            "protest": {
                "keywords": ["crowd", "signs", "banners", "demonstration"],
                "pricing_impact": "increase",
                "base_intensity": 8.0,
                "description": "Protest or demonstration"
            },
            "event": {
                "keywords": ["stage", "tent", "festival", "concert", "gathering"],
                "pricing_impact": "increase",
                "base_intensity": 6.0,
                "description": "Public event or gathering"
            },
            "shopping": {
                "keywords": ["shopping_bags", "retail", "store", "market"],
                "pricing_impact": "neutral",
                "base_intensity": 4.0,
                "description": "Shopping activity"
            },
            "delivery": {
                "keywords": ["delivery_truck", "packages", "loading"],
                "pricing_impact": "decrease",
                "base_intensity": 2.0,
                "description": "Delivery activity"
            },
            "emergency": {
                "keywords": ["ambulance", "fire_truck", "police", "emergency"],
                "pricing_impact": "increase",
                "base_intensity": 9.0,
                "description": "Emergency services activity"
            },
            "transit": {
                "keywords": ["bus_stop", "subway", "train", "transit"],
                "pricing_impact": "increase",
                "base_intensity": 5.0,
                "description": "Public transit activity"
            },
            "vendor": {
                "keywords": ["food_cart", "vendor", "street_food"],
                "pricing_impact": "neutral",
                "base_intensity": 3.0,
                "description": "Street vendor activity"
            },
            "nightlife": {
                "keywords": ["bar", "club", "nightlife", "entertainment"],
                "pricing_impact": "increase",
                "base_intensity": 6.0,
                "description": "Nightlife activity"
            }
        }
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(
            "Activity classifier initialized",
            extra={
                "device": str(self.device),
                "activity_types": len(self.activity_definitions),
                "confidence_threshold": self.confidence_threshold
            }
        )
    
    def _load_activity_model(self):
        """Load activity classification model."""
        try:
            # For now, we'll use a placeholder model
            # In production, this would be a trained CNN for activity classification
            logger.info("Activity model initialized (using heuristic approach)")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load activity model: {str(e)}")
            return None
    
    def _load_scene_model(self):
        """Load scene classification model."""
        try:
            # Load a pre-trained scene classification model (e.g., Places365)
            # For now, we'll use a placeholder
            logger.info("Scene model initialized (using heuristic approach)")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load scene model: {str(e)}")
            return None
    
    @log_performance("activity_classification")
    def classify_activities(
        self, 
        image: np.ndarray, 
        location_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Classify activities present in the street scene.
        
        Args:
            image: Input image as numpy array
            location_context: Location context information
            
        Returns:
            List of detected activities with metadata
        """
        try:
            start_time = time.time()
            
            # Analyze image features
            image_features = self._extract_image_features(image)
            
            # Detect activities using multiple approaches
            detected_activities = []
            
            # 1. Color-based activity detection
            color_activities = self._detect_activities_by_color(image, image_features)
            detected_activities.extend(color_activities)
            
            # 2. Edge-based activity detection
            edge_activities = self._detect_activities_by_edges(image, image_features)
            detected_activities.extend(edge_activities)
            
            # 3. Texture-based activity detection
            texture_activities = self._detect_activities_by_texture(image, image_features)
            detected_activities.extend(texture_activities)
            
            # 4. Context-based activity inference
            context_activities = self._infer_activities_from_context(location_context, image_features)
            detected_activities.extend(context_activities)
            
            # 5. Temporal activity analysis (if time context available)
            if 'time_of_day' in location_context:
                temporal_activities = self._detect_temporal_activities(
                    location_context['time_of_day'], image_features
                )
                detected_activities.extend(temporal_activities)
            
            # Remove duplicates and merge similar activities
            merged_activities = self._merge_similar_activities(detected_activities)
            
            # Filter by confidence threshold
            filtered_activities = [
                activity for activity in merged_activities 
                if activity.confidence >= self.confidence_threshold
            ]
            
            # Convert to dictionary format
            activity_results = []
            for activity in filtered_activities:
                activity_dict = {
                    "activity_type": activity.activity_type,
                    "intensity_score": activity.intensity_score,
                    "pricing_impact": activity.pricing_impact,
                    "confidence": activity.confidence,
                    "description": activity.description
                }
                
                if activity.bounding_box:
                    activity_dict["bounding_box"] = {
                        "x": activity.bounding_box[0],
                        "y": activity.bounding_box[1],
                        "width": activity.bounding_box[2] - activity.bounding_box[0],
                        "height": activity.bounding_box[3] - activity.bounding_box[1]
                    }
                
                activity_results.append(activity_dict)
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(
                "Activity classification completed",
                extra={
                    "activities_detected": len(activity_results),
                    "processing_time_ms": processing_time
                }
            )
            
            return activity_results
            
        except Exception as e:
            logger.error(f"Activity classification failed: {str(e)}")
            raise ImageProcessingError(
                operation="activity_classification",
                message=f"Activity classification failed: {str(e)}",
                cause=e
            )
    
    def _extract_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract various features from the image for activity detection.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Color features
            color_features = self._extract_color_features(image)
            
            # Edge features
            edge_features = self._extract_edge_features(image)
            
            # Texture features
            texture_features = self._extract_texture_features(image)
            
            # Brightness and contrast
            brightness = np.mean(image)
            contrast = np.std(image)
            
            return {
                "color": color_features,
                "edges": edge_features,
                "texture": texture_features,
                "brightness": brightness,
                "contrast": contrast,
                "image_shape": image.shape
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return {}
    
    def _extract_color_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract color-based features."""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Calculate color histograms
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
            
            # Dominant colors
            dominant_colors = self._find_dominant_colors(image)
            
            # Color ratios
            total_pixels = image.shape[0] * image.shape[1]
            
            # Orange/yellow ratio (construction equipment)
            orange_mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))
            orange_ratio = np.sum(orange_mask > 0) / total_pixels
            
            # Blue ratio (signage, uniforms)
            blue_mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
            blue_ratio = np.sum(blue_mask > 0) / total_pixels
            
            # Red ratio (emergency vehicles, signs)
            red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            red_ratio = np.sum(red_mask > 0) / total_pixels
            
            return {
                "dominant_colors": dominant_colors,
                "orange_ratio": orange_ratio,
                "blue_ratio": blue_ratio,
                "red_ratio": red_ratio,
                "brightness_distribution": {
                    "mean": np.mean(image),
                    "std": np.std(image)
                }
            }
            
        except Exception as e:
            logger.error(f"Color feature extraction failed: {str(e)}")
            return {}
    
    def _extract_edge_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract edge-based features."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Edge density
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Line detection (for construction, barriers, etc.)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            line_count = len(lines) if lines is not None else 0
            
            # Contour analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Large rectangular contours (signs, barriers, vehicles)
            rectangular_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) == 4:  # Rectangular shape
                        rectangular_contours += 1
            
            return {
                "edge_density": edge_density,
                "line_count": line_count,
                "rectangular_contours": rectangular_contours,
                "total_contours": len(contours)
            }
            
        except Exception as e:
            logger.error(f"Edge feature extraction failed: {str(e)}")
            return {}
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract texture-based features."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture measures
            # Local Binary Pattern (simplified)
            texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            return {
                "texture_variance": texture_variance,
                "gradient_magnitude_mean": np.mean(gradient_magnitude),
                "gradient_magnitude_std": np.std(gradient_magnitude)
            }
            
        except Exception as e:
            logger.error(f"Texture feature extraction failed: {str(e)}")
            return {}
    
    def _find_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Find dominant colors in the image using K-means clustering."""
        try:
            # Reshape image to be a list of pixels
            pixels = image.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            # Apply K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to integers
            centers = np.uint8(centers)
            
            return [tuple(center) for center in centers]
            
        except Exception as e:
            logger.error(f"Dominant color extraction failed: {str(e)}")
            return []
    
    def _detect_activities_by_color(
        self, image: np.ndarray, features: Dict[str, Any]
    ) -> List[DetectedActivity]:
        """Detect activities based on color analysis."""
        activities = []
        
        try:
            color_features = features.get("color", {})
            
            # Construction activity (orange/yellow equipment)
            orange_ratio = color_features.get("orange_ratio", 0)
            if orange_ratio > 0.05:  # 5% orange pixels
                intensity = min(10.0, orange_ratio * 100)
                activities.append(DetectedActivity(
                    activity_type="construction",
                    intensity_score=intensity,
                    pricing_impact="increase",
                    confidence=min(0.9, orange_ratio * 10),
                    description="Construction equipment detected by color analysis"
                ))
            
            # Emergency activity (red vehicles/lights)
            red_ratio = color_features.get("red_ratio", 0)
            if red_ratio > 0.03:  # 3% red pixels
                intensity = min(10.0, red_ratio * 150)
                activities.append(DetectedActivity(
                    activity_type="emergency",
                    intensity_score=intensity,
                    pricing_impact="increase",
                    confidence=min(0.8, red_ratio * 15),
                    description="Emergency vehicles detected by color analysis"
                ))
            
            # Transit activity (blue signage/vehicles)
            blue_ratio = color_features.get("blue_ratio", 0)
            if blue_ratio > 0.04:  # 4% blue pixels
                intensity = min(8.0, blue_ratio * 100)
                activities.append(DetectedActivity(
                    activity_type="transit",
                    intensity_score=intensity,
                    pricing_impact="increase",
                    confidence=min(0.7, blue_ratio * 12),
                    description="Transit infrastructure detected by color analysis"
                ))
            
        except Exception as e:
            logger.error(f"Color-based activity detection failed: {str(e)}")
        
        return activities
    
    def _detect_activities_by_edges(
        self, image: np.ndarray, features: Dict[str, Any]
    ) -> List[DetectedActivity]:
        """Detect activities based on edge analysis."""
        activities = []
        
        try:
            edge_features = features.get("edges", {})
            
            # Construction/barriers (high line count and rectangular contours)
            line_count = edge_features.get("line_count", 0)
            rectangular_contours = edge_features.get("rectangular_contours", 0)
            
            if line_count > 20 and rectangular_contours > 3:
                intensity = min(10.0, (line_count + rectangular_contours * 2) / 10)
                activities.append(DetectedActivity(
                    activity_type="construction",
                    intensity_score=intensity,
                    pricing_impact="increase",
                    confidence=min(0.8, (line_count + rectangular_contours) / 50),
                    description="Construction barriers detected by edge analysis"
                ))
            
            # Event setup (many rectangular structures)
            if rectangular_contours > 5:
                intensity = min(8.0, rectangular_contours / 2)
                activities.append(DetectedActivity(
                    activity_type="event",
                    intensity_score=intensity,
                    pricing_impact="increase",
                    confidence=min(0.7, rectangular_contours / 10),
                    description="Event structures detected by edge analysis"
                ))
            
        except Exception as e:
            logger.error(f"Edge-based activity detection failed: {str(e)}")
        
        return activities
    
    def _detect_activities_by_texture(
        self, image: np.ndarray, features: Dict[str, Any]
    ) -> List[DetectedActivity]:
        """Detect activities based on texture analysis."""
        activities = []
        
        try:
            texture_features = features.get("texture", {})
            
            # High texture variance might indicate construction or busy areas
            texture_variance = texture_features.get("texture_variance", 0)
            
            if texture_variance > 1000:  # High texture variance threshold
                # Could be construction, market, or busy commercial area
                intensity = min(6.0, texture_variance / 500)
                activities.append(DetectedActivity(
                    activity_type="shopping",
                    intensity_score=intensity,
                    pricing_impact="neutral",
                    confidence=min(0.6, texture_variance / 2000),
                    description="Busy commercial area detected by texture analysis"
                ))
            
        except Exception as e:
            logger.error(f"Texture-based activity detection failed: {str(e)}")
        
        return activities
    
    def _infer_activities_from_context(
        self, location_context: Dict[str, Any], features: Dict[str, Any]
    ) -> List[DetectedActivity]:
        """Infer activities based on location context."""
        activities = []
        
        try:
            # Use location description to infer likely activities
            location_desc = location_context.get("location_description", "").lower()
            
            # Airport context
            if "airport" in location_desc:
                activities.append(DetectedActivity(
                    activity_type="transit",
                    intensity_score=7.0,
                    pricing_impact="increase",
                    confidence=0.8,
                    description="Airport transit activity inferred from location"
                ))
            
            # Stadium context
            elif "stadium" in location_desc or "sports" in location_desc:
                activities.append(DetectedActivity(
                    activity_type="event",
                    intensity_score=8.0,
                    pricing_impact="increase",
                    confidence=0.9,
                    description="Sports event activity inferred from location"
                ))
            
            # Shopping context
            elif "mall" in location_desc or "shopping" in location_desc:
                activities.append(DetectedActivity(
                    activity_type="shopping",
                    intensity_score=5.0,
                    pricing_impact="neutral",
                    confidence=0.7,
                    description="Shopping activity inferred from location"
                ))
            
            # Nightlife context
            elif "nightlife" in location_desc or "entertainment" in location_desc:
                activities.append(DetectedActivity(
                    activity_type="nightlife",
                    intensity_score=6.0,
                    pricing_impact="increase",
                    confidence=0.8,
                    description="Nightlife activity inferred from location"
                ))
            
            # Hospital context
            elif "hospital" in location_desc or "medical" in location_desc:
                activities.append(DetectedActivity(
                    activity_type="emergency",
                    intensity_score=4.0,
                    pricing_impact="increase",
                    confidence=0.6,
                    description="Medical facility activity inferred from location"
                ))
            
        except Exception as e:
            logger.error(f"Context-based activity inference failed: {str(e)}")
        
        return activities
    
    def _detect_temporal_activities(
        self, time_of_day: str, features: Dict[str, Any]
    ) -> List[DetectedActivity]:
        """Detect activities based on time of day."""
        activities = []
        
        try:
            # Parse time of day
            hour = int(time_of_day.split(':')[0]) if ':' in time_of_day else 12
            
            # Morning rush (7-9 AM)
            if 7 <= hour <= 9:
                activities.append(DetectedActivity(
                    activity_type="transit",
                    intensity_score=6.0,
                    pricing_impact="increase",
                    confidence=0.7,
                    description="Morning rush hour transit activity"
                ))
            
            # Lunch time (11 AM - 2 PM)
            elif 11 <= hour <= 14:
                activities.append(DetectedActivity(
                    activity_type="outdoor_dining",
                    intensity_score=4.0,
                    pricing_impact="neutral",
                    confidence=0.6,
                    description="Lunch time dining activity"
                ))
            
            # Evening rush (5-7 PM)
            elif 17 <= hour <= 19:
                activities.append(DetectedActivity(
                    activity_type="transit",
                    intensity_score=7.0,
                    pricing_impact="increase",
                    confidence=0.8,
                    description="Evening rush hour transit activity"
                ))
            
            # Night time (9 PM - 2 AM)
            elif hour >= 21 or hour <= 2:
                activities.append(DetectedActivity(
                    activity_type="nightlife",
                    intensity_score=5.0,
                    pricing_impact="increase",
                    confidence=0.7,
                    description="Nighttime entertainment activity"
                ))
            
        except Exception as e:
            logger.error(f"Temporal activity detection failed: {str(e)}")
        
        return activities
    
    def _merge_similar_activities(
        self, activities: List[DetectedActivity]
    ) -> List[DetectedActivity]:
        """Merge similar activities and remove duplicates."""
        if not activities:
            return []
        
        # Group activities by type
        activity_groups = {}
        for activity in activities:
            activity_type = activity.activity_type
            if activity_type not in activity_groups:
                activity_groups[activity_type] = []
            activity_groups[activity_type].append(activity)
        
        # Merge activities of the same type
        merged_activities = []
        for activity_type, group in activity_groups.items():
            if len(group) == 1:
                merged_activities.append(group[0])
            else:
                # Merge multiple detections of the same activity
                avg_intensity = np.mean([a.intensity_score for a in group])
                max_confidence = max([a.confidence for a in group])
                
                # Use the pricing impact from the highest confidence detection
                best_activity = max(group, key=lambda x: x.confidence)
                
                merged_activity = DetectedActivity(
                    activity_type=activity_type,
                    intensity_score=avg_intensity,
                    pricing_impact=best_activity.pricing_impact,
                    confidence=max_confidence,
                    description=f"Merged detection: {best_activity.description}"
                )
                
                merged_activities.append(merged_activity)
        
        return merged_activities
    
    def get_model_version(self) -> str:
        """Get the version of the activity classification model."""
        return "activity_classifier_v1.3.0"
    
    def is_healthy(self) -> bool:
        """Check if the activity classifier is healthy and ready."""
        try:
            # Test feature extraction with a small dummy image
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            features = self._extract_image_features(dummy_image)
            return len(features) > 0
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
