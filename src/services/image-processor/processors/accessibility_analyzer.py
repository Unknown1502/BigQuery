"""
Accessibility Analyzer - Analyzes pickup/dropoff accessibility and infrastructure quality.
Evaluates road conditions, barriers, accessibility for ride-sharing services.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

from ....shared.config.settings import settings
from ....shared.utils.logging_utils import get_logger, log_performance
from ....shared.utils.error_handling import ImageProcessingError


logger = get_logger(__name__)


@dataclass
class AccessibilityFeature:
    """Data class for accessibility feature information."""
    feature_type: str
    impact_score: float  # -1 to 1 (negative = blocks access, positive = improves access)
    confidence: float
    location: Optional[Tuple[int, int, int, int]] = None
    description: str = ""


class AccessibilityAnalyzer:
    """
    Analyzes street accessibility for ride-sharing pickup and dropoff.
    Identifies barriers, road quality, and infrastructure that affects vehicle access.
    """
    
    def __init__(self):
        """Initialize the accessibility analyzer."""
        self.confidence_threshold = settings.image_processing.confidence_threshold
        
        # Accessibility feature definitions
        self.accessibility_features = {
            "construction_barrier": {
                "impact_score": -0.8,
                "description": "Construction barriers blocking access"
            },
            "no_stopping_zone": {
                "impact_score": -0.9,
                "description": "No stopping or parking zone"
            },
            "fire_hydrant": {
                "impact_score": -0.6,
                "description": "Fire hydrant restricting parking"
            },
            "bus_stop": {
                "impact_score": -0.4,
                "description": "Bus stop area with restricted access"
            },
            "loading_zone": {
                "impact_score": 0.3,
                "description": "Designated loading zone"
            },
            "wide_sidewalk": {
                "impact_score": 0.5,
                "description": "Wide sidewalk for passenger access"
            },
            "curb_cut": {
                "impact_score": 0.7,
                "description": "Curb cut for accessibility"
            },
            "clear_road": {
                "impact_score": 0.8,
                "description": "Clear road with good access"
            },
            "double_parked": {
                "impact_score": -0.7,
                "description": "Double-parked vehicles blocking access"
            },
            "bike_lane": {
                "impact_score": -0.3,
                "description": "Bike lane restricting vehicle access"
            }
        }
        
        logger.info(
            "Accessibility analyzer initialized",
            extra={
                "feature_types": len(self.accessibility_features),
                "confidence_threshold": self.confidence_threshold
            }
        )
    
    @log_performance("accessibility_analysis")
    def analyze_accessibility(
        self, 
        image: np.ndarray, 
        location_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze accessibility for ride-sharing pickup/dropoff.
        
        Args:
            image: Input image as numpy array
            location_context: Location context information
            
        Returns:
            Accessibility analysis results
        """
        try:
            start_time = time.time()
            
            # Extract image features for accessibility analysis
            features = self._extract_accessibility_features(image)
            
            # Detect accessibility-related objects and conditions
            detected_features = []
            
            # 1. Detect barriers and obstacles
            barriers = self._detect_barriers(image, features)
            detected_features.extend(barriers)
            
            # 2. Analyze road conditions
            road_conditions = self._analyze_road_conditions(image, features)
            detected_features.extend(road_conditions)
            
            # 3. Detect signage and restrictions
            signage = self._detect_signage(image, features)
            detected_features.extend(signage)
            
            # 4. Analyze parking and stopping areas
            parking_areas = self._analyze_parking_areas(image, features)
            detected_features.extend(parking_areas)
            
            # 5. Evaluate sidewalk and pedestrian access
            pedestrian_access = self._analyze_pedestrian_access(image, features)
            detected_features.extend(pedestrian_access)
            
            # Calculate overall accessibility metrics
            accessibility_score = self._calculate_accessibility_score(detected_features)
            is_accessible = accessibility_score > 0.3
            blocking_factors = self._identify_blocking_factors(detected_features)
            pickup_zones = self._count_pickup_zones(detected_features, image.shape)
            
            processing_time = (time.time() - start_time) * 1000
            
            results = {
                "is_accessible": is_accessible,
                "accessibility_score": accessibility_score,
                "blocking_factors": blocking_factors,
                "pickup_zones_available": pickup_zones,
                "detected_features": [
                    {
                        "feature_type": f.feature_type,
                        "impact_score": f.impact_score,
                        "confidence": f.confidence,
                        "description": f.description
                    }
                    for f in detected_features if f.confidence >= self.confidence_threshold
                ],
                "processing_time_ms": processing_time
            }
            
            logger.info(
                "Accessibility analysis completed",
                extra={
                    "accessibility_score": accessibility_score,
                    "is_accessible": is_accessible,
                    "blocking_factors_count": len(blocking_factors),
                    "processing_time_ms": processing_time
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Accessibility analysis failed: {str(e)}")
            raise ImageProcessingError(
                operation="accessibility_analysis",
                message=f"Accessibility analysis failed: {str(e)}",
                cause=e
            )
    
    def _extract_accessibility_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract features relevant to accessibility analysis."""
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Edge detection for structural elements
            edges = cv2.Canny(gray, 50, 150)
            
            # Line detection for barriers, curbs, lane markings
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            # Contour detection for objects
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Color analysis for signage and markings
            color_features = self._analyze_colors_for_accessibility(image, hsv)
            
            # Texture analysis for road surface
            texture_features = self._analyze_road_texture(gray)
            
            return {
                "edges": edges,
                "lines": lines if lines is not None else [],
                "contours": contours,
                "color_features": color_features,
                "texture_features": texture_features,
                "image_shape": image.shape
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return {}
    
    def _analyze_colors_for_accessibility(self, image: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Analyze colors relevant to accessibility (signs, markings, etc.)."""
        try:
            total_pixels = image.shape[0] * image.shape[1]
            
            # Red signage (no parking, stop signs)
            red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            red_ratio = np.sum(red_mask > 0) / total_pixels
            
            # Yellow markings (caution, construction)
            yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
            yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
            
            # Blue signage (parking, accessibility)
            blue_mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
            blue_ratio = np.sum(blue_mask > 0) / total_pixels
            
            # White markings (lane lines, crosswalks)
            white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
            white_ratio = np.sum(white_mask > 0) / total_pixels
            
            # Orange (construction equipment, cones)
            orange_mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))
            orange_ratio = np.sum(orange_mask > 0) / total_pixels
            
            return {
                "red_ratio": red_ratio,
                "yellow_ratio": yellow_ratio,
                "blue_ratio": blue_ratio,
                "white_ratio": white_ratio,
                "orange_ratio": orange_ratio
            }
            
        except Exception as e:
            logger.error(f"Color analysis failed: {str(e)}")
            return {}
    
    def _analyze_road_texture(self, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze road surface texture and quality."""
        try:
            # Calculate texture measures
            texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Gradient analysis for road surface quality
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Road surface smoothness (lower variance = smoother)
            smoothness_score = max(0, 1 - (texture_variance / 1000))
            
            return {
                "texture_variance": texture_variance,
                "gradient_magnitude_mean": np.mean(gradient_magnitude),
                "smoothness_score": smoothness_score
            }
            
        except Exception as e:
            logger.error(f"Road texture analysis failed: {str(e)}")
            return {}
    
    def _detect_barriers(self, image: np.ndarray, features: Dict[str, Any]) -> List[AccessibilityFeature]:
        """Detect physical barriers that block vehicle access."""
        barriers = []
        
        try:
            color_features = features.get("color_features", {})
            lines = features.get("lines", [])
            contours = features.get("contours", [])
            
            # Construction barriers (orange/yellow colors + linear structures)
            orange_ratio = color_features.get("orange_ratio", 0)
            yellow_ratio = color_features.get("yellow_ratio", 0)
            
            if (orange_ratio > 0.02 or yellow_ratio > 0.03) and len(lines) > 10:
                confidence = min(0.9, (orange_ratio + yellow_ratio) * 20)
                barriers.append(AccessibilityFeature(
                    feature_type="construction_barrier",
                    impact_score=-0.8,
                    confidence=confidence,
                    description="Construction barriers detected"
                ))
            
            # Detect large rectangular objects (barriers, vehicles)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Significant size
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:  # Rectangular shape
                        # Could be a barrier or parked vehicle
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        
                        if 0.5 < aspect_ratio < 4:  # Reasonable aspect ratio for barriers/vehicles
                            barriers.append(AccessibilityFeature(
                                feature_type="physical_barrier",
                                impact_score=-0.6,
                                confidence=0.7,
                                location=(x, y, x+w, y+h),
                                description="Physical barrier or obstacle detected"
                            ))
            
        except Exception as e:
            logger.error(f"Barrier detection failed: {str(e)}")
        
        return barriers
    
    def _analyze_road_conditions(self, image: np.ndarray, features: Dict[str, Any]) -> List[AccessibilityFeature]:
        """Analyze road surface conditions and quality."""
        road_features = []
        
        try:
            texture_features = features.get("texture_features", {})
            
            # Road surface quality
            smoothness_score = texture_features.get("smoothness_score", 0.5)
            
            if smoothness_score > 0.7:
                road_features.append(AccessibilityFeature(
                    feature_type="good_road_surface",
                    impact_score=0.4,
                    confidence=0.8,
                    description="Good road surface quality"
                ))
            elif smoothness_score < 0.3:
                road_features.append(AccessibilityFeature(
                    feature_type="poor_road_surface",
                    impact_score=-0.3,
                    confidence=0.7,
                    description="Poor road surface quality"
                ))
            
            # Detect potholes or road damage (high texture variance in localized areas)
            texture_variance = texture_features.get("texture_variance", 0)
            if texture_variance > 1500:
                road_features.append(AccessibilityFeature(
                    feature_type="road_damage",
                    impact_score=-0.4,
                    confidence=0.6,
                    description="Potential road damage or potholes"
                ))
            
        except Exception as e:
            logger.error(f"Road condition analysis failed: {str(e)}")
        
        return road_features
    
    def _detect_signage(self, image: np.ndarray, features: Dict[str, Any]) -> List[AccessibilityFeature]:
        """Detect traffic signs and parking restrictions."""
        signage_features = []
        
        try:
            color_features = features.get("color_features", {})
            contours = features.get("contours", [])
            
            # Red signage (no parking, stop signs)
            red_ratio = color_features.get("red_ratio", 0)
            if red_ratio > 0.01:  # 1% red pixels
                signage_features.append(AccessibilityFeature(
                    feature_type="no_stopping_zone",
                    impact_score=-0.9,
                    confidence=min(0.8, red_ratio * 50),
                    description="No parking/stopping signage detected"
                ))
            
            # Blue signage (parking allowed, accessibility)
            blue_ratio = color_features.get("blue_ratio", 0)
            if blue_ratio > 0.015:  # 1.5% blue pixels
                signage_features.append(AccessibilityFeature(
                    feature_type="parking_allowed",
                    impact_score=0.6,
                    confidence=min(0.7, blue_ratio * 40),
                    description="Parking allowed signage detected"
                ))
            
            # Detect circular contours (typical for traffic signs)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:  # Sign-sized objects
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.7:  # Circular shape
                            signage_features.append(AccessibilityFeature(
                                feature_type="traffic_sign",
                                impact_score=-0.2,
                                confidence=0.6,
                                description="Traffic sign detected"
                            ))
            
        except Exception as e:
            logger.error(f"Signage detection failed: {str(e)}")
        
        return signage_features
    
    def _analyze_parking_areas(self, image: np.ndarray, features: Dict[str, Any]) -> List[AccessibilityFeature]:
        """Analyze parking and stopping areas."""
        parking_features = []
        
        try:
            color_features = features.get("color_features", {})
            lines = features.get("lines", [])
            
            # White lane markings (parking spaces, loading zones)
            white_ratio = color_features.get("white_ratio", 0)
            line_count = len(lines)
            
            if white_ratio > 0.05 and line_count > 15:  # Significant white markings + lines
                parking_features.append(AccessibilityFeature(
                    feature_type="marked_parking",
                    impact_score=0.7,
                    confidence=min(0.8, white_ratio * 10),
                    description="Marked parking spaces detected"
                ))
            
            # Loading zone detection (combination of markings and clear space)
            if white_ratio > 0.03 and line_count > 8:
                parking_features.append(AccessibilityFeature(
                    feature_type="loading_zone",
                    impact_score=0.5,
                    confidence=0.6,
                    description="Potential loading zone detected"
                ))
            
        except Exception as e:
            logger.error(f"Parking area analysis failed: {str(e)}")
        
        return parking_features
    
    def _analyze_pedestrian_access(self, image: np.ndarray, features: Dict[str, Any]) -> List[AccessibilityFeature]:
        """Analyze pedestrian access and sidewalk conditions."""
        pedestrian_features = []
        
        try:
            lines = features.get("lines", [])
            color_features = features.get("color_features", {})
            
            # Detect curb lines and sidewalk edges
            if len(lines) > 5:
                # Analyze line orientations to detect curbs
                horizontal_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    if abs(angle) < 15 or abs(angle) > 165:  # Nearly horizontal
                        horizontal_lines += 1
                
                if horizontal_lines > 3:
                    pedestrian_features.append(AccessibilityFeature(
                        feature_type="sidewalk_edge",
                        impact_score=0.3,
                        confidence=0.7,
                        description="Sidewalk edge detected"
                    ))
            
            # Crosswalk detection (white striped patterns)
            white_ratio = color_features.get("white_ratio", 0)
            if white_ratio > 0.08:  # High white content might indicate crosswalk
                pedestrian_features.append(AccessibilityFeature(
                    feature_type="crosswalk",
                    impact_score=0.4,
                    confidence=min(0.8, white_ratio * 8),
                    description="Crosswalk detected"
                ))
            
        except Exception as e:
            logger.error(f"Pedestrian access analysis failed: {str(e)}")
        
        return pedestrian_features
    
    def _calculate_accessibility_score(self, features: List[AccessibilityFeature]) -> float:
        """Calculate overall accessibility score from detected features."""
        if not features:
            return 0.5  # Neutral score when no features detected
        
        try:
            # Weight features by confidence and impact
            weighted_sum = 0.0
            total_weight = 0.0
            
            for feature in features:
                if feature.confidence >= self.confidence_threshold:
                    weight = feature.confidence
                    weighted_sum += feature.impact_score * weight
                    total_weight += weight
            
            if total_weight > 0:
                # Normalize to 0-1 scale
                raw_score = weighted_sum / total_weight
                # Convert from -1,1 range to 0,1 range
                normalized_score = (raw_score + 1) / 2
                return max(0.0, min(1.0, normalized_score))
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            logger.error(f"Accessibility score calculation failed: {str(e)}")
            return 0.5
    
    def _identify_blocking_factors(self, features: List[AccessibilityFeature]) -> List[str]:
        """Identify factors that block or restrict access."""
        blocking_factors = []
        
        try:
            for feature in features:
                if (feature.confidence >= self.confidence_threshold and 
                    feature.impact_score < -0.3):  # Significant negative impact
                    
                    # Map feature types to user-friendly blocking factors
                    factor_map = {
                        "construction_barrier": "construction",
                        "no_stopping_zone": "no_stopping",
                        "physical_barrier": "blocked_lane",
                        "double_parked": "heavy_traffic",
                        "road_damage": "poor_visibility"
                    }
                    
                    factor = factor_map.get(feature.feature_type, feature.feature_type)
                    if factor not in blocking_factors:
                        blocking_factors.append(factor)
            
        except Exception as e:
            logger.error(f"Blocking factor identification failed: {str(e)}")
        
        return blocking_factors
    
    def _count_pickup_zones(self, features: List[AccessibilityFeature], image_shape: Tuple[int, int, int]) -> int:
        """Count available pickup zones based on detected features."""
        try:
            # Start with base zones based on image area
            height, width = image_shape[:2]
            base_zones = max(1, (width * height) // 100000)  # Rough estimate
            
            # Adjust based on detected features
            zone_adjustments = 0
            
            for feature in features:
                if feature.confidence >= self.confidence_threshold:
                    if feature.feature_type in ["marked_parking", "loading_zone"]:
                        zone_adjustments += 1
                    elif feature.feature_type in ["construction_barrier", "no_stopping_zone"]:
                        zone_adjustments -= 1
            
            return max(0, base_zones + zone_adjustments)
            
        except Exception as e:
            logger.error(f"Pickup zone counting failed: {str(e)}")
            return 1  # Default to at least one zone
    
    @log_performance("infrastructure_analysis")
    def analyze_infrastructure(
        self, 
        image: np.ndarray, 
        location_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze infrastructure quality and safety.
        
        Args:
            image: Input image
            location_context: Location context information
            
        Returns:
            Infrastructure analysis results
        """
        try:
            # Extract features for infrastructure analysis
            features = self._extract_accessibility_features(image)
            
            # Analyze construction presence
            construction_present = self._detect_construction(image, features)
            
            # Analyze road quality
            road_quality = self._assess_road_quality(features)
            
            # Analyze lighting conditions
            lighting_quality = self._assess_lighting_quality(image)
            
            # Calculate safety score
            safety_score = self._calculate_safety_score(features, construction_present, lighting_quality)
            
            # Analyze signage visibility
            signage_visibility = self._assess_signage_visibility(features)
            
            results = {
                "construction_present": construction_present,
                "road_quality": road_quality,
                "lighting_quality": lighting_quality,
                "safety_score": safety_score,
                "signage_visibility": signage_visibility
            }
            
            logger.info(
                "Infrastructure analysis completed",
                extra={
                    "construction_present": construction_present,
                    "road_quality": road_quality,
                    "safety_score": safety_score
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Infrastructure analysis failed: {str(e)}")
            raise ImageProcessingError(
                operation="infrastructure_analysis",
                message=f"Infrastructure analysis failed: {str(e)}",
                cause=e
            )
    
    def _detect_construction(self, image: np.ndarray, features: Dict[str, Any]) -> bool:
        """Detect presence of construction activity."""
        try:
            color_features = features.get("color_features", {})
            
            # Construction typically has orange/yellow equipment and barriers
            orange_ratio = color_features.get("orange_ratio", 0)
            yellow_ratio = color_features.get("yellow_ratio", 0)
            
            return (orange_ratio > 0.02 or yellow_ratio > 0.03)
            
        except Exception as e:
            logger.error(f"Construction detection failed: {str(e)}")
            return False
    
    def _assess_road_quality(self, features: Dict[str, Any]) -> str:
        """Assess road surface quality."""
        try:
            texture_features = features.get("texture_features", {})
            smoothness_score = texture_features.get("smoothness_score", 0.5)
            
            if smoothness_score > 0.8:
                return "excellent"
            elif smoothness_score > 0.6:
                return "good"
            elif smoothness_score > 0.4:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Road quality assessment failed: {str(e)}")
            return "unknown"
    
    def _assess_lighting_quality(self, image: np.ndarray) -> str:
        """Assess lighting conditions in the image."""
        try:
            # Calculate average brightness
            brightness = np.mean(image)
            
            if brightness > 180:
                return "bright"
            elif brightness > 120:
                return "adequate"
            elif brightness > 60:
                return "dim"
            else:
                return "dark"
                
        except Exception as e:
            logger.error(f"Lighting assessment failed: {str(e)}")
            return "unknown"
    
    def _calculate_safety_score(
        self, 
        features: Dict[str, Any], 
        construction_present: bool, 
        lighting_quality: str
    ) -> float:
        """Calculate overall safety score."""
        try:
            base_score = 0.7  # Start with neutral safety score
            
            # Adjust for construction
            if construction_present:
                base_score -= 0.2
            
            # Adjust for lighting
            lighting_adjustments = {
                "bright": 0.2,
                "adequate": 0.1,
                "dim": -0.1,
                "dark": -0.3
            }
            base_score += lighting_adjustments.get(lighting_quality, 0)
            
            # Adjust for road quality
            texture_features = features.get("texture_features", {})
            smoothness_score = texture_features.get("smoothness_score", 0.5)
            base_score += (smoothness_score - 0.5) * 0.2
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.error(f"Safety score calculation failed: {str(e)}")
            return 0.5
    
    def _assess_signage_visibility(self, features: Dict[str, Any]) -> str:
        """Assess visibility of traffic signage."""
        try:
            color_features = features.get("color_features", {})
            
            # Signs typically have red, blue, or high contrast colors
            red_ratio = color_features.get("red_ratio", 0)
            blue_ratio = color_features.get("blue_ratio", 0)
            white_ratio = color_features.get("white_ratio", 0)
            
            total_signage_ratio = red_ratio + blue_ratio + white_ratio
            
            if total_signage_ratio > 0.1:
                return "clear"
            elif total_signage_ratio > 0.05:
                return "partially_obscured"
            elif total_signage_ratio > 0.02:
                return "obscured"
            else:
                return "missing"
                
        except Exception as e:
            logger.error(f"Signage visibility assessment failed: {str(e)}")
            return "unknown"
    
    def is_healthy(self) -> bool:
        """Check if the accessibility analyzer is healthy and ready."""
        try:
            # Test feature extraction with a small dummy image
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            features = self._extract_accessibility_features(dummy_image)
            return len(features) > 0
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
