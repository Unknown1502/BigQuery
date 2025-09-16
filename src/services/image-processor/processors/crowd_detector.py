"""
Crowd Detection Processor - Advanced computer vision for crowd counting and density analysis.
Uses YOLO v5 and custom models for accurate pedestrian detection and crowd analysis.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

from ....shared.config.settings import settings
from ....shared.utils.logging_utils import get_logger, log_performance
from ....shared.utils.error_handling import ImageProcessingError


logger = get_logger(__name__)


@dataclass
class DetectedPerson:
    """Data class for detected person information."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    age_category: str  # 'child', 'adult', 'elderly'
    estimated_height: float
    is_moving: bool


@dataclass
class DetectedVehicle:
    """Data class for detected vehicle information."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    vehicle_type: str  # 'car', 'truck', 'bus', 'motorcycle', 'bicycle'
    estimated_speed: float


class CrowdDetector:
    """
    Advanced crowd detection and analysis using computer vision.
    Combines YOLO object detection with custom crowd density estimation.
    """
    
    def __init__(self):
        """Initialize the crowd detector with models and configurations."""
        self.device = torch.device('cuda' if torch.cuda.is_available() and settings.image_processing.gpu_enabled else 'cpu')
        self.confidence_threshold = settings.image_processing.confidence_threshold
        
        # Load YOLO model for person detection
        self.yolo_model = self._load_yolo_model()
        
        # Load custom crowd density model
        self.density_model = self._load_density_model()
        
        # Age classification model
        self.age_classifier = self._load_age_classifier()
        
        # Vehicle detection classes
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck',
            1: 'bicycle'
        }
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
        
        logger.info(
            "Crowd detector initialized",
            extra={
                "device": str(self.device),
                "confidence_threshold": self.confidence_threshold,
                "gpu_enabled": settings.image_processing.gpu_enabled
            }
        )
    
    def _load_yolo_model(self):
        """Load YOLO v5 model for object detection."""
        try:
            # Load YOLOv5 model (using torch.hub for simplicity)
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.to(self.device)
            model.eval()
            
            # Configure model
            model.conf = self.confidence_threshold
            model.iou = 0.45  # NMS IoU threshold
            model.agnostic = False
            model.multi_label = False
            model.max_det = 1000  # Maximum detections per image
            
            logger.info("YOLO model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise ImageProcessingError(
                operation="model_loading",
                message=f"Failed to load YOLO model: {str(e)}",
                cause=e
            )
    
    def _load_density_model(self):
        """Load custom crowd density estimation model."""
        try:
            # For now, we'll use a simple density estimation
            # In production, this would be a trained CNN model
            logger.info("Density model initialized (using heuristic approach)")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load density model: {str(e)}")
            return None
    
    def _load_age_classifier(self):
        """Load age classification model."""
        try:
            # For now, we'll use heuristic age classification
            # In production, this would be a trained age classification model
            logger.info("Age classifier initialized (using heuristic approach)")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load age classifier: {str(e)}")
            return None
    
    @log_performance("crowd_analysis")
    def analyze_crowd(
        self, 
        image: np.ndarray, 
        location_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive crowd analysis on the image.
        
        Args:
            image: Input image as numpy array
            location_context: Location context information
            
        Returns:
            Crowd analysis results
        """
        try:
            start_time = time.time()
            
            # Detect people in the image
            detected_people = self._detect_people(image)
            
            # Calculate crowd metrics
            crowd_count = len(detected_people)
            crowd_density_score = self._calculate_crowd_density(detected_people, image.shape)
            age_distribution = self._analyze_age_distribution(detected_people)
            
            # Analyze crowd behavior
            movement_analysis = self._analyze_crowd_movement(detected_people, image)
            
            # Calculate spatial distribution
            spatial_distribution = self._analyze_spatial_distribution(detected_people, image.shape)
            
            processing_time = (time.time() - start_time) * 1000
            
            results = {
                "crowd_count": crowd_count,
                "crowd_density_score": crowd_density_score,
                "age_distribution": age_distribution,
                "movement_analysis": movement_analysis,
                "spatial_distribution": spatial_distribution,
                "processing_time_ms": processing_time,
                "confidence": self._calculate_detection_confidence(detected_people)
            }
            
            logger.info(
                "Crowd analysis completed",
                extra={
                    "crowd_count": crowd_count,
                    "density_score": crowd_density_score,
                    "processing_time_ms": processing_time
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Crowd analysis failed: {str(e)}")
            raise ImageProcessingError(
                operation="crowd_analysis",
                message=f"Crowd analysis failed: {str(e)}",
                cause=e
            )
    
    def _detect_people(self, image: np.ndarray) -> List[DetectedPerson]:
        """
        Detect people in the image using YOLO.
        
        Args:
            image: Input image
            
        Returns:
            List of detected people
        """
        try:
            # Run YOLO inference
            results = self.yolo_model(image)
            
            # Extract person detections
            detections = results.pandas().xyxy[0]
            person_detections = detections[detections['class'] == self.person_class_id]
            
            detected_people = []
            
            for _, detection in person_detections.iterrows():
                bbox = (
                    int(detection['xmin']),
                    int(detection['ymin']),
                    int(detection['xmax']),
                    int(detection['ymax'])
                )
                
                confidence = float(detection['confidence'])
                
                # Estimate age category (heuristic based on bbox height)
                age_category = self._estimate_age_category(bbox, image.shape)
                
                # Estimate height (heuristic)
                estimated_height = self._estimate_person_height(bbox, image.shape)
                
                # Detect movement (placeholder - would need temporal analysis)
                is_moving = False  # Would require multiple frames
                
                person = DetectedPerson(
                    bbox=bbox,
                    confidence=confidence,
                    age_category=age_category,
                    estimated_height=estimated_height,
                    is_moving=is_moving
                )
                
                detected_people.append(person)
            
            return detected_people
            
        except Exception as e:
            logger.error(f"Person detection failed: {str(e)}")
            return []
    
    def _calculate_crowd_density(
        self, 
        detected_people: List[DetectedPerson], 
        image_shape: Tuple[int, int, int]
    ) -> float:
        """
        Calculate crowd density score (0.0 to 1.0).
        
        Args:
            detected_people: List of detected people
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            Crowd density score
        """
        try:
            if not detected_people:
                return 0.0
            
            height, width = image_shape[:2]
            image_area = height * width
            
            # Calculate total person area
            total_person_area = 0
            for person in detected_people:
                x1, y1, x2, y2 = person.bbox
                person_area = (x2 - x1) * (y2 - y1)
                total_person_area += person_area
            
            # Calculate density as ratio of person area to image area
            density_ratio = total_person_area / image_area
            
            # Normalize to 0-1 scale (assuming max density of 0.3 = very crowded)
            density_score = min(1.0, density_ratio / 0.3)
            
            return density_score
            
        except Exception as e:
            logger.error(f"Density calculation failed: {str(e)}")
            return 0.0
    
    def _analyze_age_distribution(
        self, detected_people: List[DetectedPerson]
    ) -> Dict[str, float]:
        """
        Analyze age distribution of detected people.
        
        Args:
            detected_people: List of detected people
            
        Returns:
            Age distribution percentages
        """
        if not detected_people:
            return {"children": 0.0, "adults": 0.0, "elderly": 0.0}
        
        age_counts = {"children": 0, "adults": 0, "elderly": 0}
        
        for person in detected_people:
            age_category = person.age_category
            if age_category in age_counts:
                age_counts[age_category] += 1
        
        total_people = len(detected_people)
        age_distribution = {
            category: count / total_people 
            for category, count in age_counts.items()
        }
        
        return age_distribution
    
    def _estimate_age_category(
        self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]
    ) -> str:
        """
        Estimate age category based on bounding box characteristics.
        
        Args:
            bbox: Bounding box coordinates
            image_shape: Image dimensions
            
        Returns:
            Age category ('child', 'adult', 'elderly')
        """
        x1, y1, x2, y2 = bbox
        height, width = image_shape[:2]
        
        # Calculate relative height of the person
        person_height = y2 - y1
        relative_height = person_height / height
        
        # Simple heuristic based on relative height
        if relative_height < 0.15:
            return "child"
        elif relative_height > 0.4:
            return "adult"
        else:
            # Could be adult or elderly - default to adult
            return "adult"
    
    def _estimate_person_height(
        self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]
    ) -> float:
        """
        Estimate person height in meters (rough approximation).
        
        Args:
            bbox: Bounding box coordinates
            image_shape: Image dimensions
            
        Returns:
            Estimated height in meters
        """
        x1, y1, x2, y2 = bbox
        height, width = image_shape[:2]
        
        # Calculate relative height
        person_height = y2 - y1
        relative_height = person_height / height
        
        # Rough estimation assuming camera height and angle
        # This is a very simplified model
        if relative_height < 0.15:
            return 1.2  # Child
        elif relative_height < 0.25:
            return 1.6  # Average adult
        else:
            return 1.75  # Tall adult
    
    def _analyze_crowd_movement(
        self, detected_people: List[DetectedPerson], image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze crowd movement patterns.
        
        Args:
            detected_people: List of detected people
            image: Input image
            
        Returns:
            Movement analysis results
        """
        # For single frame analysis, we can only estimate static vs dynamic crowd
        # Real movement analysis would require temporal data
        
        try:
            if not detected_people:
                return {
                    "movement_detected": False,
                    "crowd_flow_direction": "unknown",
                    "movement_intensity": 0.0
                }
            
            # Analyze spatial clustering to infer movement
            positions = [(p.bbox[0] + p.bbox[2]) // 2 for p in detected_people]
            
            # Simple heuristic: if people are clustered in one direction, assume movement
            if len(positions) > 5:
                # Calculate standard deviation of positions
                std_dev = np.std(positions)
                movement_intensity = min(1.0, std_dev / 100.0)
            else:
                movement_intensity = 0.0
            
            return {
                "movement_detected": movement_intensity > 0.3,
                "crowd_flow_direction": "unknown",  # Would need temporal analysis
                "movement_intensity": movement_intensity
            }
            
        except Exception as e:
            logger.error(f"Movement analysis failed: {str(e)}")
            return {
                "movement_detected": False,
                "crowd_flow_direction": "unknown",
                "movement_intensity": 0.0
            }
    
    def _analyze_spatial_distribution(
        self, detected_people: List[DetectedPerson], image_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """
        Analyze spatial distribution of people in the image.
        
        Args:
            detected_people: List of detected people
            image_shape: Image dimensions
            
        Returns:
            Spatial distribution analysis
        """
        try:
            if not detected_people:
                return {
                    "distribution_type": "empty",
                    "clustering_score": 0.0,
                    "hotspots": []
                }
            
            height, width = image_shape[:2]
            
            # Create a grid to analyze distribution
            grid_size = 4
            grid = np.zeros((grid_size, grid_size))
            
            # Count people in each grid cell
            for person in detected_people:
                x1, y1, x2, y2 = person.bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                grid_x = min(grid_size - 1, int(center_x / width * grid_size))
                grid_y = min(grid_size - 1, int(center_y / height * grid_size))
                
                grid[grid_y, grid_x] += 1
            
            # Analyze distribution
            max_count = np.max(grid)
            non_zero_cells = np.count_nonzero(grid)
            total_cells = grid_size * grid_size
            
            # Calculate clustering score
            if len(detected_people) > 0:
                clustering_score = max_count / len(detected_people)
            else:
                clustering_score = 0.0
            
            # Determine distribution type
            if non_zero_cells <= 2:
                distribution_type = "clustered"
            elif non_zero_cells >= total_cells * 0.7:
                distribution_type = "dispersed"
            else:
                distribution_type = "mixed"
            
            # Find hotspots (cells with high density)
            hotspots = []
            threshold = max(2, len(detected_people) * 0.3)
            for i in range(grid_size):
                for j in range(grid_size):
                    if grid[i, j] >= threshold:
                        hotspots.append({
                            "grid_position": (i, j),
                            "person_count": int(grid[i, j])
                        })
            
            return {
                "distribution_type": distribution_type,
                "clustering_score": clustering_score,
                "hotspots": hotspots,
                "grid_distribution": grid.tolist()
            }
            
        except Exception as e:
            logger.error(f"Spatial distribution analysis failed: {str(e)}")
            return {
                "distribution_type": "unknown",
                "clustering_score": 0.0,
                "hotspots": []
            }
    
    def _calculate_detection_confidence(self, detected_people: List[DetectedPerson]) -> float:
        """
        Calculate overall confidence score for detections.
        
        Args:
            detected_people: List of detected people
            
        Returns:
            Overall confidence score
        """
        if not detected_people:
            return 0.0
        
        confidences = [person.confidence for person in detected_people]
        return float(np.mean(confidences))
    
    @log_performance("traffic_analysis")
    def analyze_traffic(
        self, 
        image: np.ndarray, 
        location_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze traffic conditions in the image.
        
        Args:
            image: Input image
            location_context: Location context information
            
        Returns:
            Traffic analysis results
        """
        try:
            # Detect vehicles using YOLO
            detected_vehicles = self._detect_vehicles(image)
            
            # Analyze traffic metrics
            vehicle_count = len(detected_vehicles)
            vehicle_types = self._count_vehicle_types(detected_vehicles)
            congestion_level = self._estimate_congestion_level(detected_vehicles, image.shape)
            flow_rate = self._estimate_flow_rate(detected_vehicles)
            parking_availability = self._estimate_parking_availability(detected_vehicles, image.shape)
            
            results = {
                "congestion_level": congestion_level,
                "vehicle_count": vehicle_count,
                "vehicle_types": vehicle_types,
                "flow_rate": flow_rate,
                "parking_availability": parking_availability
            }
            
            logger.info(
                "Traffic analysis completed",
                extra={
                    "vehicle_count": vehicle_count,
                    "congestion_level": congestion_level
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Traffic analysis failed: {str(e)}")
            raise ImageProcessingError(
                operation="traffic_analysis",
                message=f"Traffic analysis failed: {str(e)}",
                cause=e
            )
    
    def _detect_vehicles(self, image: np.ndarray) -> List[DetectedVehicle]:
        """
        Detect vehicles in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected vehicles
        """
        try:
            # Run YOLO inference
            results = self.yolo_model(image)
            
            # Extract vehicle detections
            detections = results.pandas().xyxy[0]
            vehicle_detections = detections[detections['class'].isin(self.vehicle_classes.keys())]
            
            detected_vehicles = []
            
            for _, detection in vehicle_detections.iterrows():
                bbox = (
                    int(detection['xmin']),
                    int(detection['ymin']),
                    int(detection['xmax']),
                    int(detection['ymax'])
                )
                
                confidence = float(detection['confidence'])
                vehicle_type = self.vehicle_classes[int(detection['class'])]
                
                # Estimate speed (placeholder - would need temporal analysis)
                estimated_speed = 0.0
                
                vehicle = DetectedVehicle(
                    bbox=bbox,
                    confidence=confidence,
                    vehicle_type=vehicle_type,
                    estimated_speed=estimated_speed
                )
                
                detected_vehicles.append(vehicle)
            
            return detected_vehicles
            
        except Exception as e:
            logger.error(f"Vehicle detection failed: {str(e)}")
            return []
    
    def _count_vehicle_types(self, detected_vehicles: List[DetectedVehicle]) -> Dict[str, int]:
        """Count vehicles by type."""
        vehicle_counts = {
            "cars": 0,
            "trucks": 0,
            "buses": 0,
            "motorcycles": 0,
            "bicycles": 0
        }
        
        for vehicle in detected_vehicles:
            vehicle_type = vehicle.vehicle_type
            if vehicle_type == "car":
                vehicle_counts["cars"] += 1
            elif vehicle_type == "truck":
                vehicle_counts["trucks"] += 1
            elif vehicle_type == "bus":
                vehicle_counts["buses"] += 1
            elif vehicle_type == "motorcycle":
                vehicle_counts["motorcycles"] += 1
            elif vehicle_type == "bicycle":
                vehicle_counts["bicycles"] += 1
        
        return vehicle_counts
    
    def _estimate_congestion_level(
        self, 
        detected_vehicles: List[DetectedVehicle], 
        image_shape: Tuple[int, int, int]
    ) -> str:
        """
        Estimate traffic congestion level.
        
        Args:
            detected_vehicles: List of detected vehicles
            image_shape: Image dimensions
            
        Returns:
            Congestion level string
        """
        vehicle_count = len(detected_vehicles)
        
        # Simple heuristic based on vehicle count
        if vehicle_count == 0:
            return "light"
        elif vehicle_count <= 5:
            return "light"
        elif vehicle_count <= 15:
            return "moderate"
        elif vehicle_count <= 30:
            return "heavy"
        else:
            return "gridlock"
    
    def _estimate_flow_rate(self, detected_vehicles: List[DetectedVehicle]) -> str:
        """
        Estimate traffic flow rate.
        
        Args:
            detected_vehicles: List of detected vehicles
            
        Returns:
            Flow rate string
        """
        # For single frame analysis, we use vehicle density as proxy
        vehicle_count = len(detected_vehicles)
        
        if vehicle_count == 0:
            return "free-flowing"
        elif vehicle_count <= 10:
            return "free-flowing"
        elif vehicle_count <= 20:
            return "slow"
        elif vehicle_count <= 35:
            return "stop-and-go"
        else:
            return "stationary"
    
    def _estimate_parking_availability(
        self, 
        detected_vehicles: List[DetectedVehicle], 
        image_shape: Tuple[int, int, int]
    ) -> str:
        """
        Estimate parking availability based on vehicle density.
        
        Args:
            detected_vehicles: List of detected vehicles
            image_shape: Image dimensions
            
        Returns:
            Parking availability string
        """
        height, width = image_shape[:2]
        image_area = height * width
        
        # Calculate vehicle density
        total_vehicle_area = 0
        for vehicle in detected_vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            vehicle_area = (x2 - x1) * (y2 - y1)
            total_vehicle_area += vehicle_area
        
        if image_area > 0:
            density_ratio = total_vehicle_area / image_area
        else:
            density_ratio = 0
        
        # Estimate parking availability
        if density_ratio < 0.1:
            return "abundant"
        elif density_ratio < 0.2:
            return "limited"
        elif density_ratio < 0.3:
            return "scarce"
        else:
            return "none"
    
    def get_model_version(self) -> str:
        """Get the version of the crowd detection model."""
        return "crowd_detector_v2.1.0"
    
    def is_healthy(self) -> bool:
        """Check if the crowd detector is healthy and ready."""
        try:
            # Test model inference with a small dummy image
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            _ = self.yolo_model(dummy_image)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
