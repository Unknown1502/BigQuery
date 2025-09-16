"""Image Transformer - Processes and analyzes street imagery data."""

import json
import base64
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import logging
import random
import io

from src.shared.utils.logging_utils import get_logger
from src.shared.utils.error_handling import PricingIntelligenceError, ErrorCategory, ErrorSeverity

# Create specific error classes for data ingestion
class DataIngestionError(PricingIntelligenceError):
    """Error during data ingestion operations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

logger = get_logger(__name__)


class ImageTransformer:
    """Transform and analyze street scene images for pricing insights.
    
    Processes street camera images to extract features relevant to
    dynamic pricing such as crowd density, activity levels, and
    environmental factors.
    """
    
    def __init__(self):
        """Initialize the image transformer."""
        self.vision_client = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        self._initialize_vision_client()
        logger.info("Image transformer initialized")
    
    def _initialize_vision_client(self):
        """Initialize Google Cloud Vision client if available."""
        try:
            from google.cloud import vision
            self.vision_client = vision.ImageAnnotatorClient()
            logger.info("Google Cloud Vision client initialized")
        except ImportError:
            logger.warning("Google Cloud Vision not available, using simulation mode")
            self.vision_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Vision client: {str(e)}")
            self.vision_client = None
    
    def analyze_street_scene(self, image_url: str, location_id: str) -> Dict[str, Any]:
        """Analyze a street scene image for pricing-relevant features.
        
        Args:
            image_url: URL or path to the image
            location_id: Location identifier
        
        Returns:
            Analysis results including crowd density, activity level, etc.
        """
        try:
            # Check cache
            cache_key = f"image_{location_id}_{image_url}"
            if cache_key in self.cache:
                cached_result, _ = self.cache[cache_key]
                return cached_result
            
            # Perform analysis
            if self.vision_client:
                result = self._analyze_with_vision_api(image_url)
            else:
                result = self._simulate_image_analysis(image_url, location_id)
            
            # Cache result
            self.cache[cache_key] = (result, datetime.now(timezone.utc))
            
            return result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return self._get_default_analysis(location_id)
    
    def _analyze_with_vision_api(self, image_url: str) -> Dict[str, Any]:
        """Analyze image using Google Cloud Vision API.
        
        Args:
            image_url: URL or path to the image
        
        Returns:
            Vision API analysis results
        """
        try:
            from google.cloud import vision
            
            # Load image
            if image_url.startswith('http'):
                image = vision.Image()
                image.source.image_uri = image_url
            else:
                with io.open(image_url, 'rb') as image_file:
                    content = image_file.read()
                image = vision.Image(content=content)
            
            # Perform multiple analyses
            features = [
                vision.Feature(type_=vision.Feature.Type.FACE_DETECTION, max_results=50),
                vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=50),
                vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=20),
                vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
                vision.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION)
            ]
            
            request = vision.AnnotateImageRequest(image=image, features=features)
            response = self.vision_client.annotate_image(request=request)
            
            # Process results
            crowd_count = len(response.face_annotations)
            
            # Count people from object detection
            people_count = sum(1 for obj in response.localized_object_annotations 
                             if obj.name.lower() in ['person', 'people', 'pedestrian'])
            
            # Use maximum of face and people detection
            crowd_count = max(crowd_count, people_count)
            
            # Extract activity indicators
            activity_labels = [label.description.lower() for label in response.label_annotations]
            activity_level = self._calculate_activity_level(activity_labels)
            
            # Extract visual factors
            visual_factors = self._extract_visual_factors(response)
            
            # Calculate accessibility score
            accessibility_score = self._calculate_accessibility(response)
            
            return {
                'crowd_count': crowd_count,
                'activity_level': activity_level,
                'accessibility_score': accessibility_score,
                'visual_factors': visual_factors,
                'confidence': self._calculate_confidence(response),
                'weather_conditions': self._extract_weather_conditions(activity_labels),
                'time_of_day': self._estimate_time_of_day(response),
                'safety_score': self._calculate_safety_score(response),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Vision API analysis failed: {str(e)}")
            return self._simulate_image_analysis(image_url, "unknown")
    
    def _simulate_image_analysis(self, image_url: str, location_id: str) -> Dict[str, Any]:
        """Simulate image analysis for testing.
        
        Args:
            image_url: URL or path to the image
            location_id: Location identifier
        
        Returns:
            Simulated analysis results
        """
        # Simulate time-based patterns
        current_hour = datetime.now().hour
        
        # Base crowd density varies by time
        if 7 <= current_hour <= 9:  # Morning rush
            base_crowd = random.randint(20, 40)
        elif 12 <= current_hour <= 14:  # Lunch time
            base_crowd = random.randint(15, 30)
        elif 17 <= current_hour <= 19:  # Evening rush
            base_crowd = random.randint(25, 45)
        elif 20 <= current_hour <= 23:  # Evening
            base_crowd = random.randint(10, 25)
        else:  # Night/early morning
            base_crowd = random.randint(0, 10)
        
        # Add location-based variation
        location_modifier = hash(location_id) % 10 - 5
        crowd_count = max(0, base_crowd + location_modifier + random.randint(-5, 5))
        
        # Calculate activity level based on crowd
        if crowd_count < 10:
            activity_level = random.uniform(0.1, 0.3)
        elif crowd_count < 25:
            activity_level = random.uniform(0.3, 0.6)
        else:
            activity_level = random.uniform(0.6, 0.9)
        
        # Generate visual factors
        visual_factors = []
        factor_options = [
            'busy_street', 'commercial_area', 'residential_area',
            'transportation_hub', 'entertainment_district', 'business_district',
            'tourist_area', 'shopping_area', 'restaurant_zone'
        ]
        num_factors = random.randint(1, 3)
        visual_factors = random.sample(factor_options, num_factors)
        
        # Weather conditions
        weather_options = ['clear', 'cloudy', 'rainy', 'foggy']
        weather_weights = [0.5, 0.3, 0.15, 0.05]
        weather = random.choices(weather_options, weights=weather_weights)[0]
        
        return {
            'crowd_count': crowd_count,
            'activity_level': round(activity_level, 2),
            'accessibility_score': round(random.uniform(0.6, 1.0), 2),
            'visual_factors': visual_factors,
            'confidence': round(random.uniform(0.7, 0.95), 2),
            'weather_conditions': weather,
            'time_of_day': self._get_time_period(current_hour),
            'safety_score': round(random.uniform(0.7, 1.0), 2),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'is_simulated': True
        }
    
    def _calculate_activity_level(self, labels: List[str]) -> float:
        """Calculate activity level from image labels.
        
        Args:
            labels: List of detected labels
        
        Returns:
            Activity level score (0-1)
        """
        activity_indicators = {
            'crowd': 0.9,
            'busy': 0.8,
            'traffic': 0.7,
            'pedestrian': 0.6,
            'shopping': 0.7,
            'restaurant': 0.6,
            'entertainment': 0.8,
            'event': 0.9,
            'festival': 0.95,
            'market': 0.7,
            'rush hour': 0.85,
            'congestion': 0.8
        }
        
        score = 0.3  # Base activity level
        matches = 0
        
        for label in labels:
            for indicator, weight in activity_indicators.items():
                if indicator in label.lower():
                    score += weight * 0.1
                    matches += 1
        
        # Normalize score
        if matches > 0:
            score = min(1.0, score)
        
        return round(score, 2)
    
    def _extract_visual_factors(self, response: Any) -> List[str]:
        """Extract visual factors from Vision API response.
        
        Args:
            response: Vision API response
        
        Returns:
            List of visual factors
        """
        factors = []
        
        # Extract from labels
        if hasattr(response, 'label_annotations'):
            relevant_labels = [
                'street', 'building', 'traffic', 'pedestrian',
                'commercial', 'residential', 'urban', 'downtown',
                'shopping', 'restaurant', 'transportation'
            ]
            
            for label in response.label_annotations:
                label_lower = label.description.lower()
                for relevant in relevant_labels:
                    if relevant in label_lower and label_lower not in factors:
                        factors.append(label_lower)
        
        # Extract from objects
        if hasattr(response, 'localized_object_annotations'):
            object_types = set()
            for obj in response.localized_object_annotations:
                object_types.add(obj.name.lower())
            
            if 'car' in object_types or 'vehicle' in object_types:
                factors.append('vehicular_traffic')
            if 'person' in object_types:
                factors.append('pedestrian_activity')
            if 'bicycle' in object_types:
                factors.append('bike_friendly')
        
        return factors[:5]  # Limit to 5 factors
    
    def _calculate_accessibility(self, response: Any) -> float:
        """Calculate accessibility score from image.
        
        Args:
            response: Vision API response
        
        Returns:
            Accessibility score (0-1)
        """
        score = 0.7  # Base accessibility
        
        if hasattr(response, 'label_annotations'):
            labels = [label.description.lower() for label in response.label_annotations]
            
            # Positive indicators
            if any(word in ' '.join(labels) for word in ['sidewalk', 'pedestrian', 'walkway']):
                score += 0.1
            if any(word in ' '.join(labels) for word in ['ramp', 'accessible']):
                score += 0.15
            if any(word in ' '.join(labels) for word in ['crosswalk', 'crossing']):
                score += 0.1
            
            # Negative indicators
            if any(word in ' '.join(labels) for word in ['construction', 'barrier', 'blocked']):
                score -= 0.2
            if any(word in ' '.join(labels) for word in ['stairs', 'steep']):
                score -= 0.1
        
        return round(max(0.1, min(1.0, score)), 2)
    
    def _calculate_confidence(self, response: Any) -> float:
        """Calculate confidence score for the analysis.
        
        Args:
            response: Vision API response
        
        Returns:
            Confidence score (0-1)
        """
        confidence_scores = []
        
        if hasattr(response, 'face_annotations'):
            for face in response.face_annotations:
                if hasattr(face, 'detection_confidence'):
                    confidence_scores.append(face.detection_confidence)
        
        if hasattr(response, 'label_annotations'):
            for label in response.label_annotations:
                if hasattr(label, 'score'):
                    confidence_scores.append(label.score)
        
        if confidence_scores:
            return round(sum(confidence_scores) / len(confidence_scores), 2)
        
        return 0.8  # Default confidence
    
    def _extract_weather_conditions(self, labels: List[str]) -> str:
        """Extract weather conditions from image labels.
        
        Args:
            labels: List of detected labels
        
        Returns:
            Weather condition string
        """
        weather_keywords = {
            'rain': ['rain', 'rainy', 'wet', 'precipitation'],
            'snow': ['snow', 'snowy', 'winter', 'blizzard'],
            'fog': ['fog', 'foggy', 'mist', 'misty', 'haze'],
            'cloudy': ['cloud', 'cloudy', 'overcast'],
            'sunny': ['sun', 'sunny', 'bright', 'clear sky'],
            'night': ['night', 'dark', 'evening']
        }
        
        for condition, keywords in weather_keywords.items():
            if any(keyword in ' '.join(labels) for keyword in keywords):
                return condition
        
        return 'clear'
    
    def _estimate_time_of_day(self, response: Any) -> str:
        """Estimate time of day from image properties.
        
        Args:
            response: Vision API response
        
        Returns:
            Time period string
        """
        if hasattr(response, 'image_properties_annotation'):
            props = response.image_properties_annotation
            if hasattr(props, 'dominant_colors'):
                # Analyze dominant colors for brightness
                colors = props.dominant_colors.colors
                if colors:
                    # Calculate average brightness
                    brightness = sum(
                        (c.color.red + c.color.green + c.color.blue) / 3
                        for c in colors[:3]
                    ) / min(3, len(colors))
                    
                    if brightness < 50:
                        return 'night'
                    elif brightness < 100:
                        return 'evening'
                    elif brightness < 150:
                        return 'afternoon'
                    else:
                        return 'morning'
        
        # Fallback to current time
        hour = datetime.now().hour
        return self._get_time_period(hour)
    
    def _get_time_period(self, hour: int) -> str:
        """Get time period from hour.
        
        Args:
            hour: Hour of day (0-23)
        
        Returns:
            Time period string
        """
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    def _calculate_safety_score(self, response: Any) -> float:
        """Calculate safety score from image.
        
        Args:
            response: Vision API response
        
        Returns:
            Safety score (0-1)
        """
        score = 0.8  # Base safety score
        
        if hasattr(response, 'safe_search_annotation'):
            safe_search = response.safe_search_annotation
            
            # Check for unsafe content
            if safe_search.violence >= 3:  # LIKELY or VERY_LIKELY
                score -= 0.3
            if safe_search.adult >= 3:
                score -= 0.2
        
        if hasattr(response, 'label_annotations'):
            labels = [label.description.lower() for label in response.label_annotations]
            
            # Positive safety indicators
            if any(word in ' '.join(labels) for word in ['police', 'security', 'patrol']):
                score += 0.1
            if any(word in ' '.join(labels) for word in ['lighting', 'streetlight', 'illuminated']):
                score += 0.1
            
            # Negative safety indicators
            if any(word in ' '.join(labels) for word in ['accident', 'danger', 'hazard']):
                score -= 0.2
            if any(word in ' '.join(labels) for word in ['dark', 'abandoned', 'graffiti']):
                score -= 0.1
        
        return round(max(0.1, min(1.0, score)), 2)
    
    def _get_default_analysis(self, location_id: str) -> Dict[str, Any]:
        """Get default analysis results for fallback.
        
        Args:
            location_id: Location identifier
        
        Returns:
            Default analysis results
        """
        return {
            'crowd_count': 0,
            'activity_level': 0.5,
            'accessibility_score': 0.8,
            'visual_factors': ['unknown'],
            'confidence': 0.0,
            'weather_conditions': 'unknown',
            'time_of_day': self._get_time_period(datetime.now().hour),
            'safety_score': 0.7,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': 'analysis_failed'
        }
    
    def batch_analyze(self, image_urls: List[str], location_id: str) -> List[Dict[str, Any]]:
        """Analyze multiple images in batch.
        
        Args:
            image_urls: List of image URLs
            location_id: Location identifier
        
        Returns:
            List of analysis results
        """
        results = []
        for url in image_urls:
            try:
                result = self.analyze_street_scene(url, location_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze image {url}: {str(e)}")
                results.append(self._get_default_analysis(location_id))
        
        return results
