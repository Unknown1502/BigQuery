"""Location Transformer - Enriches and transforms location data."""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import logging
import math
import random

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

class RetryableError(PricingIntelligenceError):
    """Error that can be retried"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.LOW,
            **kwargs
        )

logger = get_logger(__name__)


class LocationTransformer:
    """Transform and enrich location data for pricing optimization.
    
    Processes location information to add contextual data such as
    business districts, transportation hubs, and demographic information.
    """
    
    def __init__(self):
        """Initialize the location transformer."""
        self.location_cache = {}
        self.location_database = self._initialize_location_database()
        self.poi_categories = self._initialize_poi_categories()
        logger.info("Location transformer initialized")
    
    def _initialize_location_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize simulated location database."""
        return {
            'manhattan_midtown': {
                'type': 'business_district',
                'population_density': 'very_high',
                'business_density': 'very_high',
                'transportation_hubs': ['Grand Central', 'Penn Station', 'Times Square'],
                'poi_density': 'very_high',
                'average_income': 'high',
                'coordinates': (40.7549, -73.9840)
            },
            'manhattan_financial': {
                'type': 'financial_district',
                'population_density': 'high',
                'business_density': 'very_high',
                'transportation_hubs': ['Wall Street Station', 'Fulton Center'],
                'poi_density': 'high',
                'average_income': 'very_high',
                'coordinates': (40.7074, -74.0113)
            },
            'brooklyn_downtown': {
                'type': 'mixed_use',
                'population_density': 'high',
                'business_density': 'medium',
                'transportation_hubs': ['Atlantic Terminal', 'Borough Hall'],
                'poi_density': 'medium',
                'average_income': 'medium',
                'coordinates': (40.6932, -73.9897)
            },
            'queens_astoria': {
                'type': 'residential',
                'population_density': 'medium',
                'business_density': 'low',
                'transportation_hubs': ['Astoria-Ditmars'],
                'poi_density': 'medium',
                'average_income': 'medium',
                'coordinates': (40.7720, -73.9301)
            },
            'jfk_airport': {
                'type': 'transportation_hub',
                'population_density': 'low',
                'business_density': 'low',
                'transportation_hubs': ['JFK Airport'],
                'poi_density': 'low',
                'average_income': 'medium',
                'coordinates': (40.6413, -73.7781)
            }
        }
    
    def _initialize_poi_categories(self) -> Dict[str, float]:
        """Initialize POI categories and their demand multipliers."""
        return {
            'airport': 1.5,
            'train_station': 1.4,
            'subway_station': 1.3,
            'bus_terminal': 1.2,
            'shopping_mall': 1.3,
            'business_center': 1.4,
            'hospital': 1.3,
            'university': 1.2,
            'stadium': 1.5,
            'concert_venue': 1.4,
            'restaurant_district': 1.2,
            'nightlife_district': 1.3,
            'tourist_attraction': 1.4,
            'hotel': 1.2,
            'convention_center': 1.4
        }
    
    def get_location_metadata(self, location_id: str) -> Dict[str, Any]:
        """Get enriched metadata for a location.
        
        Args:
            location_id: Location identifier
        
        Returns:
            Location metadata dictionary
        """
        try:
            # Check cache
            if location_id in self.location_cache:
                return self.location_cache[location_id]
            
            # Try to get from database
            if location_id in self.location_database:
                metadata = self.location_database[location_id].copy()
            else:
                # Generate metadata for unknown location
                metadata = self._generate_location_metadata(location_id)
            
            # Enrich with additional data
            metadata = self._enrich_location_data(metadata)
            
            # Cache the result
            self.location_cache[location_id] = metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get location metadata: {str(e)}")
            return self._get_default_metadata(location_id)
    
    def _generate_location_metadata(self, location_id: str) -> Dict[str, Any]:
        """Generate metadata for unknown location.
        
        Args:
            location_id: Location identifier
        
        Returns:
            Generated location metadata
        """
        # Use hash to generate consistent random data
        random.seed(hash(location_id))
        
        location_types = ['business_district', 'residential', 'mixed_use', 
                         'entertainment_district', 'industrial', 'suburban']
        density_levels = ['very_low', 'low', 'medium', 'high', 'very_high']
        income_levels = ['low', 'medium', 'high', 'very_high']
        
        # Generate random but consistent metadata
        metadata = {
            'type': random.choice(location_types),
            'population_density': random.choice(density_levels),
            'business_density': random.choice(density_levels),
            'transportation_hubs': self._generate_transportation_hubs(),
            'poi_density': random.choice(density_levels),
            'average_income': random.choice(income_levels),
            'coordinates': self._generate_coordinates()
        }
        
        # Reset random seed
        random.seed()
        
        return metadata
    
    def _generate_transportation_hubs(self) -> List[str]:
        """Generate list of nearby transportation hubs."""
        hub_types = ['Station', 'Terminal', 'Center', 'Plaza', 'Square']
        hub_names = ['Central', 'North', 'South', 'East', 'West', 'Main', 'Union']
        
        num_hubs = random.randint(0, 3)
        hubs = []
        
        for _ in range(num_hubs):
            name = f"{random.choice(hub_names)} {random.choice(hub_types)}"
            if name not in hubs:
                hubs.append(name)
        
        return hubs
    
    def _generate_coordinates(self) -> Tuple[float, float]:
        """Generate random coordinates (for simulation)."""
        # Generate coordinates around NYC area
        lat = 40.7128 + random.uniform(-0.2, 0.2)
        lon = -74.0060 + random.uniform(-0.2, 0.2)
        return (round(lat, 4), round(lon, 4))
    
    def _enrich_location_data(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich location metadata with derived information.
        
        Args:
            metadata: Basic location metadata
        
        Returns:
            Enriched metadata
        """
        # Add demand multiplier based on location type
        type_multipliers = {
            'business_district': 1.3,
            'financial_district': 1.4,
            'entertainment_district': 1.2,
            'transportation_hub': 1.5,
            'residential': 0.9,
            'mixed_use': 1.1,
            'industrial': 0.7,
            'suburban': 0.8
        }
        metadata['demand_multiplier'] = type_multipliers.get(metadata['type'], 1.0)
        
        # Add density score
        density_scores = {
            'very_low': 0.2,
            'low': 0.4,
            'medium': 0.6,
            'high': 0.8,
            'very_high': 1.0
        }
        
        pop_score = density_scores.get(metadata['population_density'], 0.5)
        bus_score = density_scores.get(metadata['business_density'], 0.5)
        poi_score = density_scores.get(metadata['poi_density'], 0.5)
        
        metadata['density_score'] = round((pop_score + bus_score + poi_score) / 3, 2)
        
        # Add accessibility score
        metadata['accessibility_score'] = self._calculate_accessibility_score(metadata)
        
        # Add time-based patterns
        metadata['peak_hours'] = self._get_peak_hours(metadata['type'])
        
        # Add nearby POIs
        metadata['nearby_pois'] = self._get_nearby_pois(metadata)
        
        # Add pricing zone
        metadata['pricing_zone'] = self._determine_pricing_zone(metadata)
        
        return metadata
    
    def _calculate_accessibility_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate location accessibility score.
        
        Args:
            metadata: Location metadata
        
        Returns:
            Accessibility score (0-1)
        """
        score = 0.5  # Base score
        
        # Transportation hubs increase accessibility
        num_hubs = len(metadata.get('transportation_hubs', []))
        score += min(0.3, num_hubs * 0.1)
        
        # High density areas are generally more accessible
        if metadata.get('population_density') in ['high', 'very_high']:
            score += 0.1
        
        # Business districts have good infrastructure
        if 'business' in metadata.get('type', ''):
            score += 0.1
        
        return round(min(1.0, score), 2)
    
    def _get_peak_hours(self, location_type: str) -> List[Tuple[int, int]]:
        """Get peak hours for location type.
        
        Args:
            location_type: Type of location
        
        Returns:
            List of peak hour ranges
        """
        peak_patterns = {
            'business_district': [(7, 10), (17, 20)],  # Morning and evening rush
            'financial_district': [(7, 10), (17, 19)],
            'entertainment_district': [(19, 23), (0, 2)],  # Evening and late night
            'transportation_hub': [(6, 10), (16, 20)],  # Extended rush hours
            'residential': [(7, 9), (18, 20)],
            'mixed_use': [(8, 10), (12, 14), (18, 21)],  # Multiple peaks
            'industrial': [(6, 8), (15, 17)],
            'suburban': [(7, 9), (17, 19)]
        }
        
        return peak_patterns.get(location_type, [(8, 10), (17, 19)])
    
    def _get_nearby_pois(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate nearby points of interest.
        
        Args:
            metadata: Location metadata
        
        Returns:
            List of nearby POIs
        """
        pois = []
        
        # Number of POIs based on density
        density_to_count = {
            'very_low': 1,
            'low': 2,
            'medium': 3,
            'high': 4,
            'very_high': 5
        }
        
        num_pois = density_to_count.get(metadata.get('poi_density', 'medium'), 3)
        
        # Generate POIs based on location type
        if metadata.get('type') == 'business_district':
            poi_types = ['business_center', 'restaurant_district', 'subway_station', 'hotel']
        elif metadata.get('type') == 'entertainment_district':
            poi_types = ['concert_venue', 'restaurant_district', 'nightlife_district', 'hotel']
        elif metadata.get('type') == 'transportation_hub':
            poi_types = ['airport', 'train_station', 'bus_terminal', 'hotel']
        else:
            poi_types = list(self.poi_categories.keys())
        
        selected_types = random.sample(poi_types, min(num_pois, len(poi_types)))
        
        for poi_type in selected_types:
            pois.append({
                'type': poi_type,
                'distance_km': round(random.uniform(0.1, 2.0), 1),
                'demand_impact': self.poi_categories[poi_type]
            })
        
        return pois
    
    def _determine_pricing_zone(self, metadata: Dict[str, Any]) -> str:
        """Determine pricing zone for location.
        
        Args:
            metadata: Location metadata
        
        Returns:
            Pricing zone identifier
        """
        # Determine zone based on multiple factors
        density_score = metadata.get('density_score', 0.5)
        income = metadata.get('average_income', 'medium')
        location_type = metadata.get('type', 'mixed_use')
        
        # Premium zones
        if (density_score > 0.7 and income in ['high', 'very_high']) or \
           location_type in ['business_district', 'financial_district', 'transportation_hub']:
            return 'premium'
        
        # Standard zones
        elif density_score > 0.4 or income == 'medium':
            return 'standard'
        
        # Economy zones
        else:
            return 'economy'
    
    def _get_default_metadata(self, location_id: str) -> Dict[str, Any]:
        """Get default metadata for error cases.
        
        Args:
            location_id: Location identifier
        
        Returns:
            Default metadata
        """
        return {
            'location_id': location_id,
            'type': 'unknown',
            'population_density': 'medium',
            'business_density': 'medium',
            'transportation_hubs': [],
            'poi_density': 'medium',
            'average_income': 'medium',
            'demand_multiplier': 1.0,
            'density_score': 0.5,
            'accessibility_score': 0.5,
            'peak_hours': [(8, 10), (17, 19)],
            'nearby_pois': [],
            'pricing_zone': 'standard',
            'error': 'metadata_generation_failed'
        }
    
    def calculate_location_demand_factor(self, location_id: str, 
                                        current_time: datetime = None) -> float:
        """Calculate demand factor for a location.
        
        Args:
            location_id: Location identifier
            current_time: Current time (optional)
        
        Returns:
            Demand factor (0.5 to 2.0)
        """
        metadata = self.get_location_metadata(location_id)
        
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        # Base demand from location type
        base_demand = metadata.get('demand_multiplier', 1.0)
        
        # Adjust for time of day
        hour = current_time.hour
        time_factor = self._calculate_time_factor(hour, metadata.get('peak_hours', []))
        
        # Adjust for nearby POIs
        poi_factor = self._calculate_poi_factor(metadata.get('nearby_pois', []))
        
        # Adjust for density
        density_factor = 0.8 + (metadata.get('density_score', 0.5) * 0.4)
        
        # Combine factors
        total_factor = base_demand * time_factor * poi_factor * density_factor
        
        # Ensure within bounds
        return round(max(0.5, min(2.0, total_factor)), 2)
    
    def _calculate_time_factor(self, hour: int, peak_hours: List[Tuple[int, int]]) -> float:
        """Calculate time-based demand factor.
        
        Args:
            hour: Current hour
            peak_hours: List of peak hour ranges
        
        Returns:
            Time factor
        """
        # Check if current hour is in peak hours
        for start, end in peak_hours:
            if start <= hour < end:
                return 1.3  # Peak hour multiplier
        
        # Off-peak hours
        if 0 <= hour < 6:
            return 0.7  # Late night/early morning
        elif 10 <= hour < 16:
            return 0.9  # Mid-day
        else:
            return 1.0  # Normal hours
    
    def _calculate_poi_factor(self, nearby_pois: List[Dict[str, Any]]) -> float:
        """Calculate POI-based demand factor.
        
        Args:
            nearby_pois: List of nearby POIs
        
        Returns:
            POI factor
        """
        if not nearby_pois:
            return 1.0
        
        # Weight POI impacts by distance
        total_impact = 0
        total_weight = 0
        
        for poi in nearby_pois:
            distance = poi.get('distance_km', 1.0)
            impact = poi.get('demand_impact', 1.0)
            
            # Closer POIs have more impact
            weight = 1.0 / (1 + distance)
            total_impact += impact * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_impact = total_impact / total_weight
            # Scale to reasonable range
            return 0.9 + (weighted_impact - 1.0) * 0.5
        
        return 1.0
    
    def get_location_cluster(self, location_ids: List[str]) -> Dict[str, List[str]]:
        """Cluster locations by similarity.
        
        Args:
            location_ids: List of location identifiers
        
        Returns:
            Dictionary of cluster assignments
        """
        clusters = {
            'premium': [],
            'standard': [],
            'economy': []
        }
        
        for location_id in location_ids:
            metadata = self.get_location_metadata(location_id)
            zone = metadata.get('pricing_zone', 'standard')
            clusters[zone].append(location_id)
        
        return clusters
