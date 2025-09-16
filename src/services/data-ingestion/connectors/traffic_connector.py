"""
Traffic Data Connector - Integrates with traffic monitoring systems and APIs.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import logging
import random
import requests

from src.shared.utils.logging_utils import get_logger
from src.shared.utils.error_handling import DataIngestionError
from src.shared.config.settings import settings

logger = get_logger(__name__)


class TrafficConnector:
    """
    Connector for traffic data sources.
    
    Integrates with traffic monitoring APIs and systems to fetch real-time
    traffic conditions for dynamic pricing based on congestion levels.
    """
    
    def __init__(self):
        """Initialize the traffic connector."""
        self.api_key = getattr(settings, 'traffic_api_key', None) if hasattr(settings, 'traffic_api_key') else None
        self.base_url = "https://api.tomtom.com/traffic/services/4"
        self.cache = {}
        self.cache_ttl = 120  # 2 minutes cache for traffic data
        self.traffic_patterns = self._initialize_traffic_patterns()
        logger.info("Traffic connector initialized")
    
    def _initialize_traffic_patterns(self) -> Dict[str, List[float]]:
        """Initialize typical traffic patterns for simulation."""
        return {
            'weekday': [
                0.3, 0.2, 0.2, 0.2, 0.3, 0.5,  # 00:00 - 05:59
                0.7, 0.9, 0.95, 0.8, 0.7, 0.6,  # 06:00 - 11:59
                0.7, 0.6, 0.6, 0.7, 0.8, 0.95,  # 12:00 - 17:59
                0.9, 0.7, 0.6, 0.5, 0.4, 0.3   # 18:00 - 23:59
            ],
            'weekend': [
                0.4, 0.3, 0.2, 0.2, 0.2, 0.3,  # 00:00 - 05:59
                0.4, 0.5, 0.6, 0.7, 0.8, 0.8,  # 06:00 - 11:59
                0.8, 0.8, 0.7, 0.7, 0.7, 0.7,  # 12:00 - 17:59
                0.8, 0.8, 0.7, 0.6, 0.5, 0.4   # 18:00 - 23:59
            ]
        }
    
    def fetch_traffic_data(self, location_id: str, lat: float = None, lon: float = None) -> Dict[str, Any]:
        """
        Fetch current traffic data for a location.
        
        Args:
            location_id: Unique identifier for the location
            lat: Latitude (optional)
            lon: Longitude (optional)
        
        Returns:
            Dictionary containing traffic data
        """
        try:
            # Check cache first
            cache_key = f"traffic_{location_id}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug(f"Returning cached traffic data for {location_id}")
                    return cached_data
            
            # Simulate traffic data if no API key is configured
            if not self.api_key:
                traffic_data = self._generate_simulated_traffic(location_id)
            else:
                traffic_data = self._fetch_from_api(lat or 40.7128, lon or -74.0060)
            
            # Cache the result
            self.cache[cache_key] = (traffic_data, time.time())
            
            return traffic_data
            
        except Exception as e:
            logger.error(f"Failed to fetch traffic data: {str(e)}")
            raise DataIngestionError(f"Traffic data fetch failed: {str(e)}")
    
    def _fetch_from_api(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch traffic data from TomTom Traffic API.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Processed traffic data
        """
        try:
            # TomTom Traffic Flow API
            url = f"{self.base_url}/flowSegmentData/absolute/10/json"
            params = {
                'key': self.api_key,
                'point': f"{lat},{lon}",
                'unit': 'KMPH'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            flow_data = data.get('flowSegmentData', {})
            
            # Calculate congestion level
            current_speed = flow_data.get('currentSpeed', 50)
            free_flow_speed = flow_data.get('freeFlowSpeed', 60)
            congestion_level = 1 - (current_speed / free_flow_speed) if free_flow_speed > 0 else 0
            
            return {
                'congestion_level': max(0, min(1, congestion_level)),
                'average_speed': current_speed,
                'free_flow_speed': free_flow_speed,
                'current_travel_time': flow_data.get('currentTravelTime', 0),
                'free_flow_travel_time': flow_data.get('freeFlowTravelTime', 0),
                'confidence': flow_data.get('confidence', 0.8),
                'road_closure': flow_data.get('roadClosure', False),
                'incident_count': 0,  # Would need separate incident API
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return self._generate_simulated_traffic(f"{lat},{lon}")
    
    def _generate_simulated_traffic(self, location_id: str) -> Dict[str, Any]:
        """
        Generate simulated traffic data for testing.
        
        Args:
            location_id: Location identifier
        
        Returns:
            Simulated traffic data
        """
        current_time = datetime.now()
        hour = current_time.hour
        is_weekend = current_time.weekday() >= 5
        
        # Get base congestion from pattern
        pattern = self.traffic_patterns['weekend' if is_weekend else 'weekday']
        base_congestion = pattern[hour]
        
        # Add some randomness
        congestion_level = base_congestion + random.uniform(-0.1, 0.1)
        congestion_level = max(0, min(1, congestion_level))
        
        # Calculate speeds based on congestion
        free_flow_speed = random.uniform(50, 70)  # km/h
        average_speed = free_flow_speed * (1 - congestion_level * 0.8)
        
        # Simulate incidents based on congestion
        incident_probability = congestion_level * 0.3
        incident_count = 1 if random.random() < incident_probability else 0
        
        # Add more incidents during rush hours
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            if random.random() < 0.2:
                incident_count += 1
        
        # Calculate travel times
        base_distance = 5  # km
        current_travel_time = (base_distance / average_speed) * 60 if average_speed > 0 else 30  # minutes
        free_flow_travel_time = (base_distance / free_flow_speed) * 60  # minutes
        
        return {
            'congestion_level': round(congestion_level, 2),
            'average_speed': round(average_speed, 1),
            'free_flow_speed': round(free_flow_speed, 1),
            'current_travel_time': round(current_travel_time, 1),
            'free_flow_travel_time': round(free_flow_travel_time, 1),
            'confidence': round(0.7 + random.uniform(0, 0.3), 2),
            'road_closure': random.random() < 0.02,  # 2% chance of road closure
            'incident_count': incident_count,
            'road_conditions': self._get_road_conditions(congestion_level),
            'traffic_flow': self._get_traffic_flow_description(congestion_level),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'location_id': location_id,
            'is_simulated': True
        }
    
    def _get_road_conditions(self, congestion_level: float) -> str:
        """Get road condition description based on congestion."""
        if congestion_level < 0.2:
            return "clear"
        elif congestion_level < 0.4:
            return "light"
        elif congestion_level < 0.6:
            return "moderate"
        elif congestion_level < 0.8:
            return "heavy"
        else:
            return "severe"
    
    def _get_traffic_flow_description(self, congestion_level: float) -> str:
        """Get traffic flow description."""
        if congestion_level < 0.2:
            return "free_flow"
        elif congestion_level < 0.4:
            return "stable_flow"
        elif congestion_level < 0.6:
            return "unstable_flow"
        elif congestion_level < 0.8:
            return "forced_flow"
        else:
            return "breakdown"
    
    def get_traffic_impact_factor(self, traffic_data: Dict[str, Any]) -> float:
        """
        Calculate traffic impact factor on demand.
        
        Args:
            traffic_data: Traffic data dictionary
        
        Returns:
            Impact factor (0.8 to 2.0)
        """
        factor = 1.0
        congestion = traffic_data.get('congestion_level', 0)
        
        # Higher congestion increases demand for ride services
        if congestion < 0.2:
            factor = 0.9  # Low traffic, less demand
        elif congestion < 0.4:
            factor = 1.0  # Normal
        elif congestion < 0.6:
            factor = 1.2  # Moderate traffic increases demand
        elif congestion < 0.8:
            factor = 1.5  # Heavy traffic significantly increases demand
        else:
            factor = 1.8  # Severe congestion, maximum demand
        
        # Incidents increase demand
        incidents = traffic_data.get('incident_count', 0)
        factor += incidents * 0.1
        
        # Road closures significantly increase demand
        if traffic_data.get('road_closure', False):
            factor *= 1.3
        
        # Ensure factor stays within bounds
        return max(0.8, min(2.0, factor))
    
    def get_route_alternatives(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> List[Dict[str, Any]]:
        """
        Get alternative routes between origin and destination.
        
        Args:
            origin: Tuple of (latitude, longitude)
            destination: Tuple of (latitude, longitude)
        
        Returns:
            List of route alternatives with traffic data
        """
        routes = []
        
        # Simulate 3 alternative routes
        for i in range(3):
            base_distance = random.uniform(5, 15)  # km
            congestion = random.uniform(0, 0.8)
            
            free_speed = random.uniform(40, 60)
            actual_speed = free_speed * (1 - congestion * 0.7)
            
            route = {
                'route_id': f"route_{i+1}",
                'distance_km': round(base_distance * (1 + i * 0.1), 1),
                'estimated_time_minutes': round((base_distance / actual_speed) * 60, 1),
                'congestion_level': round(congestion, 2),
                'average_speed': round(actual_speed, 1),
                'toll_cost': round(random.uniform(0, 5), 2) if random.random() < 0.3 else 0,
                'route_type': ['fastest', 'shortest', 'balanced'][i],
                'via_points': self._generate_via_points(i + 2)
            }
            routes.append(route)
        
        # Sort by estimated time
        routes.sort(key=lambda x: x['estimated_time_minutes'])
        
        return routes
    
    def _generate_via_points(self, count: int) -> List[str]:
        """Generate via point descriptions for a route."""
        streets = ['Main St', 'Broadway', '5th Ave', 'Park Ave', 'Highway 101', 'Interstate 95']
        return random.sample(streets, min(count, len(streets)))
    
    def get_historical_traffic(self, location_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get historical traffic patterns.
        
        Args:
            location_id: Location identifier
            days: Number of days of history
        
        Returns:
            List of historical traffic data points
        """
        historical = []
        
        for day in range(days):
            is_weekend = (datetime.now().weekday() - day) % 7 >= 5
            pattern = self.traffic_patterns['weekend' if is_weekend else 'weekday']
            
            for hour in range(24):
                base_congestion = pattern[hour]
                
                # Add some daily variation
                daily_modifier = random.uniform(-0.05, 0.05)
                congestion = max(0, min(1, base_congestion + daily_modifier))
                
                traffic_data = {
                    'location_id': location_id,
                    'congestion_level': round(congestion, 2),
                    'average_speed': round(60 * (1 - congestion * 0.7), 1),
                    'hour': hour,
                    'day_offset': -day,
                    'is_weekend': is_weekend,
                    'timestamp': (datetime.now(timezone.utc) - timedelta(days=day, hours=datetime.now().hour - hour)).isoformat()
                }
                
                historical.append(traffic_data)
        
        return historical
    
    def predict_traffic(self, location_id: str, hours_ahead: int = 1) -> Dict[str, Any]:
        """
        Predict future traffic conditions.
        
        Args:
            location_id: Location identifier
            hours_ahead: Hours to predict ahead
        
        Returns:
            Predicted traffic data
        """
        future_time = datetime.now() + timedelta(hours=hours_ahead)
        hour = future_time.hour
        is_weekend = future_time.weekday() >= 5
        
        pattern = self.traffic_patterns['weekend' if is_weekend else 'weekday']
        predicted_congestion = pattern[hour % 24]
        
        # Add uncertainty to prediction
        uncertainty = 0.05 * hours_ahead  # More uncertainty for longer predictions
        predicted_congestion += random.uniform(-uncertainty, uncertainty)
        predicted_congestion = max(0, min(1, predicted_congestion))
        
        return {
            'location_id': location_id,
            'predicted_congestion': round(predicted_congestion, 2),
            'predicted_speed': round(60 * (1 - predicted_congestion * 0.7), 1),
            'prediction_time': future_time.isoformat(),
            'hours_ahead': hours_ahead,
            'confidence': round(max(0.5, 1 - (hours_ahead * 0.05)), 2),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
