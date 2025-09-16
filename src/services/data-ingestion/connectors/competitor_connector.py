"""
Competitor Data Connector - Monitors competitor pricing and availability.
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import logging
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.shared.utils.logging_utils import get_logger
from src.shared.utils.error_handling import DataIngestionError
from src.shared.config.settings import settings

logger = get_logger(__name__)


class CompetitorConnector:
    """
    Connector for competitor pricing data.
    
    Monitors competitor ride-sharing services for pricing intelligence
    and market positioning strategies.
    """
    
    def __init__(self):
        """Initialize the competitor connector."""
        self.competitors = self._initialize_competitors()
        self.cache = {}
        self.cache_ttl = 180  # 3 minutes cache for competitor data
        self.api_endpoints = self._setup_api_endpoints()
        self.executor = ThreadPoolExecutor(max_workers=5)
        logger.info("Competitor connector initialized")
    
    def _initialize_competitors(self) -> List[Dict[str, Any]]:
        """Initialize competitor configurations."""
        return [
            {
                'name': 'UberX',
                'base_price': 2.50,
                'per_km': 1.50,
                'per_minute': 0.35,
                'surge_range': (1.0, 3.0),
                'market_share': 0.35
            },
            {
                'name': 'Lyft',
                'base_price': 2.25,
                'per_km': 1.45,
                'per_minute': 0.32,
                'surge_range': (1.0, 2.5),
                'market_share': 0.25
            },
            {
                'name': 'Bolt',
                'base_price': 2.00,
                'per_km': 1.35,
                'per_minute': 0.30,
                'surge_range': (1.0, 2.0),
                'market_share': 0.15
            },
            {
                'name': 'DiDi',
                'base_price': 2.10,
                'per_km': 1.40,
                'per_minute': 0.31,
                'surge_range': (1.0, 2.2),
                'market_share': 0.10
            },
            {
                'name': 'Local Taxi',
                'base_price': 3.50,
                'per_km': 2.00,
                'per_minute': 0.50,
                'surge_range': (1.0, 1.5),
                'market_share': 0.15
            }
        ]
    
    def _setup_api_endpoints(self) -> Dict[str, str]:
        """Setup API endpoints for different competitors."""
        return {
            'UberX': 'https://api.uber.com/v1/estimates/price',
            'Lyft': 'https://api.lyft.com/v1/cost',
            'Bolt': 'https://api.bolt.eu/v1/pricing',
            'DiDi': 'https://api.didiglobal.com/v1/estimate',
            'Local Taxi': None  # No API, use estimation
        }
    
    def fetch_competitor_prices(self, location_id: str, origin: tuple = None, destination: tuple = None) -> List[Dict[str, Any]]:
        """
        Fetch current competitor prices for a route.
        
        Args:
            location_id: Unique identifier for the location
            origin: Tuple of (lat, lon) for origin
            destination: Tuple of (lat, lon) for destination
        
        Returns:
            List of competitor pricing data
        """
        try:
            # Check cache first
            cache_key = f"competitors_{location_id}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug(f"Returning cached competitor data for {location_id}")
                    return cached_data
            
            # Fetch from all competitors in parallel
            competitor_prices = []
            futures = []
            
            for competitor in self.competitors:
                future = self.executor.submit(
                    self._fetch_single_competitor,
                    competitor,
                    location_id,
                    origin or (40.7128, -74.0060),
                    destination or (40.7580, -73.9855)
                )
                futures.append((future, competitor))
            
            # Collect results
            for future, competitor in futures:
                try:
                    result = future.result(timeout=5)
                    competitor_prices.append(result)
                except Exception as e:
                    logger.error(f"Failed to fetch {competitor['name']} pricing: {str(e)}")
                    # Add fallback data
                    competitor_prices.append(
                        self._generate_fallback_pricing(competitor, location_id)
                    )
            
            # Cache the results
            self.cache[cache_key] = (competitor_prices, time.time())
            
            return competitor_prices
            
        except Exception as e:
            logger.error(f"Failed to fetch competitor prices: {str(e)}")
            raise DataIngestionError(f"Competitor price fetch failed: {str(e)}")
    
    def _fetch_single_competitor(self, competitor: Dict[str, Any], location_id: str, 
                                origin: tuple, destination: tuple) -> Dict[str, Any]:
        """
        Fetch pricing from a single competitor.
        
        Args:
            competitor: Competitor configuration
            location_id: Location identifier
            origin: Origin coordinates
            destination: Destination coordinates
        
        Returns:
            Competitor pricing data
        """
        endpoint = self.api_endpoints.get(competitor['name'])
        
        # If no real API endpoint, generate simulated data
        if not endpoint or not hasattr(settings, f"{competitor['name'].lower()}_api_key"):
            return self._generate_simulated_pricing(competitor, location_id, origin, destination)
        
        try:
            # Make API request (would be real in production)
            api_key = getattr(settings, f"{competitor['name'].lower()}_api_key")
            headers = {'Authorization': f'Bearer {api_key}'}
            params = {
                'start_latitude': origin[0],
                'start_longitude': origin[1],
                'end_latitude': destination[0],
                'end_longitude': destination[1]
            }
            
            response = requests.get(endpoint, headers=headers, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_api_response(competitor['name'], data)
            
        except Exception as e:
            logger.warning(f"API call failed for {competitor['name']}: {str(e)}")
            return self._generate_simulated_pricing(competitor, location_id, origin, destination)
    
    def _generate_simulated_pricing(self, competitor: Dict[str, Any], location_id: str,
                                   origin: tuple, destination: tuple) -> Dict[str, Any]:
        """
        Generate simulated competitor pricing.
        
        Args:
            competitor: Competitor configuration
            location_id: Location identifier
            origin: Origin coordinates
            destination: Destination coordinates
        
        Returns:
            Simulated pricing data
        """
        # Calculate distance (simplified)
        distance_km = self._calculate_distance(origin, destination)
        
        # Estimate duration based on distance and time of day
        current_hour = datetime.now().hour
        is_rush_hour = (7 <= current_hour <= 9) or (17 <= current_hour <= 19)
        
        # Base duration calculation
        avg_speed = 25 if is_rush_hour else 35  # km/h
        duration_minutes = (distance_km / avg_speed) * 60
        
        # Calculate surge multiplier
        surge_multiplier = self._calculate_surge(competitor, current_hour, is_rush_hour)
        
        # Calculate base price
        base_price = (
            competitor['base_price'] +
            (competitor['per_km'] * distance_km) +
            (competitor['per_minute'] * duration_minutes)
        )
        
        # Apply surge
        total_price = base_price * surge_multiplier
        
        # Add some randomness
        total_price *= random.uniform(0.95, 1.05)
        
        # Calculate wait time
        base_wait = random.uniform(3, 8)
        if is_rush_hour:
            base_wait *= random.uniform(1.5, 2.5)
        
        return {
            'competitor_name': competitor['name'],
            'location_id': location_id,
            'base_price': round(base_price, 2),
            'surge_multiplier': round(surge_multiplier, 2),
            'total_price': round(total_price, 2),
            'currency': 'USD',
            'estimated_duration_minutes': round(duration_minutes, 1),
            'estimated_distance_km': round(distance_km, 2),
            'estimated_wait_time_minutes': round(base_wait, 1),
            'availability': self._calculate_availability(competitor, is_rush_hour),
            'service_type': self._get_service_type(competitor['name']),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'is_simulated': True
        }
    
    def _calculate_distance(self, origin: tuple, destination: tuple) -> float:
        """
        Calculate distance between two points (simplified).
        
        Args:
            origin: (lat, lon) tuple
            destination: (lat, lon) tuple
        
        Returns:
            Distance in kilometers
        """
        # Simplified distance calculation
        # In production, would use proper geodesic distance
        import math
        
        lat1, lon1 = origin
        lat2, lon2 = destination
        
        # Haversine formula (simplified)
        R = 6371  # Earth's radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        # Add some route inefficiency
        return distance * random.uniform(1.1, 1.3)
    
    def _calculate_surge(self, competitor: Dict[str, Any], hour: int, is_rush_hour: bool) -> float:
        """
        Calculate surge pricing multiplier.
        
        Args:
            competitor: Competitor configuration
            hour: Current hour
            is_rush_hour: Whether it's rush hour
        
        Returns:
            Surge multiplier
        """
        min_surge, max_surge = competitor['surge_range']
        
        # Base surge calculation
        if is_rush_hour:
            base_surge = random.uniform(1.5, max_surge)
        elif 22 <= hour or hour <= 2:  # Late night
            base_surge = random.uniform(1.2, 1.8)
        elif 10 <= hour <= 16:  # Mid-day
            base_surge = random.uniform(min_surge, 1.2)
        else:
            base_surge = random.uniform(min_surge, 1.5)
        
        # Add event-based surge (random events)
        if random.random() < 0.1:  # 10% chance of special event
            base_surge *= random.uniform(1.2, 1.5)
        
        return min(max_surge, base_surge)
    
    def _calculate_availability(self, competitor: Dict[str, Any], is_rush_hour: bool) -> str:
        """
        Calculate driver availability.
        
        Args:
            competitor: Competitor configuration
            is_rush_hour: Whether it's rush hour
        
        Returns:
            Availability status
        """
        # Market share affects availability
        availability_score = competitor['market_share']
        
        if is_rush_hour:
            availability_score *= 0.6
        
        if availability_score > 0.2:
            return 'high'
        elif availability_score > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def _get_service_type(self, competitor_name: str) -> str:
        """Get service type for competitor."""
        service_types = {
            'UberX': 'economy',
            'Lyft': 'economy',
            'Bolt': 'economy',
            'DiDi': 'economy',
            'Local Taxi': 'standard'
        }
        return service_types.get(competitor_name, 'economy')
    
    def _parse_api_response(self, competitor_name: str, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse API response from competitor.
        
        Args:
            competitor_name: Name of competitor
            response_data: Raw API response
        
        Returns:
            Standardized pricing data
        """
        # This would contain actual parsing logic for each competitor's API
        # For now, return simulated data
        return {
            'competitor_name': competitor_name,
            'base_price': response_data.get('base_fare', 0),
            'surge_multiplier': response_data.get('surge', 1.0),
            'total_price': response_data.get('total', 0),
            'currency': response_data.get('currency', 'USD'),
            'estimated_duration_minutes': response_data.get('duration', 0),
            'estimated_distance_km': response_data.get('distance', 0),
            'estimated_wait_time_minutes': response_data.get('wait_time', 5),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_market_analysis(self, location_id: str) -> Dict[str, Any]:
        """
        Analyze competitor market positioning.
        
        Args:
            location_id: Location identifier
        
        Returns:
            Market analysis data
        """
        competitor_prices = self.fetch_competitor_prices(location_id)
        
        if not competitor_prices:
            return {}
        
        prices = [c['total_price'] for c in competitor_prices]
        surge_levels = [c['surge_multiplier'] for c in competitor_prices]
        wait_times = [c['estimated_wait_time_minutes'] for c in competitor_prices]
        
        return {
            'location_id': location_id,
            'average_price': round(sum(prices) / len(prices), 2),
            'min_price': round(min(prices), 2),
            'max_price': round(max(prices), 2),
            'price_range': round(max(prices) - min(prices), 2),
            'average_surge': round(sum(surge_levels) / len(surge_levels), 2),
            'average_wait_time': round(sum(wait_times) / len(wait_times), 1),
            'competitor_count': len(competitor_prices),
            'price_leader': min(competitor_prices, key=lambda x: x['total_price'])['competitor_name'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_historical_pricing(self, location_id: str, competitor_name: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get historical pricing data for a competitor.
        
        Args:
            location_id: Location identifier
            competitor_name: Name of competitor
            days: Number of days of history
        
        Returns:
            List of historical pricing data
        """
        competitor = next((c for c in self.competitors if c['name'] == competitor_name), None)
        if not competitor:
            return []
        
        historical = []
        
        for day in range(days):
            for hour in range(0, 24, 2):  # Every 2 hours
                # Simulate historical pricing
                is_rush = (7 <= hour <= 9) or (17 <= hour <= 19)
                surge = self._calculate_surge(competitor, hour, is_rush)
                
                base_price = competitor['base_price'] + (competitor['per_km'] * 5)  # Assume 5km trip
                total_price = base_price * surge
                
                timestamp = datetime.now(timezone.utc) - timedelta(days=day, hours=datetime.now().hour - hour)
                
                historical.append({
                    'competitor_name': competitor_name,
                    'location_id': location_id,
                    'base_price': round(base_price, 2),
                    'surge_multiplier': round(surge, 2),
                    'total_price': round(total_price, 2),
                    'hour': hour,
                    'day_offset': -day,
                    'is_rush_hour': is_rush,
                    'timestamp': timestamp.isoformat()
                })
        
        return historical
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
