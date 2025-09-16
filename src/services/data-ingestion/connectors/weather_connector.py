"""
Weather Data Connector - Integrates with weather APIs and services.
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import logging
import requests

from src.shared.utils.logging_utils import get_logger
from src.shared.utils.error_handling import DataIngestionError
from src.shared.config.settings import settings

logger = get_logger(__name__)


class WeatherConnector:
    """
    Connector for weather data sources.
    
    Integrates with various weather APIs to fetch real-time weather data
    for dynamic pricing calculations based on weather conditions.
    """
    
    def __init__(self):
        """Initialize the weather connector."""
        self.api_key = getattr(settings, 'weather_api_key', None) if hasattr(settings, 'weather_api_key') else None
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        logger.info("Weather connector initialized")
    
    def fetch_current_weather(self, location_id: str, lat: float = None, lon: float = None) -> Dict[str, Any]:
        """
        Fetch current weather data for a location.
        
        Args:
            location_id: Unique identifier for the location
            lat: Latitude (optional)
            lon: Longitude (optional)
        
        Returns:
            Dictionary containing weather data
        """
        try:
            # Check cache first
            cache_key = f"weather_{location_id}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug(f"Returning cached weather data for {location_id}")
                    return cached_data
            
            # Simulate weather data if no API key is configured
            if not self.api_key:
                return self._generate_simulated_weather(location_id)
            
            # Fetch from API
            weather_data = self._fetch_from_api(lat or 40.7128, lon or -74.0060)
            
            # Cache the result
            self.cache[cache_key] = (weather_data, time.time())
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Failed to fetch weather data: {str(e)}")
            raise DataIngestionError(f"Weather data fetch failed: {str(e)}")
    
    def _fetch_from_api(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch weather data from OpenWeatherMap API.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Processed weather data
        """
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'clouds': data['clouds']['all'],
                'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                'precipitation': self._get_precipitation(data),
                'weather_condition': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return self._generate_simulated_weather(f"{lat},{lon}")
    
    def _get_precipitation(self, data: Dict[str, Any]) -> float:
        """Extract precipitation data from API response."""
        precipitation = 0.0
        
        if 'rain' in data:
            precipitation += data['rain'].get('1h', 0)
        if 'snow' in data:
            precipitation += data['snow'].get('1h', 0)
        
        return precipitation
    
    def _generate_simulated_weather(self, location_id: str) -> Dict[str, Any]:
        """
        Generate simulated weather data for testing.
        
        Args:
            location_id: Location identifier
        
        Returns:
            Simulated weather data
        """
        import random
        
        # Generate realistic weather patterns
        hour = datetime.now().hour
        
        # Temperature varies by time of day
        base_temp = 20.0
        if 6 <= hour < 12:
            temp_modifier = hour - 6  # Morning warming
        elif 12 <= hour < 18:
            temp_modifier = 6 - (hour - 12) * 0.5  # Afternoon cooling
        else:
            temp_modifier = -2  # Night cooling
        
        temperature = base_temp + temp_modifier + random.uniform(-3, 3)
        
        # Weather conditions with probabilities
        conditions = [
            ('Clear', 0.4),
            ('Clouds', 0.3),
            ('Rain', 0.2),
            ('Drizzle', 0.1)
        ]
        
        condition = self._weighted_choice(conditions)
        
        # Adjust other parameters based on condition
        if condition == 'Clear':
            clouds = random.randint(0, 20)
            precipitation = 0.0
            visibility = random.uniform(8, 10)
        elif condition == 'Clouds':
            clouds = random.randint(40, 80)
            precipitation = 0.0
            visibility = random.uniform(5, 8)
        elif condition == 'Rain':
            clouds = random.randint(70, 100)
            precipitation = random.uniform(1, 5)
            visibility = random.uniform(2, 5)
        else:  # Drizzle
            clouds = random.randint(60, 90)
            precipitation = random.uniform(0.1, 1)
            visibility = random.uniform(3, 6)
        
        return {
            'temperature': round(temperature, 1),
            'feels_like': round(temperature + random.uniform(-2, 2), 1),
            'humidity': random.randint(30, 90),
            'pressure': random.randint(1000, 1030),
            'wind_speed': round(random.uniform(0, 15), 1),
            'wind_direction': random.randint(0, 360),
            'clouds': clouds,
            'visibility': round(visibility, 1),
            'precipitation': round(precipitation, 1),
            'weather_condition': condition,
            'weather_description': condition.lower(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'location_id': location_id,
            'is_simulated': True
        }
    
    def _weighted_choice(self, choices: List[tuple]) -> str:
        """Make a weighted random choice."""
        import random
        
        total = sum(weight for _, weight in choices)
        r = random.uniform(0, total)
        upto = 0
        
        for choice, weight in choices:
            if upto + weight >= r:
                return choice
            upto += weight
        
        return choices[-1][0]
    
    def get_weather_impact_factor(self, weather_data: Dict[str, Any]) -> float:
        """
        Calculate weather impact factor on demand.
        
        Args:
            weather_data: Weather data dictionary
        
        Returns:
            Impact factor (0.5 to 2.0)
        """
        factor = 1.0
        
        # Temperature impact
        temp = weather_data.get('temperature', 20)
        if temp < 0:
            factor *= 0.7  # Very cold reduces demand
        elif temp > 35:
            factor *= 0.8  # Very hot reduces demand
        elif 20 <= temp <= 25:
            factor *= 1.2  # Ideal temperature increases demand
        
        # Precipitation impact
        precipitation = weather_data.get('precipitation', 0)
        if precipitation > 0:
            if precipitation < 1:
                factor *= 1.1  # Light rain increases demand
            elif precipitation < 5:
                factor *= 1.3  # Moderate rain increases demand
            else:
                factor *= 1.5  # Heavy rain significantly increases demand
        
        # Wind impact
        wind_speed = weather_data.get('wind_speed', 0)
        if wind_speed > 20:
            factor *= 0.9  # Strong wind reduces demand
        
        # Visibility impact
        visibility = weather_data.get('visibility', 10)
        if visibility < 2:
            factor *= 1.2  # Poor visibility increases demand
        
        # Ensure factor stays within bounds
        return max(0.5, min(2.0, factor))
    
    def get_forecast(self, location_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get weather forecast for the next N hours.
        
        Args:
            location_id: Location identifier
            hours: Number of hours to forecast
        
        Returns:
            List of hourly weather forecasts
        """
        forecasts = []
        base_weather = self.fetch_current_weather(location_id)
        
        for hour in range(hours):
            forecast = base_weather.copy()
            
            # Simulate gradual changes
            forecast['temperature'] += random.uniform(-0.5, 0.5) * hour
            forecast['humidity'] += random.randint(-2, 2)
            forecast['wind_speed'] *= random.uniform(0.9, 1.1)
            
            # Update timestamp
            forecast_time = datetime.now(timezone.utc)
            forecast_time = forecast_time.replace(hour=(forecast_time.hour + hour) % 24)
            forecast['timestamp'] = forecast_time.isoformat()
            forecast['forecast_hour'] = hour
            
            forecasts.append(forecast)
        
        return forecasts
    
    def get_historical_weather(self, location_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get historical weather data.
        
        Args:
            location_id: Location identifier
            days: Number of days of history
        
        Returns:
            List of historical weather data points
        """
        historical = []
        
        for day in range(days):
            for hour in range(0, 24, 3):  # Every 3 hours
                weather = self._generate_simulated_weather(location_id)
                
                # Adjust timestamp to past
                past_time = datetime.now(timezone.utc)
                past_time = past_time.replace(
                    day=past_time.day - day,
                    hour=hour,
                    minute=0,
                    second=0,
                    microsecond=0
                )
                weather['timestamp'] = past_time.isoformat()
                
                historical.append(weather)
        
        return historical


# For backward compatibility
import random
