"""Weather Transformer - Processes and transforms weather data."""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import logging
import math

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


class WeatherTransformer:
    """Transform weather data for pricing calculations.
    
    Processes weather information to determine impact on ride demand
    and optimal pricing strategies.
    """
    
    def __init__(self):
        """Initialize the weather transformer."""
        self.impact_thresholds = self._initialize_thresholds()
        self.seasonal_adjustments = self._initialize_seasonal_adjustments()
        logger.info("Weather transformer initialized")
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize weather impact thresholds."""
        return {
            'temperature': {
                'extreme_cold': -5,  # Celsius
                'cold': 5,
                'cool': 15,
                'comfortable': 25,
                'warm': 30,
                'hot': 35,
                'extreme_hot': 40
            },
            'precipitation': {
                'none': 0,
                'light': 0.5,  # mm/hour
                'moderate': 2.5,
                'heavy': 7.5,
                'extreme': 15
            },
            'wind': {
                'calm': 5,  # km/h
                'light': 15,
                'moderate': 30,
                'strong': 50,
                'extreme': 70
            },
            'visibility': {
                'excellent': 10,  # km
                'good': 5,
                'moderate': 2,
                'poor': 1,
                'very_poor': 0.5
            }
        }
    
    def _initialize_seasonal_adjustments(self) -> Dict[str, float]:
        """Initialize seasonal adjustment factors."""
        return {
            'winter': 1.2,
            'spring': 1.0,
            'summer': 0.9,
            'fall': 1.1
        }
    
    def calculate_impact_score(self, temperature: float, precipitation: float = 0, 
                              wind_speed: float = 0, visibility: float = 10,
                              humidity: float = 50) -> float:
        """Calculate weather impact score on demand.
        
        Args:
            temperature: Temperature in Celsius
            precipitation: Precipitation in mm/hour
            wind_speed: Wind speed in km/h
            visibility: Visibility in km
            humidity: Humidity percentage
        
        Returns:
            Impact score (0.5 to 2.0)
        """
        try:
            # Calculate individual impact factors
            temp_impact = self._calculate_temperature_impact(temperature)
            precip_impact = self._calculate_precipitation_impact(precipitation)
            wind_impact = self._calculate_wind_impact(wind_speed)
            visibility_impact = self._calculate_visibility_impact(visibility)
            humidity_impact = self._calculate_humidity_impact(humidity, temperature)
            
            # Combine impacts with weights
            weights = {
                'temperature': 0.25,
                'precipitation': 0.35,
                'wind': 0.15,
                'visibility': 0.15,
                'humidity': 0.10
            }
            
            combined_impact = (
                temp_impact * weights['temperature'] +
                precip_impact * weights['precipitation'] +
                wind_impact * weights['wind'] +
                visibility_impact * weights['visibility'] +
                humidity_impact * weights['humidity']
            )
            
            # Apply seasonal adjustment
            season = self._get_current_season()
            seasonal_factor = self.seasonal_adjustments.get(season, 1.0)
            
            final_impact = combined_impact * seasonal_factor
            
            # Ensure within bounds
            return round(max(0.5, min(2.0, final_impact)), 2)
            
        except Exception as e:
            logger.error(f"Failed to calculate weather impact: {str(e)}")
            return 1.0  # Neutral impact on error
    
    def _calculate_temperature_impact(self, temperature: float) -> float:
        """Calculate temperature impact on demand.
        
        Args:
            temperature: Temperature in Celsius
        
        Returns:
            Temperature impact factor
        """
        thresholds = self.impact_thresholds['temperature']
        
        if temperature < thresholds['extreme_cold']:
            return 1.8  # Very high demand in extreme cold
        elif temperature < thresholds['cold']:
            return 1.5  # High demand in cold
        elif temperature < thresholds['cool']:
            return 1.2  # Moderate increase in cool weather
        elif temperature < thresholds['comfortable']:
            return 0.9  # Comfortable weather, less demand
        elif temperature < thresholds['warm']:
            return 1.0  # Neutral in warm weather
        elif temperature < thresholds['hot']:
            return 1.3  # Increased demand in hot weather
        elif temperature < thresholds['extreme_hot']:
            return 1.6  # High demand in very hot weather
        else:
            return 1.8  # Very high demand in extreme heat
    
    def _calculate_precipitation_impact(self, precipitation: float) -> float:
        """Calculate precipitation impact on demand.
        
        Args:
            precipitation: Precipitation in mm/hour
        
        Returns:
            Precipitation impact factor
        """
        thresholds = self.impact_thresholds['precipitation']
        
        if precipitation <= thresholds['none']:
            return 1.0  # No impact
        elif precipitation <= thresholds['light']:
            return 1.2  # Light rain increases demand
        elif precipitation <= thresholds['moderate']:
            return 1.5  # Moderate rain significantly increases demand
        elif precipitation <= thresholds['heavy']:
            return 1.8  # Heavy rain greatly increases demand
        else:
            return 2.0  # Extreme precipitation maximizes demand
    
    def _calculate_wind_impact(self, wind_speed: float) -> float:
        """Calculate wind impact on demand.
        
        Args:
            wind_speed: Wind speed in km/h
        
        Returns:
            Wind impact factor
        """
        thresholds = self.impact_thresholds['wind']
        
        if wind_speed <= thresholds['calm']:
            return 1.0  # No impact
        elif wind_speed <= thresholds['light']:
            return 1.05  # Slight increase
        elif wind_speed <= thresholds['moderate']:
            return 1.15  # Moderate increase
        elif wind_speed <= thresholds['strong']:
            return 1.3  # Significant increase
        else:
            return 1.5  # High impact from extreme wind
    
    def _calculate_visibility_impact(self, visibility: float) -> float:
        """Calculate visibility impact on demand.
        
        Args:
            visibility: Visibility in km
        
        Returns:
            Visibility impact factor
        """
        thresholds = self.impact_thresholds['visibility']
        
        if visibility >= thresholds['excellent']:
            return 1.0  # No impact
        elif visibility >= thresholds['good']:
            return 1.05  # Slight increase
        elif visibility >= thresholds['moderate']:
            return 1.2  # Moderate increase
        elif visibility >= thresholds['poor']:
            return 1.4  # Significant increase
        else:
            return 1.6  # High impact from very poor visibility
    
    def _calculate_humidity_impact(self, humidity: float, temperature: float) -> float:
        """Calculate humidity impact on demand.
        
        Args:
            humidity: Humidity percentage
            temperature: Temperature in Celsius
        
        Returns:
            Humidity impact factor
        """
        # Humidity mainly affects comfort in hot weather
        if temperature < 20:
            return 1.0  # Minimal impact in cool weather
        
        # Calculate discomfort index
        discomfort = (temperature * 0.55) + (humidity * 0.2)
        
        if discomfort < 20:
            return 1.0  # Comfortable
        elif discomfort < 25:
            return 1.1  # Slightly uncomfortable
        elif discomfort < 30:
            return 1.2  # Uncomfortable
        else:
            return 1.3  # Very uncomfortable
    
    def _get_current_season(self) -> str:
        """Get current season based on date.
        
        Returns:
            Season name
        """
        month = datetime.now().month
        
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def transform_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw weather data into standardized format.
        
        Args:
            raw_data: Raw weather data from various sources
        
        Returns:
            Standardized weather data
        """
        try:
            # Extract and standardize fields
            temperature = self._extract_temperature(raw_data)
            precipitation = self._extract_precipitation(raw_data)
            wind_speed = self._extract_wind_speed(raw_data)
            visibility = self._extract_visibility(raw_data)
            humidity = self._extract_humidity(raw_data)
            
            # Calculate derived metrics
            impact_score = self.calculate_impact_score(
                temperature, precipitation, wind_speed, visibility, humidity
            )
            
            feels_like = self._calculate_feels_like(temperature, wind_speed, humidity)
            weather_category = self._categorize_weather(
                temperature, precipitation, wind_speed, visibility
            )
            
            return {
                'temperature': round(temperature, 1),
                'feels_like': round(feels_like, 1),
                'precipitation': round(precipitation, 1),
                'wind_speed': round(wind_speed, 1),
                'visibility': round(visibility, 1),
                'humidity': round(humidity, 0),
                'impact_score': impact_score,
                'weather_category': weather_category,
                'comfort_index': self._calculate_comfort_index(
                    temperature, humidity, wind_speed
                ),
                'timestamp': raw_data.get('timestamp', datetime.now(timezone.utc).isoformat())
            }
            
        except Exception as e:
            logger.error(f"Failed to transform weather data: {str(e)}")
            raise DataIngestionError(f"Weather transformation failed: {str(e)}")
    
    def _extract_temperature(self, data: Dict[str, Any]) -> float:
        """Extract temperature from raw data."""
        # Try different field names
        for field in ['temperature', 'temp', 'air_temperature', 'temperature_c']:
            if field in data:
                return float(data[field])
        
        # Check if in Fahrenheit
        for field in ['temperature_f', 'temp_f']:
            if field in data:
                return (float(data[field]) - 32) * 5/9
        
        return 20.0  # Default temperature
    
    def _extract_precipitation(self, data: Dict[str, Any]) -> float:
        """Extract precipitation from raw data."""
        for field in ['precipitation', 'precip', 'rainfall', 'rain']:
            if field in data:
                return float(data[field])
        
        # Check for rain and snow separately
        rain = data.get('rain', {}).get('1h', 0) if isinstance(data.get('rain'), dict) else 0
        snow = data.get('snow', {}).get('1h', 0) if isinstance(data.get('snow'), dict) else 0
        
        return rain + snow
    
    def _extract_wind_speed(self, data: Dict[str, Any]) -> float:
        """Extract wind speed from raw data."""
        for field in ['wind_speed', 'wind', 'windspeed']:
            if field in data:
                value = data[field]
                if isinstance(value, dict):
                    return float(value.get('speed', 0))
                return float(value)
        
        # Check for m/s and convert to km/h
        for field in ['wind_ms', 'wind_speed_ms']:
            if field in data:
                return float(data[field]) * 3.6
        
        return 0.0  # Default wind speed
    
    def _extract_visibility(self, data: Dict[str, Any]) -> float:
        """Extract visibility from raw data."""
        for field in ['visibility', 'vis', 'visibility_km']:
            if field in data:
                return float(data[field])
        
        # Check if in meters
        for field in ['visibility_m', 'vis_m']:
            if field in data:
                return float(data[field]) / 1000
        
        return 10.0  # Default visibility
    
    def _extract_humidity(self, data: Dict[str, Any]) -> float:
        """Extract humidity from raw data."""
        for field in ['humidity', 'relative_humidity', 'rh']:
            if field in data:
                return float(data[field])
        
        return 50.0  # Default humidity
    
    def _calculate_feels_like(self, temperature: float, wind_speed: float, humidity: float) -> float:
        """Calculate feels-like temperature.
        
        Args:
            temperature: Actual temperature in Celsius
            wind_speed: Wind speed in km/h
            humidity: Humidity percentage
        
        Returns:
            Feels-like temperature
        """
        # Wind chill for cold temperatures
        if temperature < 10 and wind_speed > 5:
            wind_chill = 13.12 + 0.6215 * temperature - 11.37 * (wind_speed ** 0.16) + \
                        0.3965 * temperature * (wind_speed ** 0.16)
            return wind_chill
        
        # Heat index for hot temperatures
        if temperature > 27:
            # Simplified heat index calculation
            heat_index = temperature + (humidity - 50) * 0.1
            return heat_index
        
        return temperature
    
    def _categorize_weather(self, temperature: float, precipitation: float,
                           wind_speed: float, visibility: float) -> str:
        """Categorize weather conditions.
        
        Args:
            temperature: Temperature in Celsius
            precipitation: Precipitation in mm/hour
            wind_speed: Wind speed in km/h
            visibility: Visibility in km
        
        Returns:
            Weather category
        """
        # Check for severe conditions first
        if precipitation > 7.5:
            return 'heavy_rain'
        elif precipitation > 2.5:
            return 'moderate_rain'
        elif precipitation > 0.5:
            return 'light_rain'
        elif visibility < 1:
            return 'foggy'
        elif wind_speed > 50:
            return 'windy'
        elif temperature < 0:
            return 'freezing'
        elif temperature > 35:
            return 'very_hot'
        elif temperature > 30:
            return 'hot'
        elif temperature < 10:
            return 'cold'
        elif 20 <= temperature <= 25 and precipitation == 0:
            return 'perfect'
        else:
            return 'normal'
    
    def _calculate_comfort_index(self, temperature: float, humidity: float, wind_speed: float) -> float:
        """Calculate overall comfort index.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            wind_speed: Wind speed in km/h
        
        Returns:
            Comfort index (0-1, higher is more comfortable)
        """
        # Ideal conditions
        ideal_temp = 22
        ideal_humidity = 50
        ideal_wind = 10
        
        # Calculate deviations
        temp_deviation = abs(temperature - ideal_temp) / 20
        humidity_deviation = abs(humidity - ideal_humidity) / 50
        wind_deviation = abs(wind_speed - ideal_wind) / 30
        
        # Weight the deviations
        comfort = 1.0 - (
            temp_deviation * 0.5 +
            humidity_deviation * 0.3 +
            wind_deviation * 0.2
        )
        
        return round(max(0, min(1, comfort)), 2)
    
    def get_weather_forecast_impact(self, forecast_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze forecast data for pricing strategy.
        
        Args:
            forecast_data: List of forecast data points
        
        Returns:
            Forecast analysis with recommendations
        """
        if not forecast_data:
            return {}
        
        impacts = []
        categories = []
        
        for forecast in forecast_data:
            transformed = self.transform_weather_data(forecast)
            impacts.append(transformed['impact_score'])
            categories.append(transformed['weather_category'])
        
        # Analyze trends
        avg_impact = sum(impacts) / len(impacts)
        max_impact = max(impacts)
        min_impact = min(impacts)
        
        # Find peak demand periods
        peak_periods = [
            i for i, impact in enumerate(impacts)
            if impact > avg_impact * 1.2
        ]
        
        return {
            'average_impact': round(avg_impact, 2),
            'max_impact': round(max_impact, 2),
            'min_impact': round(min_impact, 2),
            'impact_variance': round(max_impact - min_impact, 2),
            'peak_demand_periods': peak_periods,
            'dominant_weather': max(set(categories), key=categories.count),
            'recommendation': self._get_pricing_recommendation(avg_impact, max_impact)
        }
    
    def _get_pricing_recommendation(self, avg_impact: float, max_impact: float) -> str:
        """Get pricing recommendation based on weather impact.
        
        Args:
            avg_impact: Average weather impact
            max_impact: Maximum weather impact
        
        Returns:
            Pricing recommendation
        """
        if max_impact > 1.7:
            return 'prepare_surge_pricing'
        elif avg_impact > 1.3:
            return 'moderate_price_increase'
        elif avg_impact < 0.8:
            return 'consider_promotions'
        else:
            return 'maintain_standard_pricing'
