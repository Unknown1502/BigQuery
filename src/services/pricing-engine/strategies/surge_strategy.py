"""
Surge Pricing Strategy - Implements dynamic surge pricing logic based on demand and supply.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import math

from src.shared.utils.logging_utils import get_logger
from src.shared.clients.bigquery_client import BigQueryClient
from src.shared.config.gcp_config import GCPConfig
from src.shared.utils.cache_manager import CacheManager


logger = get_logger(__name__)


class SurgeLevel(Enum):
    """Surge pricing levels."""
    NO_SURGE = "no_surge"
    LOW_SURGE = "low_surge"
    MEDIUM_SURGE = "medium_surge"
    HIGH_SURGE = "high_surge"
    EXTREME_SURGE = "extreme_surge"


@dataclass
class SurgeFactors:
    """Data class for surge pricing factors."""
    demand_factor: float
    supply_factor: float
    temporal_factor: float
    event_factor: float
    weather_factor: float
    competition_factor: float
    location_factor: float


@dataclass
class SurgeResult:
    """Data class for surge pricing result."""
    surge_multiplier: float
    surge_level: SurgeLevel
    factors: SurgeFactors
    reasoning: str
    confidence_score: float
    timestamp: datetime


class SurgeStrategy:
    """
    Implements intelligent surge pricing strategy with multiple factors.
    """
    
    def __init__(self):
        """Initialize the surge strategy."""
        self.bigquery_client = BigQueryClient()
        self.cache_manager = CacheManager()
        
        # Surge configuration
        self.surge_thresholds = {
            SurgeLevel.NO_SURGE: (1.0, 1.1),
            SurgeLevel.LOW_SURGE: (1.1, 1.3),
            SurgeLevel.MEDIUM_SURGE: (1.3, 1.7),
            SurgeLevel.HIGH_SURGE: (1.7, 2.5),
            SurgeLevel.EXTREME_SURGE: (2.5, 5.0)
        }
        
        # Factor weights
        self.factor_weights = {
            "demand": 0.35,
            "supply": 0.25,
            "temporal": 0.15,
            "event": 0.10,
            "weather": 0.08,
            "competition": 0.05,
            "location": 0.02
        }
        
        # Surge limits
        self.min_surge_multiplier = 0.8
        self.max_surge_multiplier = 4.0
        self.surge_smoothing_factor = 0.3  # For gradual surge changes
        
        logger.info("Surge strategy initialized")
    
    def apply_surge_logic(
        self, 
        base_multiplier: float, 
        location_id: str, 
        timestamp: datetime
    ) -> float:
        """
        Apply surge pricing logic to the base multiplier.
        
        Args:
            base_multiplier: Base surge multiplier from other calculations
            location_id: Location identifier
            timestamp: Current timestamp
            
        Returns:
            Adjusted surge multiplier
        """
        try:
            # Get surge factors
            surge_factors = self._calculate_surge_factors(location_id, timestamp)
            
            # Calculate weighted surge multiplier
            weighted_multiplier = self._calculate_weighted_multiplier(surge_factors)
            
            # Combine with base multiplier
            combined_multiplier = (base_multiplier + weighted_multiplier) / 2
            
            # Apply surge smoothing
            smoothed_multiplier = self._apply_surge_smoothing(
                combined_multiplier, location_id, timestamp
            )
            
            # Apply surge limits
            final_multiplier = max(
                self.min_surge_multiplier,
                min(self.max_surge_multiplier, smoothed_multiplier)
            )
            
            # Determine surge level
            surge_level = self._determine_surge_level(final_multiplier)
            
            logger.info(
                f"Surge logic applied for {location_id}",
                extra={
                    "base_multiplier": base_multiplier,
                    "final_multiplier": final_multiplier,
                    "surge_level": surge_level.value
                }
            )
            
            return final_multiplier
            
        except Exception as e:
            logger.error(f"Failed to apply surge logic: {str(e)}")
            return max(self.min_surge_multiplier, min(self.max_surge_multiplier, base_multiplier))
    
    async def calculate_surge_multiplier(
        self, 
        location_id: str, 
        timestamp: datetime,
        context: Optional[Dict[str, Any]] = None
    ) -> SurgeResult:
        """
        Calculate surge multiplier with detailed analysis.
        
        Args:
            location_id: Location identifier
            timestamp: Current timestamp
            context: Additional context information
            
        Returns:
            Detailed surge result
        """
        try:
            # Calculate all surge factors
            surge_factors = await self._calculate_detailed_surge_factors(
                location_id, timestamp, context
            )
            
            # Calculate weighted multiplier
            surge_multiplier = self._calculate_weighted_multiplier(surge_factors)
            
            # Apply constraints
            surge_multiplier = max(
                self.min_surge_multiplier,
                min(self.max_surge_multiplier, surge_multiplier)
            )
            
            # Determine surge level
            surge_level = self._determine_surge_level(surge_multiplier)
            
            # Generate reasoning
            reasoning = self._generate_surge_reasoning(surge_factors, surge_level)
            
            # Calculate confidence score
            confidence_score = self._calculate_surge_confidence(surge_factors)
            
            return SurgeResult(
                surge_multiplier=surge_multiplier,
                surge_level=surge_level,
                factors=surge_factors,
                reasoning=reasoning,
                confidence_score=confidence_score,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate surge multiplier: {str(e)}")
            return SurgeResult(
                surge_multiplier=1.0,
                surge_level=SurgeLevel.NO_SURGE,
                factors=self._get_default_factors(),
                reasoning="Default surge due to calculation error",
                confidence_score=0.0,
                timestamp=timestamp
            )
    
    def _calculate_surge_factors(
        self, 
        location_id: str, 
        timestamp: datetime
    ) -> SurgeFactors:
        """
        Calculate basic surge factors (synchronous version).
        
        Args:
            location_id: Location identifier
            timestamp: Current timestamp
            
        Returns:
            Surge factors
        """
        try:
            # Simplified factor calculations for synchronous use
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Temporal factor based on time patterns
            temporal_factor = self._calculate_temporal_factor(hour, day_of_week)
            
            # Default factors (would be enhanced with real data)
            demand_factor = 1.0 + (0.3 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0.0)
            supply_factor = 1.0 - (0.2 if 22 <= hour or hour <= 2 else 0.0)
            event_factor = 1.0  # Would be calculated from event data
            weather_factor = 1.0  # Would be calculated from weather data
            competition_factor = 1.0  # Would be calculated from competitor data
            location_factor = 1.0  # Would be calculated from location characteristics
            
            return SurgeFactors(
                demand_factor=demand_factor,
                supply_factor=supply_factor,
                temporal_factor=temporal_factor,
                event_factor=event_factor,
                weather_factor=weather_factor,
                competition_factor=competition_factor,
                location_factor=location_factor
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate surge factors: {str(e)}")
            return self._get_default_factors()
    
    async def _calculate_detailed_surge_factors(
        self, 
        location_id: str, 
        timestamp: datetime,
        context: Optional[Dict[str, Any]] = None
    ) -> SurgeFactors:
        """
        Calculate detailed surge factors with data queries.
        
        Args:
            location_id: Location identifier
            timestamp: Current timestamp
            context: Additional context
            
        Returns:
            Detailed surge factors
        """
        try:
            # Run factor calculations in parallel
            demand_task = asyncio.create_task(
                self._calculate_demand_factor(location_id, timestamp)
            )
            supply_task = asyncio.create_task(
                self._calculate_supply_factor(location_id, timestamp)
            )
            event_task = asyncio.create_task(
                self._calculate_event_factor(location_id, timestamp)
            )
            weather_task = asyncio.create_task(
                self._calculate_weather_factor(location_id, timestamp)
            )
            
            # Wait for all calculations
            demand_factor, supply_factor, event_factor, weather_factor = await asyncio.gather(
                demand_task, supply_task, event_task, weather_task,
                return_exceptions=True
            )
            
            # Handle exceptions and ensure float types
            demand_factor = float(demand_factor) if not isinstance(demand_factor, Exception) else 1.0
            supply_factor = float(supply_factor) if not isinstance(supply_factor, Exception) else 1.0
            event_factor = float(event_factor) if not isinstance(event_factor, Exception) else 1.0
            weather_factor = float(weather_factor) if not isinstance(weather_factor, Exception) else 1.0
            
            # Calculate other factors
            temporal_factor = self._calculate_temporal_factor(timestamp.hour, timestamp.weekday())
            competition_factor = await self._calculate_competition_factor(location_id, timestamp)
            location_factor = await self._calculate_location_factor(location_id)
            
            return SurgeFactors(
                demand_factor=demand_factor,
                supply_factor=supply_factor,
                temporal_factor=temporal_factor,
                event_factor=event_factor,
                weather_factor=weather_factor,
                competition_factor=competition_factor,
                location_factor=location_factor
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate detailed surge factors: {str(e)}")
            return self._get_default_factors()
    
    async def _calculate_demand_factor(self, location_id: str, timestamp: datetime) -> float:
        """Calculate demand-based surge factor."""
        try:
            # Query recent ride requests
            query = """
            SELECT COUNT(*) as request_count
            FROM `{project}.{dataset}.ride_requests`
            WHERE location_id = @location_id
              AND timestamp >= TIMESTAMP_SUB(@current_time, INTERVAL 15 MINUTE)
              AND status IN ('pending', 'matched')
            """
            
            results = await self.bigquery_client.execute_query(
                query, {
                    "location_id": location_id,
                    "current_time": timestamp.isoformat()
                }
            )
            
            if results:
                request_count = results[0]["request_count"]
                # Normalize demand (assuming 10+ requests = high demand)
                demand_factor = 1.0 + min(1.0, request_count / 10.0)
                return demand_factor
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Failed to calculate demand factor: {str(e)}")
            return 1.0
    
    async def _calculate_supply_factor(self, location_id: str, timestamp: datetime) -> float:
        """Calculate supply-based surge factor."""
        try:
            # Query available drivers
            query = """
            SELECT COUNT(*) as driver_count
            FROM `{project}.{dataset}.driver_locations`
            WHERE ST_DWITHIN(
                ST_GEOGPOINT(longitude, latitude),
                (SELECT ST_GEOGPOINT(longitude, latitude) FROM `{project}.{dataset}.locations` WHERE location_id = @location_id),
                1000  -- 1km radius
              )
              AND status = 'available'
              AND timestamp >= TIMESTAMP_SUB(@current_time, INTERVAL 5 MINUTE)
            """
            
            results = await self.bigquery_client.execute_query(
                query, {
                    "location_id": location_id,
                    "current_time": timestamp.isoformat()
                }
            )
            
            if results:
                driver_count = results[0]["driver_count"]
                # Inverse relationship: fewer drivers = higher surge
                supply_factor = 1.0 + max(0.0, (5 - driver_count) / 5.0)
                return min(2.0, supply_factor)  # Cap at 2x
            else:
                return 1.2  # Default to slight surge if no data
                
        except Exception as e:
            logger.error(f"Failed to calculate supply factor: {str(e)}")
            return 1.0
    
    def _calculate_temporal_factor(self, hour: int, day_of_week: int) -> float:
        """Calculate time-based surge factor."""
        try:
            base_factor = 1.0
            
            # Rush hour multipliers
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                base_factor += 0.3
            
            # Late night multiplier
            elif 22 <= hour or hour <= 2:
                base_factor += 0.2
            
            # Weekend multiplier
            if day_of_week >= 5:  # Saturday or Sunday
                base_factor += 0.15
                
                # Weekend night multiplier
                if 20 <= hour <= 2:
                    base_factor += 0.25
            
            return base_factor
            
        except Exception as e:
            logger.error(f"Failed to calculate temporal factor: {str(e)}")
            return 1.0
    
    async def _calculate_event_factor(self, location_id: str, timestamp: datetime) -> float:
        """Calculate event-based surge factor."""
        try:
            # Query nearby events
            query = """
            SELECT 
                event_type,
                expected_attendance,
                ST_DISTANCE(
                    ST_GEOGPOINT(event_longitude, event_latitude),
                    (SELECT ST_GEOGPOINT(longitude, latitude) FROM `{project}.{dataset}.locations` WHERE location_id = @location_id)
                ) as distance_meters
            FROM `{project}.{dataset}.events`
            WHERE start_time <= @current_time
              AND end_time >= @current_time
              AND ST_DWITHIN(
                ST_GEOGPOINT(event_longitude, event_latitude),
                (SELECT ST_GEOGPOINT(longitude, latitude) FROM `{project}.{dataset}.locations` WHERE location_id = @location_id),
                2000  -- 2km radius
              )
            ORDER BY distance_meters ASC
            LIMIT 5
            """
            
            results = await self.bigquery_client.execute_query(
                query, {
                    "location_id": location_id,
                    "current_time": timestamp.isoformat()
                }
            )
            
            if results:
                max_event_factor = 1.0
                for event in results:
                    attendance = event.get("expected_attendance", 0)
                    distance = event.get("distance_meters", 2000)
                    
                    # Calculate event impact based on size and proximity
                    proximity_factor = max(0.1, 1.0 - (distance / 2000.0))
                    size_factor = min(1.0, attendance / 10000.0)  # Normalize to 10k attendance
                    
                    event_factor = 1.0 + (proximity_factor * size_factor * 0.5)
                    max_event_factor = max(max_event_factor, event_factor)
                
                return max_event_factor
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Failed to calculate event factor: {str(e)}")
            return 1.0
    
    async def _calculate_weather_factor(self, location_id: str, timestamp: datetime) -> float:
        """Calculate weather-based surge factor."""
        try:
            # Query current weather conditions
            query = """
            SELECT 
                weather_condition,
                temperature_celsius,
                precipitation_mm,
                wind_speed_kmh
            FROM `{project}.{dataset}.weather_data`
            WHERE location_id = @location_id
              AND timestamp >= TIMESTAMP_SUB(@current_time, INTERVAL 1 HOUR)
            ORDER BY timestamp DESC
            LIMIT 1
            """
            
            results = await self.bigquery_client.execute_query(
                query, {
                    "location_id": location_id,
                    "current_time": timestamp.isoformat()
                }
            )
            
            if results:
                weather = results[0]
                weather_factor = 1.0
                
                # Bad weather increases demand
                condition = weather.get("weather_condition", "").lower()
                if any(bad_weather in condition for bad_weather in ["rain", "snow", "storm"]):
                    weather_factor += 0.3
                
                # Extreme temperatures
                temp = weather.get("temperature_celsius", 20)
                if temp < 0 or temp > 35:
                    weather_factor += 0.2
                
                # Heavy precipitation
                precipitation = weather.get("precipitation_mm", 0)
                if precipitation > 5:
                    weather_factor += min(0.4, precipitation / 20.0)
                
                return min(2.0, weather_factor)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Failed to calculate weather factor: {str(e)}")
            return 1.0
    
    async def _calculate_competition_factor(self, location_id: str, timestamp: datetime) -> float:
        """Calculate competition-based surge factor."""
        try:
            # Query competitor surge levels
            query = """
            SELECT AVG(surge_multiplier) as avg_competitor_surge
            FROM `{project}.{dataset}.competitor_pricing`
            WHERE location_id = @location_id
              AND timestamp >= TIMESTAMP_SUB(@current_time, INTERVAL 30 MINUTE)
              AND is_available = true
            """
            
            results = await self.bigquery_client.execute_query(
                query, {
                    "location_id": location_id,
                    "current_time": timestamp.isoformat()
                }
            )
            
            if results and results[0]["avg_competitor_surge"]:
                avg_competitor_surge = float(results[0]["avg_competitor_surge"])
                # Adjust our surge based on competition (slight influence)
                competition_factor = 1.0 + (avg_competitor_surge - 1.0) * 0.3
                return max(0.8, min(1.5, competition_factor))
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Failed to calculate competition factor: {str(e)}")
            return 1.0
    
    async def _calculate_location_factor(self, location_id: str) -> float:
        """Calculate location-specific surge factor."""
        try:
            # Query location characteristics
            query = """
            SELECT 
                location_type,
                business_density,
                residential_density,
                transport_accessibility
            FROM `{project}.{dataset}.location_profiles`
            WHERE location_id = @location_id
            """
            
            results = await self.bigquery_client.execute_query(
                query, {"location_id": location_id}
            )
            
            if results:
                location = results[0]
                location_factor = 1.0
                
                # High-demand location types
                location_type = location.get("location_type", "").lower()
                if location_type in ["airport", "train_station", "business_district"]:
                    location_factor += 0.1
                elif location_type in ["entertainment", "shopping_mall"]:
                    location_factor += 0.05
                
                # Business density impact
                business_density = location.get("business_density", 0.5)
                location_factor += business_density * 0.1
                
                # Transport accessibility (lower accessibility = higher surge)
                accessibility = location.get("transport_accessibility", 0.5)
                location_factor += (1.0 - accessibility) * 0.1
                
                return location_factor
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Failed to calculate location factor: {str(e)}")
            return 1.0
    
    def _calculate_weighted_multiplier(self, factors: SurgeFactors) -> float:
        """Calculate weighted surge multiplier from all factors."""
        try:
            weighted_sum = (
                factors.demand_factor * self.factor_weights["demand"] +
                factors.supply_factor * self.factor_weights["supply"] +
                factors.temporal_factor * self.factor_weights["temporal"] +
                factors.event_factor * self.factor_weights["event"] +
                factors.weather_factor * self.factor_weights["weather"] +
                factors.competition_factor * self.factor_weights["competition"] +
                factors.location_factor * self.factor_weights["location"]
            )
            
            return weighted_sum
            
        except Exception as e:
            logger.error(f"Failed to calculate weighted multiplier: {str(e)}")
            return 1.0
    
    def _apply_surge_smoothing(
        self, 
        new_multiplier: float, 
        location_id: str, 
        timestamp: datetime
    ) -> float:
        """Apply smoothing to prevent sudden surge changes."""
        try:
            # Get previous surge multiplier (would be cached in real implementation)
            previous_multiplier = 1.0  # Simplified for now
            
            # Apply exponential smoothing
            smoothed_multiplier = (
                self.surge_smoothing_factor * new_multiplier +
                (1 - self.surge_smoothing_factor) * previous_multiplier
            )
            
            return smoothed_multiplier
            
        except Exception as e:
            logger.error(f"Failed to apply surge smoothing: {str(e)}")
            return new_multiplier
    
    def _determine_surge_level(self, multiplier: float) -> SurgeLevel:
        """Determine surge level based on multiplier."""
        for level, (min_val, max_val) in self.surge_thresholds.items():
            if min_val <= multiplier < max_val:
                return level
        
        # Handle edge cases
        if multiplier >= 2.5:
            return SurgeLevel.EXTREME_SURGE
        else:
            return SurgeLevel.NO_SURGE
    
    def _generate_surge_reasoning(self, factors: SurgeFactors, level: SurgeLevel) -> str:
        """Generate human-readable reasoning for surge pricing."""
        reasons = []
        
        # Surge level description
        if level == SurgeLevel.EXTREME_SURGE:
            reasons.append("Extreme surge due to very high demand")
        elif level == SurgeLevel.HIGH_SURGE:
            reasons.append("High surge due to increased demand")
        elif level == SurgeLevel.MEDIUM_SURGE:
            reasons.append("Moderate surge pricing active")
        elif level == SurgeLevel.LOW_SURGE:
            reasons.append("Low surge pricing active")
        else:
            reasons.append("Standard pricing")
        
        # Factor contributions
        if factors.demand_factor > 1.2:
            reasons.append(f"High demand (factor: {factors.demand_factor:.2f})")
        
        if factors.supply_factor > 1.2:
            reasons.append(f"Limited supply (factor: {factors.supply_factor:.2f})")
        
        if factors.temporal_factor > 1.1:
            reasons.append(f"Peak time pricing (factor: {factors.temporal_factor:.2f})")
        
        if factors.event_factor > 1.1:
            reasons.append(f"Event impact (factor: {factors.event_factor:.2f})")
        
        if factors.weather_factor > 1.1:
            reasons.append(f"Weather impact (factor: {factors.weather_factor:.2f})")
        
        return "; ".join(reasons)
    
    def _calculate_surge_confidence(self, factors: SurgeFactors) -> float:
        """Calculate confidence score for surge pricing decision."""
        try:
            # Base confidence
            confidence = 0.7
            
            # Increase confidence if multiple factors align
            factor_values = [
                factors.demand_factor, factors.supply_factor, factors.temporal_factor,
                factors.event_factor, factors.weather_factor, factors.competition_factor
            ]
            
            # Count factors significantly above 1.0
            significant_factors = sum(1 for f in factor_values if f > 1.1)
            
            # Increase confidence with more supporting factors
            confidence += min(0.3, significant_factors * 0.05)
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Failed to calculate surge confidence: {str(e)}")
            return 0.5
    
    def _get_default_factors(self) -> SurgeFactors:
        """Get default surge factors."""
        return SurgeFactors(
            demand_factor=1.0,
            supply_factor=1.0,
            temporal_factor=1.0,
            event_factor=1.0,
            weather_factor=1.0,
            competition_factor=1.0,
            location_factor=1.0
        )
    
    def is_healthy(self) -> bool:
        """Check if the surge strategy is healthy."""
        try:
            return (
                self.bigquery_client is not None and
                self.cache_manager is not None and
                len(self.surge_thresholds) > 0 and
                len(self.factor_weights) > 0
            )
        except Exception:
            return False
