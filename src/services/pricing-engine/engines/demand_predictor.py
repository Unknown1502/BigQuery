"""
Demand Prediction Engine
Advanced demand forecasting using BigQuery ML and real-time data streams
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

from src.shared.utils.logging_utils import get_logger
from src.shared.clients.bigquery_client import BigQueryClient
from src.shared.config.gcp_config import GCPConfig
from src.shared.models.pricing_models import DemandForecast, LocationContext
from src.shared.utils.cache_utils import CacheManager

logger = get_logger(__name__)

@dataclass
class DemandPredictionResult:
    """Result structure for demand prediction"""
    location_id: str
    prediction_timestamp: datetime
    predicted_demand: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    uncertainty_score: float
    contributing_factors: Dict[str, float]
    model_version: str
    processing_time_ms: float

class DemandPredictor:
    """
    Advanced demand prediction engine using BigQuery ML and multimodal data
    """
    
    def __init__(self, config: GCPConfig):
        self.config = config
        self.bigquery_client = BigQueryClient(config)
        self.cache_manager = CacheManager(config)
        self.model_name = f"{config.project_id}.ride_intelligence.demand_forecast_model"
        
    async def predict_demand(
        self, 
        location_id: str, 
        prediction_horizon_hours: int = 1,
        include_visual_features: bool = True
    ) -> DemandPredictionResult:
        """
        Predict demand for a specific location and time horizon
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check cache first
            cache_key = f"demand_prediction:{location_id}:{prediction_horizon_hours}"
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                logger.debug(f"Retrieved cached demand prediction for {location_id}")
                return DemandPredictionResult(**cached_result)
            
            # Gather contextual data
            context = await self._gather_location_context(location_id)
            
            # Get visual intelligence features if enabled
            visual_features = {}
            if include_visual_features:
                visual_features = await self._get_visual_features(location_id)
            
            # Prepare prediction query
            prediction_query = self._build_prediction_query(
                location_id, 
                prediction_horizon_hours, 
                context, 
                visual_features
            )
            
            # Execute prediction
            prediction_results = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.bigquery_client.execute_query, 
                prediction_query
            )
            
            if not prediction_results:
                raise ValueError(f"No prediction results for location {location_id}")
            
            result_row = prediction_results[0]
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Create result object
            result = DemandPredictionResult(
                location_id=location_id,
                prediction_timestamp=start_time,
                predicted_demand=float(result_row.get('predicted_demand', 0.0)),
                confidence_interval_lower=float(result_row.get('confidence_interval_lower', 0.0)),
                confidence_interval_upper=float(result_row.get('confidence_interval_upper', 0.0)),
                uncertainty_score=float(result_row.get('uncertainty_score', 0.0)),
                contributing_factors=self._parse_contributing_factors(result_row),
                model_version=result_row.get('model_version', 'v1.0'),
                processing_time_ms=processing_time
            )
            
            # Cache result for 5 minutes
            await self.cache_manager.set(
                cache_key, 
                result.__dict__, 
                ttl=300
            )
            
            logger.info(f"Predicted demand for {location_id}: {result.predicted_demand:.2f} "
                       f"(confidence: {1.0 - result.uncertainty_score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting demand for {location_id}: {e}")
            raise
    
    async def _gather_location_context(self, location_id: str) -> LocationContext:
        """Gather contextual information for the location"""
        
        context_query = f"""
        WITH location_info AS (
            SELECT 
                location_id,
                name,
                location_type,
                latitude,
                longitude,
                business_density,
                residential_density,
                transport_accessibility
            FROM `{self.config.project_id}.ride_intelligence.locations`
            WHERE location_id = '{location_id}'
        ),
        
        recent_activity AS (
            SELECT 
                AVG(ride_count) as avg_recent_rides,
                AVG(wait_time_minutes) as avg_wait_time,
                COUNT(*) as data_points
            FROM `{self.config.project_id}.ride_intelligence.historical_rides`
            WHERE location_id = '{location_id}'
              AND ride_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        ),
        
        weather_context AS (
            SELECT 
                temperature,
                weather_condition,
                precipitation_probability,
                wind_speed
            FROM `{self.config.project_id}.ride_intelligence.weather_data`
            WHERE location_id = '{location_id}'
              AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
            ORDER BY timestamp DESC
            LIMIT 1
        ),
        
        event_context AS (
            SELECT 
                COUNT(*) as active_events,
                AVG(expected_attendance) as avg_event_size
            FROM `{self.config.project_id}.ride_intelligence.events`
            WHERE location_id = '{location_id}'
              AND event_start <= CURRENT_TIMESTAMP()
              AND event_end >= CURRENT_TIMESTAMP()
        )
        
        SELECT 
            li.*,
            ra.avg_recent_rides,
            ra.avg_wait_time,
            ra.data_points,
            wc.temperature,
            wc.weather_condition,
            wc.precipitation_probability,
            wc.wind_speed,
            ec.active_events,
            ec.avg_event_size,
            EXTRACT(HOUR FROM CURRENT_TIMESTAMP()) as current_hour,
            EXTRACT(DAYOFWEEK FROM CURRENT_TIMESTAMP()) as day_of_week,
            CASE 
                WHEN EXTRACT(DAYOFWEEK FROM CURRENT_TIMESTAMP()) IN (1, 7) THEN 'weekend'
                ELSE 'weekday'
            END as day_type
        FROM location_info li
        LEFT JOIN recent_activity ra ON li.location_id = ra.location_id
        LEFT JOIN weather_context wc ON li.location_id = wc.location_id
        LEFT JOIN event_context ec ON li.location_id = ec.location_id
        """
        
        results = await asyncio.get_event_loop().run_in_executor(
            None, 
            self.bigquery_client.execute_query, 
            context_query
        )
        
        if not results:
            # Return default context if no data found
            return LocationContext(
                location_id=location_id,
                location_type="unknown",
                current_hour=datetime.now().hour,
                day_of_week=datetime.now().weekday() + 1,
                day_type="weekday" if datetime.now().weekday() < 5 else "weekend"
            )
        
        row = results[0]
        return LocationContext(
            location_id=location_id,
            name=row.get('name', ''),
            location_type=row.get('location_type', 'unknown'),
            latitude=float(row.get('latitude', 0.0)),
            longitude=float(row.get('longitude', 0.0)),
            business_density=float(row.get('business_density', 0.0)),
            residential_density=float(row.get('residential_density', 0.0)),
            transport_accessibility=float(row.get('transport_accessibility', 0.0)),
            avg_recent_rides=float(row.get('avg_recent_rides', 0.0)),
            avg_wait_time=float(row.get('avg_wait_time', 0.0)),
            temperature=float(row.get('temperature', 20.0)),
            weather_condition=row.get('weather_condition', 'clear'),
            precipitation_probability=float(row.get('precipitation_probability', 0.0)),
            wind_speed=float(row.get('wind_speed', 0.0)),
            active_events=int(row.get('active_events', 0)),
            avg_event_size=float(row.get('avg_event_size', 0.0)),
            current_hour=int(row.get('current_hour', datetime.now().hour)),
            day_of_week=int(row.get('day_of_week', datetime.now().weekday() + 1)),
            day_type=row.get('day_type', 'weekday')
        )
    
    async def _get_visual_features(self, location_id: str) -> Dict[str, Any]:
        """Get latest visual intelligence features for the location"""
        
        visual_query = f"""
        SELECT 
            crowd_count,
            crowd_density_per_sqm,
            dominant_activity,
            accessibility_score,
            infrastructure_quality,
            safety_score,
            confidence_score,
            analysis_timestamp
        FROM `{self.config.project_id}.ride_intelligence.image_analysis_results`
        WHERE location_id = '{location_id}'
          AND analysis_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 MINUTE)
        ORDER BY analysis_timestamp DESC
        LIMIT 1
        """
        
        results = await asyncio.get_event_loop().run_in_executor(
            None, 
            self.bigquery_client.execute_query, 
            visual_query
        )
        
        if not results:
            return {
                'crowd_count': 0,
                'crowd_density': 0.0,
                'dominant_activity': 'unknown',
                'accessibility_score': 0.5,
                'infrastructure_quality': 0.5,
                'safety_score': 0.5,
                'visual_confidence': 0.0
            }
        
        row = results[0]
        return {
            'crowd_count': int(row.get('crowd_count', 0)),
            'crowd_density': float(row.get('crowd_density_per_sqm', 0.0)),
            'dominant_activity': row.get('dominant_activity', 'unknown'),
            'accessibility_score': float(row.get('accessibility_score', 0.5)),
            'infrastructure_quality': float(row.get('infrastructure_quality', 0.5)),
            'safety_score': float(row.get('safety_score', 0.5)),
            'visual_confidence': float(row.get('confidence_score', 0.0))
        }
    
    def _build_prediction_query(
        self, 
        location_id: str, 
        horizon_hours: int, 
        context: LocationContext, 
        visual_features: Dict[str, Any]
    ) -> str:
        """Build BigQuery ML prediction query with all features"""
        
        return f"""
        WITH prediction_features AS (
            SELECT 
                '{location_id}' as location_id,
                TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL {horizon_hours} HOUR) as prediction_timestamp,
                
                -- Temporal features
                {context.current_hour} as hour_of_day,
                {context.day_of_week} as day_of_week,
                '{context.day_type}' as day_type,
                
                -- Location features
                '{context.location_type}' as location_type,
                {context.business_density} as business_density,
                {context.residential_density} as residential_density,
                {context.transport_accessibility} as transport_accessibility,
                
                -- Historical features
                {context.avg_recent_rides} as avg_recent_rides,
                {context.avg_wait_time} as avg_wait_time,
                
                -- Weather features
                {context.temperature} as temperature,
                '{context.weather_condition}' as weather_condition,
                {context.precipitation_probability} as precipitation_probability,
                {context.wind_speed} as wind_speed,
                
                -- Event features
                {context.active_events} as active_events,
                {context.avg_event_size} as avg_event_size,
                
                -- Visual intelligence features
                {visual_features.get('crowd_count', 0)} as crowd_count,
                {visual_features.get('crowd_density', 0.0)} as crowd_density,
                '{visual_features.get('dominant_activity', 'unknown')}' as dominant_activity,
                {visual_features.get('accessibility_score', 0.5)} as accessibility_score,
                {visual_features.get('infrastructure_quality', 0.5)} as infrastructure_quality,
                {visual_features.get('safety_score', 0.5)} as safety_score,
                {visual_features.get('visual_confidence', 0.0)} as visual_confidence
        )
        
        SELECT 
            location_id,
            prediction_timestamp,
            forecast_value as predicted_demand,
            prediction_interval_lower_bound as confidence_interval_lower,
            prediction_interval_upper_bound as confidence_interval_upper,
            
            -- Calculate uncertainty score
            (prediction_interval_upper_bound - prediction_interval_lower_bound) / 
            (2 * GREATEST(forecast_value, 1.0)) as uncertainty_score,
            
            -- Contributing factors (simplified feature importance)
            STRUCT(
                crowd_count * 0.15 as visual_crowd_impact,
                accessibility_score * 0.10 as accessibility_impact,
                CASE weather_condition 
                    WHEN 'rain' THEN 0.20 
                    WHEN 'snow' THEN 0.25 
                    ELSE 0.05 
                END as weather_impact,
                active_events * 0.30 as event_impact,
                CASE 
                    WHEN hour_of_day BETWEEN 7 AND 9 THEN 0.25
                    WHEN hour_of_day BETWEEN 17 AND 19 THEN 0.25
                    ELSE 0.10
                END as time_impact
            ) as contributing_factors,
            
            'v2.1' as model_version
            
        FROM ML.FORECAST(
            MODEL `{self.model_name}`,
            STRUCT({horizon_hours} as horizon, 0.95 as confidence_level),
            (SELECT * FROM prediction_features)
        )
        """
    
    def _parse_contributing_factors(self, result_row: Dict[str, Any]) -> Dict[str, float]:
        """Parse contributing factors from BigQuery result"""
        factors = result_row.get('contributing_factors', {})
        
        if isinstance(factors, dict):
            return {
                'visual_crowd_impact': float(factors.get('visual_crowd_impact', 0.0)),
                'accessibility_impact': float(factors.get('accessibility_impact', 0.0)),
                'weather_impact': float(factors.get('weather_impact', 0.0)),
                'event_impact': float(factors.get('event_impact', 0.0)),
                'time_impact': float(factors.get('time_impact', 0.0))
            }
        
        return {
            'visual_crowd_impact': 0.0,
            'accessibility_impact': 0.0,
            'weather_impact': 0.0,
            'event_impact': 0.0,
            'time_impact': 0.0
        }
    
    async def batch_predict_demand(
        self, 
        location_ids: List[str], 
        prediction_horizon_hours: int = 1
    ) -> List[DemandPredictionResult]:
        """Predict demand for multiple locations efficiently"""
        
        # Process predictions concurrently
        tasks = [
            self.predict_demand(location_id, prediction_horizon_hours)
            for location_id in location_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to predict demand for {location_ids[i]}: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def get_demand_trends(
        self, 
        location_id: str, 
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """Get historical demand trends for analysis"""
        
        trends_query = f"""
        SELECT 
            TIMESTAMP_TRUNC(ride_timestamp, HOUR) as hour_timestamp,
            COUNT(*) as actual_demand,
            AVG(wait_time_minutes) as avg_wait_time,
            AVG(surge_multiplier) as avg_surge
        FROM `{self.config.project_id}.ride_intelligence.historical_rides`
        WHERE location_id = '{location_id}'
          AND ride_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours_back} HOUR)
        GROUP BY hour_timestamp
        ORDER BY hour_timestamp DESC
        """
        
        results = await asyncio.get_event_loop().run_in_executor(
            None, 
            self.bigquery_client.execute_query, 
            trends_query
        )
        
        return [dict(row) for row in results] if results else []
    
    async def evaluate_prediction_accuracy(
        self, 
        location_id: str, 
        evaluation_hours: int = 24
    ) -> Dict[str, float]:
        """Evaluate prediction accuracy over recent period"""
        
        accuracy_query = f"""
        WITH predictions AS (
            SELECT 
                location_id,
                prediction_timestamp,
                predicted_demand,
                confidence_interval_lower,
                confidence_interval_upper
            FROM `{self.config.project_id}.ride_intelligence.model_predictions`
            WHERE location_id = '{location_id}'
              AND prediction_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {evaluation_hours} HOUR)
              AND prediction_type = 'demand_forecast'
        ),
        
        actuals AS (
            SELECT 
                TIMESTAMP_TRUNC(ride_timestamp, HOUR) as hour_timestamp,
                COUNT(*) as actual_demand
            FROM `{self.config.project_id}.ride_intelligence.historical_rides`
            WHERE location_id = '{location_id}'
              AND ride_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {evaluation_hours} HOUR)
            GROUP BY hour_timestamp
        )
        
        SELECT 
            COUNT(*) as total_predictions,
            AVG(ABS(p.predicted_demand - a.actual_demand)) as mean_absolute_error,
            SQRT(AVG(POW(p.predicted_demand - a.actual_demand, 2))) as root_mean_square_error,
            AVG(ABS(p.predicted_demand - a.actual_demand) / GREATEST(a.actual_demand, 1)) as mean_absolute_percentage_error,
            AVG(CASE 
                WHEN a.actual_demand BETWEEN p.confidence_interval_lower AND p.confidence_interval_upper 
                THEN 1.0 ELSE 0.0 
            END) as confidence_interval_coverage
        FROM predictions p
        JOIN actuals a ON TIMESTAMP_TRUNC(p.prediction_timestamp, HOUR) = a.hour_timestamp
        """
        
        results = await asyncio.get_event_loop().run_in_executor(
            None, 
            self.bigquery_client.execute_query, 
            accuracy_query
        )
        
        if not results:
            return {
                'total_predictions': 0,
                'mean_absolute_error': 0.0,
                'root_mean_square_error': 0.0,
                'mean_absolute_percentage_error': 0.0,
                'confidence_interval_coverage': 0.0
            }
        
        return dict(results[0])
