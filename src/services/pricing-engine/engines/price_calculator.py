"""
Price Calculator Engine
Advanced pricing optimization using multi-objective algorithms and BigQuery AI
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.shared.utils.logging_utils import get_logger
from src.shared.clients.bigquery_client import BigQueryClient
from src.shared.config.gcp_config import GCPConfig
from src.shared.models.pricing_models import PricingCalculation, OptimizationObjective
from src.shared.utils.cache_utils import CacheManager
from .demand_predictor import DemandPredictor

logger = get_logger(__name__)

class PricingStrategy(Enum):
    """Enumeration of pricing strategies"""
    REVENUE_MAXIMIZATION = "revenue_maximization"
    UTILIZATION_OPTIMIZATION = "utilization_optimization"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    BALANCED = "balanced"
    COMPETITIVE = "competitive"

@dataclass
class PricingResult:
    """Result structure for pricing calculation"""
    location_id: str
    calculation_timestamp: datetime
    base_price: float
    surge_multiplier: float
    final_price: float
    confidence_score: float
    strategy_used: PricingStrategy
    optimization_objectives: Dict[str, float]
    contributing_factors: Dict[str, float]
    competitive_analysis: Dict[str, Any]
    reasoning: str
    model_version: str
    processing_time_ms: float

class PriceCalculator:
    """
    Advanced price calculator using multi-objective optimization
    """
    
    def __init__(self, config: GCPConfig):
        self.config = config
        self.bigquery_client = BigQueryClient(config)
        self.cache_manager = CacheManager(config)
        self.demand_predictor = DemandPredictor(config)
        
        # Pricing configuration
        self.base_price = 3.50
        self.min_surge_multiplier = 0.8
        self.max_surge_multiplier = 3.0
        self.surge_threshold = 1.5
        
        # Strategy weights for multi-objective optimization
        self.strategy_weights = {
            PricingStrategy.REVENUE_MAXIMIZATION: {
                'revenue': 0.7, 'utilization': 0.2, 'satisfaction': 0.1
            },
            PricingStrategy.UTILIZATION_OPTIMIZATION: {
                'revenue': 0.2, 'utilization': 0.7, 'satisfaction': 0.1
            },
            PricingStrategy.CUSTOMER_SATISFACTION: {
                'revenue': 0.1, 'utilization': 0.2, 'satisfaction': 0.7
            },
            PricingStrategy.BALANCED: {
                'revenue': 0.4, 'utilization': 0.3, 'satisfaction': 0.3
            },
            PricingStrategy.COMPETITIVE: {
                'revenue': 0.3, 'utilization': 0.3, 'satisfaction': 0.2, 'competitive': 0.2
            }
        }
    
    async def calculate_optimal_price(
        self, 
        location_id: str, 
        strategy: PricingStrategy = PricingStrategy.BALANCED,
        include_competitive_analysis: bool = True
    ) -> PricingResult:
        """
        Calculate optimal price using multi-objective optimization
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check cache first
            cache_key = f"price_calculation:{location_id}:{strategy.value}"
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                logger.debug(f"Retrieved cached price calculation for {location_id}")
                return PricingResult(**cached_result)
            
            # Get demand prediction
            demand_forecast = await self.demand_predictor.predict_demand(location_id)
            
            # Get current market conditions
            market_conditions = await self._get_market_conditions(location_id)
            
            # Get competitive analysis if enabled
            competitive_analysis = {}
            if include_competitive_analysis:
                competitive_analysis = await self._get_competitive_analysis(location_id)
            
            # Get visual intelligence factors
            visual_factors = await self._get_visual_intelligence_factors(location_id)
            
            # Calculate base pricing factors
            pricing_factors = await self._calculate_pricing_factors(
                location_id, 
                demand_forecast, 
                market_conditions, 
                visual_factors
            )
            
            # Apply multi-objective optimization
            optimization_result = self._optimize_price(
                pricing_factors, 
                competitive_analysis, 
                strategy
            )
            
            # Generate explanation
            reasoning = await self._generate_pricing_reasoning(
                location_id,
                pricing_factors,
                optimization_result,
                competitive_analysis
            )
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Create result object
            result = PricingResult(
                location_id=location_id,
                calculation_timestamp=start_time,
                base_price=self.base_price,
                surge_multiplier=optimization_result['surge_multiplier'],
                final_price=optimization_result['final_price'],
                confidence_score=optimization_result['confidence_score'],
                strategy_used=strategy,
                optimization_objectives=optimization_result['objectives'],
                contributing_factors=pricing_factors,
                competitive_analysis=competitive_analysis,
                reasoning=reasoning,
                model_version='v2.1',
                processing_time_ms=processing_time
            )
            
            # Store calculation in BigQuery for analytics
            await self._store_pricing_calculation(result)
            
            # Cache result for 2 minutes
            await self.cache_manager.set(
                cache_key, 
                result.__dict__, 
                ttl=120
            )
            
            logger.info(f"Calculated optimal price for {location_id}: ${result.final_price:.2f} "
                       f"(surge: {result.surge_multiplier:.2f}x, confidence: {result.confidence_score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating price for {location_id}: {e}")
            raise
    
    async def _get_market_conditions(self, location_id: str) -> Dict[str, Any]:
        """Get current market conditions for the location"""
        
        conditions_query = f"""
        WITH recent_rides AS (
            SELECT 
                COUNT(*) as ride_count_last_hour,
                AVG(wait_time_minutes) as avg_wait_time,
                AVG(surge_multiplier) as current_surge,
                STDDEV(surge_multiplier) as surge_volatility
            FROM `{self.config.project_id}.ride_intelligence.historical_rides`
            WHERE location_id = '{location_id}'
              AND ride_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
        ),
        
        driver_supply AS (
            SELECT 
                COUNT(*) as available_drivers,
                AVG(distance_to_location_km) as avg_driver_distance
            FROM `{self.config.project_id}.ride_intelligence.driver_locations`
            WHERE ST_DWITHIN(
                ST_GEOGPOINT(longitude, latitude),
                (SELECT ST_GEOGPOINT(longitude, latitude) 
                 FROM `{self.config.project_id}.ride_intelligence.locations` 
                 WHERE location_id = '{location_id}'),
                5000  -- 5km radius
            )
            AND status = 'available'
            AND last_update >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 10 MINUTE)
        ),
        
        network_metrics AS (
            SELECT 
                AVG(ride_count) as network_avg_demand,
                PERCENTILE_CONT(surge_multiplier, 0.5) OVER() as network_median_surge
            FROM (
                SELECT 
                    location_id,
                    COUNT(*) as ride_count,
                    AVG(surge_multiplier) as surge_multiplier
                FROM `{self.config.project_id}.ride_intelligence.historical_rides`
                WHERE ride_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
                GROUP BY location_id
            )
        )
        
        SELECT 
            rr.ride_count_last_hour,
            rr.avg_wait_time,
            rr.current_surge,
            rr.surge_volatility,
            ds.available_drivers,
            ds.avg_driver_distance,
            nm.network_avg_demand,
            nm.network_median_surge,
            
            -- Calculate supply-demand ratio
            CASE 
                WHEN rr.ride_count_last_hour > 0 
                THEN ds.available_drivers / rr.ride_count_last_hour
                ELSE ds.available_drivers
            END as supply_demand_ratio
            
        FROM recent_rides rr
        CROSS JOIN driver_supply ds
        CROSS JOIN network_metrics nm
        """
        
        results = await asyncio.get_event_loop().run_in_executor(
            None, 
            self.bigquery_client.execute_query, 
            conditions_query
        )
        
        if not results:
            return {
                'ride_count_last_hour': 0,
                'avg_wait_time': 5.0,
                'current_surge': 1.0,
                'surge_volatility': 0.1,
                'available_drivers': 10,
                'avg_driver_distance': 2.0,
                'network_avg_demand': 5.0,
                'network_median_surge': 1.0,
                'supply_demand_ratio': 2.0
            }
        
        return dict(results[0])
    
    async def _get_competitive_analysis(self, location_id: str) -> Dict[str, Any]:
        """Get competitive pricing analysis"""
        
        competitive_query = f"""
        WITH competitor_prices AS (
            SELECT 
                competitor_name,
                competitor_price,
                market_share,
                service_quality_score,
                data_timestamp
            FROM `{self.config.project_id}.ride_intelligence.competitor_data`
            WHERE location_id = '{location_id}'
              AND data_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 MINUTE)
        ),
        
        price_analysis AS (
            SELECT 
                COUNT(DISTINCT competitor_name) as competitor_count,
                AVG(competitor_price) as avg_competitor_price,
                PERCENTILE_CONT(competitor_price, 0.5) OVER() as median_competitor_price,
                MIN(competitor_price) as min_competitor_price,
                MAX(competitor_price) as max_competitor_price,
                STDDEV(competitor_price) as price_std_dev,
                AVG(market_share) as avg_market_share,
                AVG(service_quality_score) as avg_quality_score
            FROM competitor_prices
        )
        
        SELECT 
            pa.*,
            
            -- Calculate competitive pressure
            CASE 
                WHEN pa.competitor_count >= 5 AND pa.price_std_dev < pa.avg_competitor_price * 0.1 
                THEN 0.9
                WHEN pa.competitor_count >= 3 AND pa.price_std_dev < pa.avg_competitor_price * 0.15 
                THEN 0.7
                WHEN pa.competitor_count >= 2 
                THEN 0.5
                ELSE 0.2
            END as competitive_pressure_score
            
        FROM price_analysis pa
        """
        
        results = await asyncio.get_event_loop().run_in_executor(
            None, 
            self.bigquery_client.execute_query, 
            competitive_query
        )
        
        if not results:
            return {
                'competitor_count': 0,
                'avg_competitor_price': self.base_price,
                'median_competitor_price': self.base_price,
                'min_competitor_price': self.base_price,
                'max_competitor_price': self.base_price,
                'price_std_dev': 0.0,
                'competitive_pressure_score': 0.2,
                'avg_market_share': 0.0,
                'avg_quality_score': 0.0
            }
        
        return dict(results[0])
    
    async def _get_visual_intelligence_factors(self, location_id: str) -> Dict[str, float]:
        """Get visual intelligence factors affecting pricing"""
        
        visual_query = f"""
        SELECT 
            crowd_count,
            crowd_density_per_sqm,
            accessibility_score,
            infrastructure_quality,
            safety_score,
            confidence_score,
            
            -- Calculate visual impact multiplier
            CASE 
                WHEN crowd_count > 50 THEN 1.3
                WHEN crowd_count > 20 THEN 1.15
                WHEN crowd_count > 10 THEN 1.05
                ELSE 1.0
            END as crowd_multiplier,
            
            CASE 
                WHEN accessibility_score < 0.3 THEN 1.25  -- Poor accessibility = higher price
                WHEN accessibility_score < 0.6 THEN 1.1
                ELSE 1.0
            END as accessibility_multiplier
            
        FROM `{self.config.project_id}.ride_intelligence.image_analysis_results`
        WHERE location_id = '{location_id}'
          AND analysis_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 15 MINUTE)
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
                'crowd_multiplier': 1.0,
                'accessibility_multiplier': 1.0,
                'safety_factor': 1.0,
                'infrastructure_factor': 1.0,
                'visual_confidence': 0.0
            }
        
        row = results[0]
        return {
            'crowd_multiplier': float(row.get('crowd_multiplier', 1.0)),
            'accessibility_multiplier': float(row.get('accessibility_multiplier', 1.0)),
            'safety_factor': max(0.8, float(row.get('safety_score', 0.5))),
            'infrastructure_factor': max(0.9, float(row.get('infrastructure_quality', 0.5))),
            'visual_confidence': float(row.get('confidence_score', 0.0))
        }
    
    async def _calculate_pricing_factors(
        self, 
        location_id: str, 
        demand_forecast: Any, 
        market_conditions: Dict[str, Any], 
        visual_factors: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate all pricing factors"""
        
        # Demand factor
        predicted_demand = demand_forecast.predicted_demand
        demand_factor = min(max(predicted_demand / 10.0, 0.5), 2.0)  # Normalize to 0.5-2.0 range
        
        # Supply-demand ratio factor
        supply_demand_ratio = market_conditions.get('supply_demand_ratio', 1.0)
        supply_factor = max(0.8, min(2.0, 2.0 / max(supply_demand_ratio, 0.1)))
        
        # Wait time factor
        avg_wait_time = market_conditions.get('avg_wait_time', 5.0)
        wait_time_factor = 1.0 + (max(0, avg_wait_time - 5.0) * 0.02)  # +2% per minute over 5 min
        
        # Network surge factor
        network_median_surge = market_conditions.get('network_median_surge', 1.0)
        network_factor = max(0.9, min(1.5, network_median_surge))
        
        # Time-based factor
        current_hour = datetime.now().hour
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:  # Rush hours
            time_factor = 1.2
        elif 22 <= current_hour or current_hour <= 5:  # Late night/early morning
            time_factor = 1.15
        else:
            time_factor = 1.0
        
        # Weather factor (simplified - would integrate with weather API)
        weather_factor = 1.0  # Placeholder
        
        return {
            'demand_factor': demand_factor,
            'supply_factor': supply_factor,
            'wait_time_factor': wait_time_factor,
            'network_factor': network_factor,
            'time_factor': time_factor,
            'weather_factor': weather_factor,
            'crowd_multiplier': visual_factors.get('crowd_multiplier', 1.0),
            'accessibility_multiplier': visual_factors.get('accessibility_multiplier', 1.0),
            'safety_factor': visual_factors.get('safety_factor', 1.0),
            'infrastructure_factor': visual_factors.get('infrastructure_factor', 1.0),
            'visual_confidence': visual_factors.get('visual_confidence', 0.0)
        }
    
    def _optimize_price(
        self, 
        pricing_factors: Dict[str, float], 
        competitive_analysis: Dict[str, Any], 
        strategy: PricingStrategy
    ) -> Dict[str, Any]:
        """Multi-objective price optimization"""
        
        # Calculate base surge multiplier from factors
        base_surge = (
            pricing_factors['demand_factor'] * 0.25 +
            pricing_factors['supply_factor'] * 0.20 +
            pricing_factors['wait_time_factor'] * 0.15 +
            pricing_factors['network_factor'] * 0.10 +
            pricing_factors['time_factor'] * 0.10 +
            pricing_factors['weather_factor'] * 0.05 +
            pricing_factors['crowd_multiplier'] * 0.10 +
            pricing_factors['accessibility_multiplier'] * 0.05
        )
        
        # Apply strategy-specific adjustments
        strategy_weights = self.strategy_weights[strategy]
        
        # Revenue optimization
        if strategy_weights.get('revenue', 0) > 0.5:
            base_surge *= 1.1  # Increase for revenue maximization
        
        # Utilization optimization
        if strategy_weights.get('utilization', 0) > 0.5:
            base_surge *= 0.95  # Slight decrease to encourage more rides
        
        # Customer satisfaction optimization
        if strategy_weights.get('satisfaction', 0) > 0.5:
            base_surge *= 0.9  # Lower prices for satisfaction
        
        # Competitive adjustment
        if strategy == PricingStrategy.COMPETITIVE and competitive_analysis.get('competitor_count', 0) > 0:
            avg_competitor_price = competitive_analysis.get('avg_competitor_price', self.base_price)
            competitive_surge = avg_competitor_price / self.base_price * 0.98  # Slightly undercut
            base_surge = (base_surge * 0.7) + (competitive_surge * 0.3)
        
        # Apply bounds
        surge_multiplier = max(self.min_surge_multiplier, min(self.max_surge_multiplier, base_surge))
        final_price = self.base_price * surge_multiplier
        
        # Calculate confidence score
        visual_confidence = pricing_factors.get('visual_confidence', 0.0)
        data_quality = min(1.0, competitive_analysis.get('competitor_count', 0) / 3.0)
        confidence_score = (visual_confidence * 0.4) + (data_quality * 0.3) + 0.3  # Base confidence
        
        # Calculate objective scores
        objectives = {
            'revenue_score': min(1.0, surge_multiplier / 1.5),  # Higher surge = higher revenue
            'utilization_score': max(0.0, 1.0 - (surge_multiplier - 1.0) / 2.0),  # Lower surge = higher utilization
            'satisfaction_score': max(0.0, 1.0 - (surge_multiplier - 1.0) / 1.5),  # Lower surge = higher satisfaction
            'competitive_score': self._calculate_competitive_score(final_price, competitive_analysis)
        }
        
        return {
            'surge_multiplier': surge_multiplier,
            'final_price': final_price,
            'confidence_score': confidence_score,
            'objectives': objectives
        }
    
    def _calculate_competitive_score(self, final_price: float, competitive_analysis: Dict[str, Any]) -> float:
        """Calculate competitive positioning score"""
        if competitive_analysis.get('competitor_count', 0) == 0:
            return 0.5  # Neutral if no competitors
        
        avg_competitor_price = competitive_analysis.get('avg_competitor_price', final_price)
        
        if final_price <= avg_competitor_price * 0.95:
            return 0.9  # Excellent - below market
        elif final_price <= avg_competitor_price * 1.05:
            return 0.7  # Good - competitive
        elif final_price <= avg_competitor_price * 1.15:
            return 0.5  # Fair - slightly above market
        else:
            return 0.2  # Poor - significantly above market
    
    async def _generate_pricing_reasoning(
        self, 
        location_id: str, 
        pricing_factors: Dict[str, float], 
        optimization_result: Dict[str, Any], 
        competitive_analysis: Dict[str, Any]
    ) -> str:
        """Generate AI explanation for pricing decision"""
        
        surge_multiplier = optimization_result['surge_multiplier']
        final_price = optimization_result['final_price']
        
        # Use BigQuery AI to generate explanation
        reasoning_query = f"""
        SELECT AI.GENERATE_TEXT(
            CONCAT(
                'Explain the pricing decision for location {location_id}: ',
                'Base price: ${self.base_price:.2f}, ',
                'Surge multiplier: {surge_multiplier:.2f}x, ',
                'Final price: ${final_price:.2f}. ',
                'Key factors: ',
                'Demand factor: {pricing_factors.get("demand_factor", 1.0):.2f}, ',
                'Supply factor: {pricing_factors.get("supply_factor", 1.0):.2f}, ',
                'Crowd multiplier: {pricing_factors.get("crowd_multiplier", 1.0):.2f}, ',
                'Accessibility: {pricing_factors.get("accessibility_multiplier", 1.0):.2f}, ',
                'Competitors: {competitive_analysis.get("competitor_count", 0)} with avg price ${competitive_analysis.get("avg_competitor_price", 0.0):.2f}. ',
                'Provide a clear, concise explanation in 2-3 sentences.'
            )
        ) as reasoning
        """
        
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.bigquery_client.execute_query, 
                reasoning_query
            )
            
            if results and results[0].get('reasoning'):
                return results[0]['reasoning']
        except Exception as e:
            logger.warning(f"Failed to generate AI reasoning: {e}")
        
        # Fallback to rule-based reasoning
        reasons = []
        
        if surge_multiplier > 1.5:
            reasons.append("high demand and limited driver availability")
        elif surge_multiplier > 1.2:
            reasons.append("increased demand")
        elif surge_multiplier < 0.9:
            reasons.append("competitive pricing to attract more riders")
        
        if pricing_factors.get('crowd_multiplier', 1.0) > 1.1:
            reasons.append("crowded area detected")
        
        if pricing_factors.get('accessibility_multiplier', 1.0) > 1.1:
            reasons.append("challenging pickup location")
        
        if competitive_analysis.get('competitor_count', 0) > 2:
            reasons.append("competitive market conditions")
        
        if not reasons:
            reasons.append("standard market conditions")
        
        return f"Price set to ${final_price:.2f} ({surge_multiplier:.2f}x surge) due to {', '.join(reasons)}."
    
    async def _store_pricing_calculation(self, result: PricingResult) -> None:
        """Store pricing calculation in BigQuery for analytics"""
        try:
            row = {
                'calculation_id': f"{result.location_id}_{int(result.calculation_timestamp.timestamp())}",
                'location_id': result.location_id,
                'calculation_timestamp': result.calculation_timestamp.isoformat(),
                'base_price': result.base_price,
                'surge_multiplier': result.surge_multiplier,
                'final_price': result.final_price,
                'pricing_factors': result.contributing_factors,
                'visual_intelligence_score': result.contributing_factors.get('visual_confidence', 0.0),
                'demand_forecast': {'predicted_demand': result.contributing_factors.get('demand_factor', 0.0)},
                'competitive_analysis': result.competitive_analysis,
                'confidence_score': result.confidence_score,
                'calculation_method': result.strategy_used.value,
                'model_versions': {'price_calculator': result.model_version}
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.bigquery_client.insert_rows,
                'ride_intelligence.pricing_calculations',
                [row]
            )
            
        except Exception as e:
            logger.error(f"Failed to store pricing calculation: {e}")
    
    async def batch_calculate_prices(
        self, 
        location_ids: List[str], 
        strategy: PricingStrategy = PricingStrategy.BALANCED
    ) -> List[PricingResult]:
        """Calculate prices for multiple locations efficiently"""
        
        tasks = [
            self.calculate_optimal_price(location_id, strategy)
            for location_id in location_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to calculate price for {location_ids[i]}: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
