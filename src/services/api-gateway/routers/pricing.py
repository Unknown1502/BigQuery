
"""
Pricing API Router
REST API endpoints for dynamic pricing operations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import logging

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.shared.utils.logging_utils import get_logger
from src.shared.models.pricing_models import PricingRequest, PricingResponse
from src.shared.models.api_models import BatchPricingRequest
from src.shared.config.gcp_config import GCPConfig
from src.shared.utils.error_handling import PricingIntelligenceError, ErrorCategory, ErrorSeverity

logger = get_logger(__name__)

# Import pricing engine modules with proper error handling
try:
    # Use importlib to handle the hyphenated directory name
    import importlib.util
    import os
    
    # Get the absolute path to the pricing engine modules
    current_dir = Path(__file__).parent
    pricing_engine_dir = current_dir.parent.parent / "pricing-engine" / "engines"
    
    # Import price_calculator
    price_calc_path = pricing_engine_dir / "price_calculator.py"
    spec = importlib.util.spec_from_file_location("price_calculator", price_calc_path)
    if spec is not None and spec.loader is not None:
        price_calc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(price_calc_module)
        PriceCalculator = price_calc_module.PriceCalculator
        PricingStrategy = price_calc_module.PricingStrategy
    else:
        raise ImportError("Could not load price_calculator module")
    
    # Import demand_predictor
    demand_pred_path = pricing_engine_dir / "demand_predictor.py"
    spec = importlib.util.spec_from_file_location("demand_predictor", demand_pred_path)
    if spec is not None and spec.loader is not None:
        demand_pred_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demand_pred_module)
        DemandPredictor = demand_pred_module.DemandPredictor
    else:
        raise ImportError("Could not load demand_predictor module")
    
    logger.info("Successfully imported pricing engine modules")
    
except Exception as e:
    logger.error(f"Failed to import pricing engine modules: {e}")
    # Fallback - create mock classes to prevent import errors
    from enum import Enum
    
    class PricingStrategy(Enum):
        BALANCED = "balanced"
        REVENUE_MAXIMIZATION = "revenue_maximization"
        UTILIZATION_OPTIMIZATION = "utilization_optimization"
        CUSTOMER_SATISFACTION = "customer_satisfaction"
        COMPETITIVE = "competitive"
    
    class PriceCalculator:
        def __init__(self, config):
            self.config = config
        
        async def calculate_optimal_price(self, **kwargs):
            raise HTTPException(status_code=503, detail="Pricing engine unavailable")
        
        async def batch_calculate_prices(self, **kwargs):
            raise HTTPException(status_code=503, detail="Pricing engine unavailable")
    
    class DemandPredictor:
        def __init__(self, config):
            self.config = config
        
        async def predict_demand(self, **kwargs):
            raise HTTPException(status_code=503, detail="Demand predictor unavailable")
        
        async def get_demand_trends(self, **kwargs):
            raise HTTPException(status_code=503, detail="Demand predictor unavailable")
security = HTTPBearer()

# Initialize router
router = APIRouter(prefix="/pricing", tags=["pricing"])

# Initialize services
config = GCPConfig()
price_calculator = PriceCalculator(config)
demand_predictor = DemandPredictor(config)

@router.post("/calculate", response_model=PricingResponse)
async def calculate_price(
    request: PricingRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """
    Calculate optimal price for a specific location
    """
    try:
        logger.info(f"Calculating price for location {request.location_id}")
        
        # Validate request
        if not request.location_id:
            raise HTTPException(status_code=400, detail="Location ID is required")
        
        # Parse strategy
        strategy = PricingStrategy.BALANCED
        if request.strategy:
            try:
                strategy = PricingStrategy(request.strategy)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")
        
        # Calculate optimal price
        result = await price_calculator.calculate_optimal_price(
            location_id=request.location_id,
            strategy=strategy,
            include_competitive_analysis=request.include_competitive_analysis
        )
        
        # Log pricing decision for audit
        background_tasks.add_task(
            log_pricing_decision,
            request.location_id,
            result.final_price,
            result.surge_multiplier,
            result.reasoning
        )
        
        # Create response
        response = PricingResponse(
            location_id=result.location_id,
            base_price=result.base_price,
            surge_multiplier=result.surge_multiplier,
            final_price=result.final_price,
            confidence_score=result.confidence_score,
            strategy_used=result.strategy_used.value,
            reasoning=result.reasoning,
            calculation_timestamp=result.calculation_timestamp,
            expires_at=result.calculation_timestamp.replace(minute=result.calculation_timestamp.minute + 2),
            contributing_factors=result.contributing_factors,
            competitive_analysis=result.competitive_analysis if request.include_competitive_analysis else None
        )
        
        logger.info(f"Price calculated for {request.location_id}: ${result.final_price:.2f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating price for {request.location_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/batch-calculate", response_model=List[PricingResponse])
async def batch_calculate_prices(
    request: BatchPricingRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(security)
):
    """
    Calculate optimal prices for multiple locations
    """
    try:
        logger.info(f"Batch calculating prices for {len(request.location_ids)} locations")
        
        # Validate request
        if not request.location_ids:
            raise HTTPException(status_code=400, detail="Location IDs are required")
        
        if len(request.location_ids) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 locations per batch request")
        
        # Parse strategy
        strategy = PricingStrategy.BALANCED
        if request.strategy:
            try:
                strategy = PricingStrategy(request.strategy)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")
        
        # Calculate prices for all locations
        results = await price_calculator.batch_calculate_prices(
            location_ids=request.location_ids,
            strategy=strategy
        )
        
        # Convert to response format
        responses = []
        for result in results:
            response = PricingResponse(
                location_id=result.location_id,
                base_price=result.base_price,
                surge_multiplier=result.surge_multiplier,
                final_price=result.final_price,
                confidence_score=result.confidence_score,
                strategy_used=result.strategy_used.value,
                reasoning=result.reasoning,
                calculation_timestamp=result.calculation_timestamp,
                expires_at=result.calculation_timestamp.replace(minute=result.calculation_timestamp.minute + 2),
                contributing_factors=result.contributing_factors,
                competitive_analysis=result.competitive_analysis if request.include_competitive_analysis else None
            )
            responses.append(response)
        
        # Log batch operation
        background_tasks.add_task(
            log_batch_pricing_operation,
            request.location_ids,
            len(results)
        )
        
        logger.info(f"Batch price calculation completed: {len(results)}/{len(request.location_ids)} successful")
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch price calculation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/forecast/{location_id}")
async def get_demand_forecast(
    location_id: str,
    horizon_hours: int = 1,
    token: str = Depends(security)
):
    """
    Get demand forecast for a specific location
    """
    try:
        logger.info(f"Getting demand forecast for location {location_id}")
        
        # Validate parameters
        if horizon_hours < 1 or horizon_hours > 24:
            raise HTTPException(status_code=400, detail="Horizon must be between 1 and 24 hours")
        
        # Get demand prediction
        forecast = await demand_predictor.predict_demand(
            location_id=location_id,
            prediction_horizon_hours=horizon_hours
        )
        
        response = {
            "location_id": forecast.location_id,
            "prediction_timestamp": forecast.prediction_timestamp,
            "horizon_hours": horizon_hours,
            "predicted_demand": forecast.predicted_demand,
            "confidence_interval": {
                "lower": forecast.confidence_interval_lower,
                "upper": forecast.confidence_interval_upper
            },
            "uncertainty_score": forecast.uncertainty_score,
            "contributing_factors": forecast.contributing_factors,
            "model_version": forecast.model_version
        }
        
        logger.info(f"Demand forecast for {location_id}: {forecast.predicted_demand:.2f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting demand forecast for {location_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/trends/{location_id}")
async def get_pricing_trends(
    location_id: str,
    hours_back: int = 24,
    token: str = Depends(security)
):
    """
    Get historical pricing trends for a location
    """
    try:
        logger.info(f"Getting pricing trends for location {location_id}")
        
        # Validate parameters
        if hours_back < 1 or hours_back > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Hours back must be between 1 and 168")
        
        # Get demand trends
        demand_trends = await demand_predictor.get_demand_trends(
            location_id=location_id,
            hours_back=hours_back
        )
        
        response = {
            "location_id": location_id,
            "hours_back": hours_back,
            "trends": demand_trends,
            "summary": {
                "total_data_points": len(demand_trends),
                "avg_demand": sum(t.get('actual_demand', 0) for t in demand_trends) / len(demand_trends) if demand_trends else 0,
                "avg_surge": sum(t.get('avg_surge', 1.0) for t in demand_trends) / len(demand_trends) if demand_trends else 1.0
            }
        }
        
        logger.info(f"Retrieved {len(demand_trends)} trend data points for {location_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pricing trends for {location_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/strategies")
async def get_pricing_strategies(token: str = Depends(security)):
    """
    Get available pricing strategies
    """
    try:
        strategies = [
            {
                "name": strategy.value,
                "display_name": strategy.value.replace("_", " ").title(),
                "description": get_strategy_description(strategy)
            }
            for strategy in PricingStrategy
        ]
        
        return {
            "strategies": strategies,
            "default": PricingStrategy.BALANCED.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pricing strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/analyze-image")
async def analyze_image_for_pricing(
    image_uri: str,
    location_id: str,
    token: str = Depends(security)
):
    """
    Analyze street image for pricing factors
    """
    try:
        logger.info(f"Analyzing image for pricing: {image_uri}")
        
        # This would integrate with the image processing service
        # For now, return a placeholder response
        response = {
            "image_uri": image_uri,
            "location_id": location_id,
            "analysis": {
                "crowd_count": 25,
                "accessibility_score": 0.8,
                "safety_score": 0.9,
                "infrastructure_quality": 0.85,
                "pricing_impact": {
                    "crowd_multiplier": 1.15,
                    "accessibility_multiplier": 1.0,
                    "overall_impact": "moderate_increase"
                }
            },
            "confidence_score": 0.92,
            "processing_time_ms": 1250
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing image for pricing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Background task functions
async def log_pricing_decision(
    location_id: str, 
    final_price: float, 
    surge_multiplier: float, 
    reasoning: str
):
    """Log pricing decision for audit purposes"""
    try:
        # This would log to BigQuery audit table
        logger.info(f"Pricing decision logged: {location_id} -> ${final_price:.2f} ({surge_multiplier:.2f}x)")
    except Exception as e:
        logger.error(f"Error logging pricing decision: {e}")

async def log_batch_pricing_operation(location_ids: List[str], successful_count: int):
    """Log batch pricing operation"""
    try:
        logger.info(f"Batch pricing operation: {successful_count}/{len(location_ids)} successful")
    except Exception as e:
        logger.error(f"Error logging batch operation: {e}")

def get_strategy_description(strategy: PricingStrategy) -> str:
    """Get description for pricing strategy"""
    descriptions = {
        PricingStrategy.REVENUE_MAXIMIZATION: "Optimize for maximum revenue generation",
        PricingStrategy.UTILIZATION_OPTIMIZATION: "Optimize for maximum driver utilization",
        PricingStrategy.CUSTOMER_SATISFACTION: "Optimize for customer satisfaction and retention",
        PricingStrategy.BALANCED: "Balance revenue, utilization, and satisfaction",
        PricingStrategy.COMPETITIVE: "Competitive pricing based on market conditions"
    }
    return descriptions.get(strategy, "Unknown strategy")
