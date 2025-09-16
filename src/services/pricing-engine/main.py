"""
Pricing Engine Service - Core dynamic pricing intelligence with multimodal geospatial analysis.
Combines BigQuery AI, visual intelligence, and real-time data for optimal pricing decisions.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager

from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.types import PubsubMessage

from ...shared.config.settings import settings
from ...shared.utils.logging_utils import (
    get_logger, log_performance, log_pricing_decision, set_request_context
)
from ...shared.utils.error_handling import (
    PricingEngineError, safe_execute_async, ErrorHandler
)
from ...shared.clients.bigquery_client import get_bigquery_client
from .engines.demand_predictor import DemandPredictor
from .engines.price_calculator import PriceCalculator
from .engines.competitor_analyzer import CompetitorAnalyzer
from .strategies.surge_strategy import SurgeStrategy
from .strategies.ab_test_strategy import ABTestStrategy
from .validators.price_validator import PriceValidator
from .validators.business_rules import BusinessRulesEngine


logger = get_logger(__name__)


@dataclass
class PricingRequest:
    """Data class for pricing request information."""
    request_id: str
    location_id: str
    timestamp: datetime
    user_context: Dict[str, Any]
    ride_context: Dict[str, Any]
    options: Dict[str, Any]


@dataclass
class PricingResult:
    """Data class for pricing calculation results."""
    base_price: float
    surge_multiplier: float
    final_price: float
    confidence_score: float
    reasoning: str
    visual_factors: List[str]
    demand_factors: Dict[str, Any]
    competitor_analysis: Dict[str, Any]
    experiment_assignment: Optional[Dict[str, Any]]
    processing_time_ms: float


class PricingEngineService:
    """
    Advanced pricing engine that combines multiple intelligence sources for optimal pricing.
    """
    
    def __init__(self):
        """Initialize the pricing engine with all components."""
        self.subscriber = pubsub_v1.SubscriberClient()
        self.publisher = pubsub_v1.PublisherClient()
        self.bigquery_client = get_bigquery_client()
        
        # Initialize core engines
        self.demand_predictor = DemandPredictor()
        self.price_calculator = PriceCalculator()
        self.competitor_analyzer = CompetitorAnalyzer()
        
        # Initialize pricing strategies
        self.surge_strategy = SurgeStrategy()
        self.ab_test_strategy = ABTestStrategy()
        
        # Initialize validators
        self.price_validator = PriceValidator()
        self.business_rules = BusinessRulesEngine()
        
        # Subscription and topic paths
        self.subscription_path = settings.pubsub.get_pubsub_subscription_path(
            settings.pubsub.pricing_subscription
        )
        self.output_topic_path = settings.pubsub.get_pubsub_topic_path(
            settings.pubsub.pricing_events_topic
        )
        
        # Cache for recent pricing decisions
        self._pricing_cache: Dict[str, Tuple[PricingResult, datetime]] = {}
        self._cache_ttl = settings.pricing_engine.cache_ttl_seconds
        
        logger.info(
            "Pricing engine service initialized",
            extra={
                "subscription_path": self.subscription_path,
                "output_topic_path": self.output_topic_path,
                "cache_ttl_seconds": self._cache_ttl
            }
        )
    
    @log_performance("pricing_request_processing")
    async def process_pricing_message(self, message: PubsubMessage) -> None:
        """
        Process a pricing request message from Pub/Sub.
        
        Args:
            message: Pub/Sub message containing pricing request
        """
        try:
            # Parse message data
            message_data = json.loads(message.data.decode('utf-8'))
            
            # Create pricing request
            pricing_request = self._parse_pricing_request(message_data)
            
            # Set request context for logging
            set_request_context(
                request_id=pricing_request.request_id,
                location_id=pricing_request.location_id
            )
            
            logger.info(
                "Processing pricing request",
                extra={
                    "request_id": pricing_request.request_id,
                    "location_id": pricing_request.location_id,
                    "message_id": message.message_id
                }
            )
            
            # Calculate optimal pricing
            pricing_result = await self._calculate_optimal_pricing(pricing_request)
            
            # Validate pricing result
            validated_result = await self._validate_pricing_result(pricing_result, pricing_request)
            
            # Cache the result
            self._cache_pricing_result(pricing_request.location_id, validated_result)
            
            # Publish pricing decision
            await self._publish_pricing_decision(pricing_request, validated_result)
            
            # Stream to BigQuery for analytics
            await self._stream_pricing_to_bigquery(pricing_request, validated_result)
            
            # Log pricing decision
            log_pricing_decision(
                location_id=pricing_request.location_id,
                base_price=validated_result.base_price,
                surge_multiplier=validated_result.surge_multiplier,
                final_price=validated_result.final_price,
                factors={
                    "visual_factors": validated_result.visual_factors,
                    "demand_factors": validated_result.demand_factors
                },
                confidence_score=validated_result.confidence_score,
                processing_time_ms=validated_result.processing_time_ms
            )
            
            # Acknowledge message
            message.ack()
            
            logger.info(
                "Pricing request processed successfully",
                extra={
                    "request_id": pricing_request.request_id,
                    "final_price": validated_result.final_price,
                    "surge_multiplier": validated_result.surge_multiplier
                }
            )
            
        except Exception as e:
            logger.error(
                f"Failed to process pricing message: {str(e)}",
                extra={
                    "message_id": message.message_id,
                    "error": str(e)
                }
            )
            # Nack the message to retry
            message.nack()
    
    async def calculate_price_direct(
        self, 
        location_id: str,
        user_context: Optional[Dict[str, Any]] = None,
        ride_context: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> PricingResult:
        """
        Calculate pricing directly (for API calls).
        
        Args:
            location_id: Location identifier
            user_context: User context information
            ride_context: Ride context information
            options: Pricing options
            
        Returns:
            Pricing calculation result
        """
        # Create pricing request
        pricing_request = PricingRequest(
            request_id=f"direct_{int(time.time() * 1000)}",
            location_id=location_id,
            timestamp=datetime.now(timezone.utc),
            user_context=user_context or {},
            ride_context=ride_context or {},
            options=options or {}
        )
        
        # Set request context
        set_request_context(
            request_id=pricing_request.request_id,
            location_id=location_id,
            user_id=user_context.get("user_id") if user_context else None
        )
        
        # Check cache first
        cached_result = self._get_cached_pricing(location_id)
        if cached_result and not options.get("force_recalculate", False):
            logger.info("Returning cached pricing result", extra={"location_id": location_id})
            return cached_result
        
        # Calculate optimal pricing
        pricing_result = await self._calculate_optimal_pricing(pricing_request)
        
        # Validate pricing result
        validated_result = await self._validate_pricing_result(pricing_result, pricing_request)
        
        # Cache the result
        self._cache_pricing_result(location_id, validated_result)
        
        return validated_result
    
    async def _calculate_optimal_pricing(self, request: PricingRequest) -> PricingResult:
        """
        Calculate optimal pricing using all available intelligence sources.
        
        Args:
            request: Pricing request
            
        Returns:
            Pricing calculation result
        """
        start_time = time.time()
        
        try:
            # Run all analysis components in parallel
            visual_task = asyncio.create_task(
                self._get_visual_intelligence(request.location_id)
            )
            demand_task = asyncio.create_task(
                self._predict_demand(request)
            )
            competitor_task = asyncio.create_task(
                self._analyze_competitors(request.location_id)
            )
            similar_locations_task = asyncio.create_task(
                self._get_similar_locations_pricing(request.location_id)
            )
            
            # Wait for all analyses to complete
            visual_intelligence, demand_prediction, competitor_analysis, similar_pricing = await asyncio.gather(
                visual_task, demand_task, competitor_task, similar_locations_task,
                return_exceptions=True
            )
            
            # Handle any exceptions from parallel tasks
            visual_intelligence = visual_intelligence if not isinstance(visual_intelligence, Exception) else {}
            demand_prediction = demand_prediction if not isinstance(demand_prediction, Exception) else {}
            competitor_analysis = competitor_analysis if not isinstance(competitor_analysis, Exception) else {}
            similar_pricing = similar_pricing if not isinstance(similar_pricing, Exception) else {}
            
            # Calculate base price
            base_price = settings.pricing_engine.base_price_usd
            
            # Calculate surge multiplier using multiple strategies
            surge_multiplier = await self._calculate_surge_multiplier(
                request, visual_intelligence, demand_prediction, competitor_analysis, similar_pricing
            )
            
            # Apply business rules and constraints
            surge_multiplier = self.business_rules.apply_surge_constraints(
                surge_multiplier, request.location_id, request.timestamp
            )
            
            # Calculate final price
            final_price = base_price * surge_multiplier
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                visual_intelligence, demand_prediction, competitor_analysis
            )
            
            # Generate reasoning
            reasoning = self._generate_pricing_reasoning(
                surge_multiplier, visual_intelligence, demand_prediction, competitor_analysis
            )
            
            # Extract visual factors
            visual_factors = self._extract_visual_factors(visual_intelligence)
            
            # Check for A/B testing assignment
            experiment_assignment = await self._assign_experiment(request)
            if experiment_assignment:
                surge_multiplier = experiment_assignment.get("test_multiplier", surge_multiplier)
                final_price = base_price * surge_multiplier
            
            processing_time = (time.time() - start_time) * 1000
            
            return PricingResult(
                base_price=base_price,
                surge_multiplier=surge_multiplier,
                final_price=final_price,
                confidence_score=confidence_score,
                reasoning=reasoning,
                visual_factors=visual_factors,
                demand_factors=demand_prediction,
                competitor_analysis=competitor_analysis,
                experiment_assignment=experiment_assignment,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            raise PricingEngineError(
                operation="optimal_pricing_calculation",
                message=f"Failed to calculate optimal pricing: {str(e)}",
                location_id=request.location_id,
                cause=e
            )
    
    async def _get_visual_intelligence(self, location_id: str) -> Dict[str, Any]:
        """Get latest visual intelligence analysis for the location."""
        try:
            # Query BigQuery for latest visual analysis
            query = f"""
            SELECT 
                social_signals,
                timestamp
            FROM `{settings.database.get_bigquery_table_id('realtime_events')}`
            WHERE location_id = @location_id
              AND event_type = 'image_analysis_completed'
              AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
            ORDER BY timestamp DESC
            LIMIT 1
            """
            
            results = await self.bigquery_client.execute_query(
                query, {"location_id": location_id}
            )
            
            if results:
                social_signals = json.loads(results[0]["social_signals"])
                return {
                    "crowd_density_score": social_signals.get("crowd_density_score", 0.0),
                    "accessibility_score": social_signals.get("accessibility_score", 0.5),
                    "event_impact_score": social_signals.get("event_impact_score", 0.0),
                    "traffic_level": social_signals.get("traffic_level", "unknown"),
                    "data_age_minutes": (
                        datetime.now(timezone.utc) - 
                        datetime.fromisoformat(results[0]["timestamp"].replace('Z', '+00:00'))
                    ).total_seconds() / 60
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get visual intelligence: {str(e)}")
            return {}
    
    async def _predict_demand(self, request: PricingRequest) -> Dict[str, Any]:
        """Predict demand using BigQuery AI and local models."""
        try:
            # Use BigQuery AI for demand prediction
            demand_result = await self.bigquery_client.predict_demand_with_confidence(
                request.location_id,
                prediction_horizon_hours=1
            )
            
            # Enhance with local demand predictor
            local_prediction = await self.demand_predictor.predict_demand(
                request.location_id,
                request.timestamp,
                request.user_context,
                request.ride_context
            )
            
            # Combine predictions
            return {
                "predicted_demand": demand_result.get("predicted_demand", local_prediction.get("predicted_demand", 0)),
                "confidence_interval_lower": demand_result.get("confidence_interval_lower", 0),
                "confidence_interval_upper": demand_result.get("confidence_interval_upper", 0),
                "uncertainty_score": demand_result.get("uncertainty_score", 0.5),
                "local_factors": local_prediction.get("factors", {}),
                "model_version": "bigquery_ai_v2.1"
            }
            
        except Exception as e:
            logger.error(f"Failed to predict demand: {str(e)}")
            # Fallback to local prediction only
            try:
                return await self.demand_predictor.predict_demand(
                    request.location_id,
                    request.timestamp,
                    request.user_context,
                    request.ride_context
                )
            except Exception as fallback_error:
                logger.error(f"Fallback demand prediction failed: {str(fallback_error)}")
                return {"predicted_demand": 5.0, "uncertainty_score": 0.8}
    
    async def _analyze_competitors(self, location_id: str) -> Dict[str, Any]:
        """Analyze competitor pricing for the location."""
        try:
            return await self.competitor_analyzer.analyze_competitors(location_id)
        except Exception as e:
            logger.error(f"Failed to analyze competitors: {str(e)}")
            return {}
    
    async def _get_similar_locations_pricing(self, location_id: str) -> Dict[str, Any]:
        """Get pricing patterns from similar locations."""
        try:
            similar_locations = await self.bigquery_client.find_similar_locations(
                location_id, similarity_threshold=0.3
            )
            
            if similar_locations:
                avg_multiplier = sum(loc.get("pricing_pattern", {}).get("avg_surge_multiplier", 1.0) 
                                   for loc in similar_locations) / len(similar_locations)
                
                return {
                    "similar_locations_count": len(similar_locations),
                    "avg_similar_multiplier": avg_multiplier,
                    "similarity_confidence": sum(loc.get("similarity", 0) for loc in similar_locations) / len(similar_locations)
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get similar locations pricing: {str(e)}")
            return {}
    
    async def _calculate_surge_multiplier(
        self,
        request: PricingRequest,
        visual_intelligence: Dict[str, Any],
        demand_prediction: Dict[str, Any],
        competitor_analysis: Dict[str, Any],
        similar_pricing: Dict[str, Any]
    ) -> float:
        """Calculate surge multiplier using multiple factors."""
        try:
            # Start with base multiplier
            base_multiplier = 1.0
            
            # Visual intelligence factors
            crowd_factor = self._calculate_crowd_factor(visual_intelligence)
            accessibility_factor = self._calculate_accessibility_factor(visual_intelligence)
            event_factor = self._calculate_event_factor(visual_intelligence)
            
            # Demand factors
            demand_factor = self._calculate_demand_factor(demand_prediction)
            
            # Competitive factors
            competitive_factor = self._calculate_competitive_factor(competitor_analysis)
            
            # Similar location factors
            similarity_factor = self._calculate_similarity_factor(similar_pricing)
            
            # Temporal factors
            temporal_factor = self._calculate_temporal_factor(request.timestamp)
            
            # Combine all factors with weights
            surge_multiplier = base_multiplier * (
                crowd_factor * settings.pricing_engine.crowd_impact_weight +
                accessibility_factor * settings.pricing_engine.accessibility_impact_weight +
                event_factor * 0.1 +  # Event impact weight
                demand_factor * 0.3 +  # Demand impact weight
                competitive_factor * 0.1 +  # Competitive impact weight
                similarity_factor * settings.pricing_engine.semantic_similarity_weight +
                temporal_factor * 0.05  # Temporal impact weight
            )
            
            # Apply surge strategy
            surge_multiplier = self.surge_strategy.apply_surge_logic(
                surge_multiplier, request.location_id, request.timestamp
            )
            
            # Ensure within bounds
            surge_multiplier = max(
                settings.pricing_engine.min_surge_multiplier,
                min(settings.pricing_engine.max_surge_multiplier, surge_multiplier)
            )
            
            return surge_multiplier
            
        except Exception as e:
            logger.error(f"Failed to calculate surge multiplier: {str(e)}")
            return 1.0  # Default to no surge
    
    def _calculate_crowd_factor(self, visual_intelligence: Dict[str, Any]) -> float:
        """Calculate pricing factor based on crowd density."""
        crowd_density = visual_intelligence.get("crowd_density_score", 0.0)
        
        if crowd_density > 0.8:
            return 1.3  # High crowd density increases demand
        elif crowd_density > 0.5:
            return 1.15
        elif crowd_density > 0.2:
            return 1.05
        else:
            return 1.0
    
    def _calculate_accessibility_factor(self, visual_intelligence: Dict[str, Any]) -> float:
        """Calculate pricing factor based on accessibility."""
        accessibility_score = visual_intelligence.get("accessibility_score", 0.5)
        
        if accessibility_score < 0.3:
            return 1.25  # Poor accessibility increases price
        elif accessibility_score < 0.5:
            return 1.1
        else:
            return 1.0
    
    def _calculate_event_factor(self, visual_intelligence: Dict[str, Any]) -> float:
        """Calculate pricing factor based on event impact."""
        event_impact = visual_intelligence.get("event_impact_score", 0.0)
        
        if event_impact > 0.7:
            return 1.4  # Major event significantly increases demand
        elif event_impact > 0.4:
            return 1.2
        elif event_impact > 0.1:
            return 1.1
        else:
            return 1.0
    
    def _calculate_demand_factor(self, demand_prediction: Dict[str, Any]) -> float:
        """Calculate pricing factor based on predicted demand."""
        predicted_demand = demand_prediction.get("predicted_demand", 5.0)
        uncertainty = demand_prediction.get("uncertainty_score", 0.5)
        
        # Adjust for uncertainty (higher uncertainty = more conservative pricing)
        confidence_adjusted_demand = predicted_demand * (1 - uncertainty * 0.3)
        
        if confidence_adjusted_demand > 15:
            return 1.5  # Very high demand
        elif confidence_adjusted_demand > 10:
            return 1.3
        elif confidence_adjusted_demand > 7:
            return 1.15
        elif confidence_adjusted_demand > 5:
            return 1.0
        else:
            return 0.9  # Low demand, slight discount
    
    def _calculate_competitive_factor(self, competitor_analysis: Dict[str, Any]) -> float:
        """Calculate pricing factor based on competitor analysis."""
        if not competitor_analysis:
            return 1.0
        
        our_position = competitor_analysis.get("our_position", "competitive")
        avg_competitor_price = competitor_analysis.get("avg_competitor_price", 0)
        
        if our_position == "below_market" and avg_competitor_price > 0:
            return 1.1  # We can increase prices
        elif our_position == "above_market":
            return 0.95  # We should be more competitive
        else:
            return 1.0
    
    def _calculate_similarity_factor(self, similar_pricing: Dict[str, Any]) -> float:
        """Calculate pricing factor based on similar locations."""
        if not similar_pricing:
            return 1.0
        
        avg_similar_multiplier = similar_pricing.get("avg_similar_multiplier", 1.0)
        similarity_confidence = similar_pricing.get("similarity_confidence", 0.0)
        
        # Weight the similar location multiplier by confidence
        return 1.0 + (avg_similar_multiplier - 1.0) * similarity_confidence
    
    def _calculate_temporal_factor(self, timestamp: datetime) -> float:
        """Calculate pricing factor based on time of day/week."""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0 = Monday, 6 = Sunday
        
        # Rush hour multipliers
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            return 1.2
        
        # Late night multiplier
        elif 22 <= hour or hour <= 2:
            return 1.15
        
        # Weekend multiplier
        elif day_of_week >= 5:  # Saturday or Sunday
            return 1.1
        
        else:
            return 1.0
    
    def _calculate_confidence_score(
        self,
        visual_intelligence: Dict[str, Any],
        demand_prediction: Dict[str, Any],
        competitor_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score for the pricing decision."""
        confidence_factors = []
        
        # Visual intelligence confidence
        if visual_intelligence:
            data_age = visual_intelligence.get("data_age_minutes", 60)
            visual_confidence = max(0.0, 1.0 - (data_age / 60.0))  # Decay over 1 hour
            confidence_factors.append(visual_confidence)
        
        # Demand prediction confidence
        demand_uncertainty = demand_prediction.get("uncertainty_score", 0.5)
        demand_confidence = 1.0 - demand_uncertainty
        confidence_factors.append(demand_confidence)
        
        # Competitor analysis confidence
        if competitor_analysis:
            competitor_confidence = 0.8  # Assume good confidence if we have data
            confidence_factors.append(competitor_confidence)
        
        # Overall confidence is the average of available factors
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Default moderate confidence
    
    def _generate_pricing_reasoning(
        self,
        surge_multiplier: float,
        visual_intelligence: Dict[str, Any],
        demand_prediction: Dict[str, Any],
        competitor_analysis: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for the pricing decision."""
        reasons = []
        
        # Surge level reasoning
        if surge_multiplier > 1.5:
            reasons.append("High surge due to exceptional demand conditions")
        elif surge_multiplier > 1.2:
            reasons.append("Moderate surge due to increased demand")
        elif surge_multiplier < 0.95:
            reasons.append("Slight discount due to low demand")
        else:
            reasons.append("Standard pricing with minor adjustments")
        
        # Visual factors
        crowd_density = visual_intelligence.get("crowd_density_score", 0.0)
        if crowd_density > 0.7:
            reasons.append(f"High crowd density detected ({crowd_density:.1%})")
        
        accessibility_score = visual_intelligence.get("accessibility_score", 0.5)
        if accessibility_score < 0.4:
            reasons.append(f"Limited accessibility (score: {accessibility_score:.1f})")
        
        event_impact = visual_intelligence.get("event_impact_score", 0.0)
        if event_impact > 0.3:
            reasons.append(f"Event activity detected (impact: {event_impact:.1f})")
        
        # Demand factors
        predicted_demand = demand_prediction.get("predicted_demand", 0)
        if predicted_demand > 10:
            reasons.append(f"High predicted demand ({predicted_demand:.1f} rides/hour)")
        elif predicted_demand < 3:
            reasons.append(f"Low predicted demand ({predicted_demand:.1f} rides/hour)")
        
        # Competitive factors
        our_position = competitor_analysis.get("our_position", "")
        if our_position == "below_market":
            reasons.append("Pricing below market average")
        elif our_position == "above_market":
            reasons.append("Pricing above market average")
        
        return "; ".join(reasons) if reasons else "Standard pricing applied"
    
    def _extract_visual_factors(self, visual_intelligence: Dict[str, Any]) -> List[str]:
        """Extract visual factors that influenced pricing."""
        factors = []
        
        crowd_density = visual_intelligence.get("crowd_density_score", 0.0)
        if crowd_density > 0.5:
            factors.append(f"crowd_density_{crowd_density:.1%}")
        
        accessibility_score = visual_intelligence.get("accessibility_score", 0.5)
        if accessibility_score < 0.5:
            factors.append(f"limited_accessibility_{accessibility_score:.1f}")
        
        event_impact = visual_intelligence.get("event_impact_score", 0.0)
        if event_impact > 0.2:
            factors.append(f"event_activity_{event_impact:.1f}")
        
        traffic_level = visual_intelligence.get("traffic_level", "")
        if traffic_level in ["heavy", "gridlock"]:
            factors.append(f"traffic_{traffic_level}")
        
        return factors
    
    async def _assign_experiment(self, request: PricingRequest) -> Optional[Dict[str, Any]]:
        """Assign user to A/B testing experiment if applicable."""
        try:
            user_id = request.user_context.get("user_id")
            if not user_id:
                return None
            
            # Check if user should be in an experiment
            experiment_assignment = await self.bigquery_client.assign_pricing_experiment(
                request.location_id, user_id
            )
            
            return experiment_assignment
            
        except Exception as e:
            logger.error(f"Failed to assign experiment: {str(e)}")
            return None
    
    async def _validate_pricing_result(
        self, result: PricingResult, request: PricingRequest
    ) -> PricingResult:
        """Validate and potentially adjust pricing result."""
        try:
            # Validate price bounds
            validated_price = self.price_validator.validate_price(
                result.final_price, request.location_id, request.timestamp
            )
            
            # Apply business rules
            adjusted_result = self.business_rules.apply_pricing_rules(
                result, request
            )
            
            # Update final price if adjusted
            if validated_price != result.final_price:
                adjusted_result.final_price = validated_price
                adjusted_result.surge_multiplier = validated_price / adjusted_result.base_price
                adjusted_result.reasoning += f"; Price adjusted for validation (was {result.final_price:.2f})"
            
            return adjusted_result
            
        except Exception as e:
            logger.error(f"Failed to validate pricing result: {str(e)}")
            return result  # Return original if validation fails
    
    def _cache_pricing_result(self, location_id: str, result: PricingResult) -> None:
        """Cache pricing result for future requests."""
        try:
            self._pricing_cache[location_id] = (result, datetime.now(timezone.utc))
            
            # Clean old cache entries
            current_time = datetime.now(timezone.utc)
            expired_keys = [
                key for key, (_, timestamp) in self._pricing_cache.items()
                if (current_time - timestamp).total_seconds() > self._cache_ttl
            ]
            
            for key in expired_keys:
                del self._pricing_cache[key]
                
        except Exception as e:
            logger.error(f"Failed to cache pricing result: {str(e)}")
    
    def _get_cached_pricing(self, location_id: str) -> Optional[PricingResult]:
        """Get cached pricing result if available and not expired."""
        try:
            if location_id in self._pricing_cache:
                result, timestamp = self._pricing_cache[location_id]
                
                # Check if cache is still valid
                age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
                if age_seconds < self._cache_ttl:
                    return result
                else:
                    # Remove expired entry
                    del self._pricing_cache[location_id]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached pricing: {str(e)}")
            return None
    
    def _parse_pricing_request(self, message_data: Dict[str, Any]) -> PricingRequest:
        """Parse pricing request from message data."""
        return PricingRequest(
            request_id=message_data.get("request_id", f"msg_{int(time.time() * 1000)}"),
            location_id=message_data["location_id"],
            timestamp=datetime.fromisoformat(
                message_data.get("timestamp", datetime.now(timezone.utc).isoformat())
            ),
            user_context=message_data.get("user_context", {}),
            ride_context=message_data.get("ride_context", {}),
            options=message_data.get("options", {})
        )
    
    async def _publish_pricing_decision(
        self, request: PricingRequest, result: PricingResult
    ) -> None:
        """Publish pricing decision to Pub/Sub topic."""
        try:
            message_data = {
                "pricing_decision_id": f"{request.request_id}_decision",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request.request_id,
                "location_id": request.location_id,
                "pricing_result": {
                    "base_price": result.base_price,
                    "surge_multiplier": result.surge_multiplier,
                    "final_price": result.final_price,
                    "confidence_score": result.confidence_score,
                    "reasoning": result.reasoning,
                    "visual_factors": result.visual_factors,
                    "processing_time_ms": result.processing_time_ms
                },
                "user_context": request.user_context,
                "ride_context": request.ride_context
            }
            
            # Publish message
            future = self.publisher.publish(
                self.output_topic_path,
                json.dumps(message_data).encode('utf-8')
            )
            
            # Wait for publish to complete
            await asyncio.get_event_loop().run_in_executor(
                None, future.result
            )
            
            logger.info(
                "Pricing decision published successfully",
                extra={
                    "request_id": request.request_id,
                    "final_price": result.final_price
                }
            )
            
        except Exception as e:
            logger.error(
                f"Failed to publish pricing decision: {str(e)}",
                extra={"request_id": request.request_id}
            )
            raise
    
    async def _stream_pricing_to_bigquery(
        self, request: PricingRequest, result: PricingResult
    ) -> None:
        """Stream pricing decision to BigQuery for analytics."""
        try:
            # Prepare row for BigQuery
            row = {
                "timestamp": request.timestamp.isoformat(),
                "location_id": request.location_id,
                "base_price": result.base_price,
                "surge_multiplier": result.surge_multiplier,
                "final_price": result.final_price,
                "confidence_score": result.confidence_score,
                "visual_factors": json.dumps(result.visual_factors),
                "demand_factors": json.dumps(result.demand_factors),
                "competitor_analysis": json.dumps(result.competitor_analysis),
                "processing_time_ms": result.processing_time_ms,
                "reasoning": result.reasoning,
                "user_context": json.dumps(request.user_context),
                "ride_context": json.dumps(request.ride_context)
            }
            
            # Stream to BigQuery
            await self.bigquery_client.buffered_stream_insert("pricing_decisions", row)
            
        except Exception as e:
            logger.error(
                f"Failed to stream pricing to BigQuery: {str(e)}",
                extra={"request_id": request.request_id}
            )
            # Don't raise - this is not critical for the main pipeline
    
    async def start_processing(self) -> None:
        """Start the pricing engine service."""
        logger.info("Starting pricing engine service")
        
        # Configure flow control
        flow_control = pubsub_v1.types.FlowControl(
            max_messages=settings.pubsub.flow_control_max_messages
        )
        
        # Start pulling messages
        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path,
            callback=self.process_pricing_message,
            flow_control=flow_control
        )
        
        logger.info(f"Listening for messages on {self.subscription_path}")
        
        try:
            # Keep the service running
            await asyncio.get_event_loop().run_in_executor(
                None, streaming_pull_future.result
            )
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            logger.info("Pricing engine service stopped")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the service."""
        try:
            # Check component health
            components_healthy = (
                self.demand_predictor.is_healthy() and
                self.price_calculator.is_healthy() and
                self.competitor_analyzer.is_healthy()
            )
            
            # Check BigQuery connection
            bigquery_healthy = await self.bigquery_client.health_check()
            
            # Check cache status
            cache_size = len(self._pricing_cache)
            
            return {
                "status": "healthy" if components_healthy and bigquery_healthy else "unhealthy",
                "components_loaded": components_healthy,
                "bigquery_connection": bigquery_healthy,
                "cache_size": cache_size,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


@asynccontextmanager
async def pricing_engine_service():
    """Context manager for pricing engine service."""
    service = PricingEngineService()
    try:
        yield service
    finally:
        # Cleanup resources if needed
        pass


async def main():
    """Main entry point for the pricing engine service."""
    async with pricing_engine_service() as service:
        await service.start_processing()


if __name__ == "__main__":
    asyncio.run(main())
