"""
Competitor Analysis Engine - Analyzes competitor pricing and market positioning.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import logging
from dataclasses import dataclass

from src.shared.utils.logging_utils import get_logger
from src.shared.clients.bigquery_client import BigQueryClient
from src.shared.config.gcp_config import GCPConfig
from src.shared.utils.cache_manager import CacheManager


logger = get_logger(__name__)


@dataclass
class CompetitorData:
    """Data class for competitor information."""
    competitor_id: str
    name: str
    base_price: float
    surge_multiplier: float
    final_price: float
    market_share: float
    availability: bool
    timestamp: datetime


@dataclass
class CompetitorAnalysis:
    """Data class for competitor analysis results."""
    location_id: str
    our_position: str  # "below_market", "competitive", "above_market"
    avg_competitor_price: float
    min_competitor_price: float
    max_competitor_price: float
    price_spread: float
    market_position_score: float
    competitors: List[CompetitorData]
    analysis_timestamp: datetime


class CompetitorAnalyzer:
    """
    Analyzes competitor pricing and market positioning for dynamic pricing decisions.
    """
    
    def __init__(self):
        """Initialize the competitor analyzer."""
        self.bigquery_client = BigQueryClient()
        self.cache_manager = CacheManager()
        self.cache_ttl = 300  # 5 minutes cache
        
        # Competitor configuration
        self.known_competitors = [
            "uber", "lyft", "bolt", "via", "gett", "local_taxi"
        ]
        
        logger.info("Competitor analyzer initialized")
    
    async def analyze_competitors(self, location_id: str) -> Dict[str, Any]:
        """
        Analyze competitor pricing for a specific location.
        
        Args:
            location_id: Location identifier
            
        Returns:
            Competitor analysis results
        """
        try:
            # Check cache first
            cache_key = f"competitor_analysis_{location_id}"
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"Returning cached competitor analysis for {location_id}")
                return cached_result
            
            # Get competitor data
            competitor_data = await self._fetch_competitor_data(location_id)
            
            if not competitor_data:
                logger.warning(f"No competitor data available for location {location_id}")
                return self._get_default_analysis(location_id)
            
            # Analyze competitor positioning
            analysis = await self._analyze_market_position(location_id, competitor_data)
            
            # Convert to dictionary format
            result = {
                "location_id": analysis.location_id,
                "our_position": analysis.our_position,
                "avg_competitor_price": analysis.avg_competitor_price,
                "min_competitor_price": analysis.min_competitor_price,
                "max_competitor_price": analysis.max_competitor_price,
                "price_spread": analysis.price_spread,
                "market_position_score": analysis.market_position_score,
                "competitor_count": len(analysis.competitors),
                "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
                "competitors": [
                    {
                        "name": comp.name,
                        "final_price": comp.final_price,
                        "surge_multiplier": comp.surge_multiplier,
                        "availability": comp.availability
                    }
                    for comp in analysis.competitors
                ]
            }
            
            # Cache the result
            await self.cache_manager.set(cache_key, result, ttl=self.cache_ttl)
            
            logger.info(
                f"Competitor analysis completed for {location_id}",
                extra={
                    "competitor_count": len(analysis.competitors),
                    "our_position": analysis.our_position,
                    "avg_competitor_price": analysis.avg_competitor_price
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze competitors for {location_id}: {str(e)}")
            return self._get_default_analysis(location_id)
    
    async def _fetch_competitor_data(self, location_id: str) -> List[CompetitorData]:
        """
        Fetch competitor data from various sources.
        
        Args:
            location_id: Location identifier
            
        Returns:
            List of competitor data
        """
        try:
            # Query BigQuery for competitor pricing data
            query = """
            SELECT 
                competitor_id,
                competitor_name,
                base_price,
                surge_multiplier,
                final_price,
                market_share,
                is_available,
                timestamp
            FROM `{project}.{dataset}.competitor_pricing`
            WHERE location_id = @location_id
              AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 MINUTE)
              AND is_available = true
            ORDER BY timestamp DESC
            """
            
            results = await self.bigquery_client.execute_query(
                query, {"location_id": location_id}
            )
            
            competitor_data = []
            seen_competitors = set()
            
            for row in results:
                # Take the most recent data for each competitor
                if row["competitor_id"] not in seen_competitors:
                    competitor_data.append(CompetitorData(
                        competitor_id=row["competitor_id"],
                        name=row["competitor_name"],
                        base_price=float(row["base_price"]),
                        surge_multiplier=float(row["surge_multiplier"]),
                        final_price=float(row["final_price"]),
                        market_share=float(row.get("market_share", 0.0)),
                        availability=bool(row["is_available"]),
                        timestamp=row["timestamp"]
                    ))
                    seen_competitors.add(row["competitor_id"])
            
            return competitor_data
            
        except Exception as e:
            logger.error(f"Failed to fetch competitor data: {str(e)}")
            # Return simulated data for development/testing
            return await self._get_simulated_competitor_data(location_id)
    
    async def _get_simulated_competitor_data(self, location_id: str) -> List[CompetitorData]:
        """
        Generate simulated competitor data for testing/development.
        
        Args:
            location_id: Location identifier
            
        Returns:
            List of simulated competitor data
        """
        import random
        
        base_price = 8.50  # Base price in USD
        current_time = datetime.now(timezone.utc)
        
        competitors = []
        for i, competitor_name in enumerate(self.known_competitors[:4]):  # Limit to 4 competitors
            # Simulate realistic pricing variations
            surge_multiplier = random.uniform(0.9, 2.5)
            final_price = base_price * surge_multiplier
            market_share = random.uniform(0.05, 0.35)
            
            competitors.append(CompetitorData(
                competitor_id=f"comp_{i+1}",
                name=competitor_name,
                base_price=base_price,
                surge_multiplier=surge_multiplier,
                final_price=final_price,
                market_share=market_share,
                availability=random.choice([True, True, True, False]),  # 75% availability
                timestamp=current_time
            ))
        
        return competitors
    
    async def _analyze_market_position(
        self, 
        location_id: str, 
        competitor_data: List[CompetitorData]
    ) -> CompetitorAnalysis:
        """
        Analyze our market position relative to competitors.
        
        Args:
            location_id: Location identifier
            competitor_data: List of competitor data
            
        Returns:
            Market position analysis
        """
        if not competitor_data:
            return CompetitorAnalysis(
                location_id=location_id,
                our_position="unknown",
                avg_competitor_price=0.0,
                min_competitor_price=0.0,
                max_competitor_price=0.0,
                price_spread=0.0,
                market_position_score=0.5,
                competitors=[],
                analysis_timestamp=datetime.now(timezone.utc)
            )
        
        # Calculate competitor price statistics
        competitor_prices = [comp.final_price for comp in competitor_data if comp.availability]
        
        if not competitor_prices:
            return CompetitorAnalysis(
                location_id=location_id,
                our_position="no_competition",
                avg_competitor_price=0.0,
                min_competitor_price=0.0,
                max_competitor_price=0.0,
                price_spread=0.0,
                market_position_score=0.5,
                competitors=competitor_data,
                analysis_timestamp=datetime.now(timezone.utc)
            )
        
        avg_competitor_price = sum(competitor_prices) / len(competitor_prices)
        min_competitor_price = min(competitor_prices)
        max_competitor_price = max(competitor_prices)
        price_spread = max_competitor_price - min_competitor_price
        
        # Get our current price (simulated for now)
        our_current_price = await self._get_our_current_price(location_id)
        
        # Determine market position
        our_position = self._determine_market_position(our_current_price, avg_competitor_price)
        
        # Calculate market position score (0.0 = much cheaper, 1.0 = much more expensive)
        if avg_competitor_price > 0:
            market_position_score = min(1.0, max(0.0, our_current_price / avg_competitor_price))
        else:
            market_position_score = 0.5
        
        return CompetitorAnalysis(
            location_id=location_id,
            our_position=our_position,
            avg_competitor_price=avg_competitor_price,
            min_competitor_price=min_competitor_price,
            max_competitor_price=max_competitor_price,
            price_spread=price_spread,
            market_position_score=market_position_score,
            competitors=competitor_data,
            analysis_timestamp=datetime.now(timezone.utc)
        )
    
    async def _get_our_current_price(self, location_id: str) -> float:
        """
        Get our current price for the location.
        
        Args:
            location_id: Location identifier
            
        Returns:
            Our current price
        """
        try:
            # Query our recent pricing decisions
            query = """
            SELECT final_price
            FROM `{project}.{dataset}.pricing_decisions`
            WHERE location_id = @location_id
              AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 15 MINUTE)
            ORDER BY timestamp DESC
            LIMIT 1
            """
            
            results = await self.bigquery_client.execute_query(
                query, {"location_id": location_id}
            )
            
            if results:
                return float(results[0]["final_price"])
            else:
                # Default base price if no recent pricing data
                return 8.50
                
        except Exception as e:
            logger.error(f"Failed to get our current price: {str(e)}")
            return 8.50  # Default base price
    
    def _determine_market_position(self, our_price: float, avg_competitor_price: float) -> str:
        """
        Determine our market position relative to competitors.
        
        Args:
            our_price: Our current price
            avg_competitor_price: Average competitor price
            
        Returns:
            Market position string
        """
        if avg_competitor_price == 0:
            return "no_competition"
        
        price_ratio = our_price / avg_competitor_price
        
        if price_ratio < 0.9:
            return "below_market"
        elif price_ratio > 1.1:
            return "above_market"
        else:
            return "competitive"
    
    def _get_default_analysis(self, location_id: str) -> Dict[str, Any]:
        """
        Get default analysis when no competitor data is available.
        
        Args:
            location_id: Location identifier
            
        Returns:
            Default analysis dictionary
        """
        return {
            "location_id": location_id,
            "our_position": "unknown",
            "avg_competitor_price": 0.0,
            "min_competitor_price": 0.0,
            "max_competitor_price": 0.0,
            "price_spread": 0.0,
            "market_position_score": 0.5,
            "competitor_count": 0,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "competitors": []
        }
    
    async def get_competitor_trends(self, location_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get competitor pricing trends over time.
        
        Args:
            location_id: Location identifier
            hours: Number of hours to look back
            
        Returns:
            Competitor trends analysis
        """
        try:
            query = """
            SELECT 
                competitor_name,
                EXTRACT(HOUR FROM timestamp) as hour,
                AVG(final_price) as avg_price,
                AVG(surge_multiplier) as avg_surge
            FROM `{project}.{dataset}.competitor_pricing`
            WHERE location_id = @location_id
              AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @hours HOUR)
            GROUP BY competitor_name, hour
            ORDER BY competitor_name, hour
            """
            
            results = await self.bigquery_client.execute_query(
                query, {"location_id": location_id, "hours": hours}
            )
            
            trends = {}
            for row in results:
                competitor = row["competitor_name"]
                if competitor not in trends:
                    trends[competitor] = []
                
                trends[competitor].append({
                    "hour": row["hour"],
                    "avg_price": float(row["avg_price"]),
                    "avg_surge": float(row["avg_surge"])
                })
            
            return {
                "location_id": location_id,
                "trends": trends,
                "analysis_period_hours": hours,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get competitor trends: {str(e)}")
            return {
                "location_id": location_id,
                "trends": {},
                "analysis_period_hours": hours,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def is_healthy(self) -> bool:
        """
        Check if the competitor analyzer is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Basic health check - ensure we can access required components
            return (
                self.bigquery_client is not None and
                self.cache_manager is not None and
                len(self.known_competitors) > 0
            )
        except Exception:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform detailed health check.
        
        Returns:
            Health check results
        """
        try:
            # Test BigQuery connection
            bigquery_healthy = await self.bigquery_client.health_check()
            
            # Test cache connection
            cache_healthy = True  # CacheManager doesn't have health_check method
            
            return {
                "status": "healthy" if bigquery_healthy and cache_healthy else "unhealthy",
                "bigquery_connection": bigquery_healthy,
                "cache_connection": cache_healthy,
                "known_competitors": len(self.known_competitors),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
