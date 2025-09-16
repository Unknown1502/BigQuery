"""
Price Validator - Validates pricing decisions against business rules and constraints.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from src.shared.utils.logging_utils import get_logger
from src.shared.clients.bigquery_client import BigQueryClient
from src.shared.config.gcp_config import GCPConfig
from src.shared.utils.cache_manager import CacheManager, get_cache_manager


logger = get_logger(__name__)


class ValidationResult(Enum):
    """Validation result types."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    ADJUSTED = "adjusted"


@dataclass
class ValidationIssue:
    """Data class for validation issues."""
    issue_type: str
    severity: str  # "error", "warning", "info"
    message: str
    suggested_action: str
    original_value: Optional[float] = None
    suggested_value: Optional[float] = None


@dataclass
class PriceValidationResult:
    """Data class for price validation result."""
    is_valid: bool
    result_type: ValidationResult
    validated_price: float
    original_price: float
    issues: List[ValidationIssue]
    adjustments_made: List[str]
    validation_timestamp: datetime


class PriceValidator:
    """
    Validates pricing decisions against business rules, regulatory constraints, and market conditions.
    """
    
    def __init__(self):
        """Initialize the price validator."""
        self.bigquery_client = BigQueryClient()
        self.cache_manager = get_cache_manager()
        
        # Validation configuration
        self.min_price_usd = 2.50
        self.max_price_usd = 150.00
        self.max_surge_multiplier = 5.0
        self.min_surge_multiplier = 0.5
        
        # Price change limits
        self.max_price_increase_percent = 0.50  # 50% max increase
        self.max_price_decrease_percent = 0.30  # 30% max decrease
        self.price_change_window_minutes = 15
        
        # Regulatory limits (example values)
        self.regulatory_max_surge = {
            "default": 3.0,
            "airport": 2.5,
            "hospital": 2.0,
            "emergency_zone": 1.5
        }
        
        logger.info("Price validator initialized")
    
    def validate_price(
        self, 
        price: float, 
        location_id: str, 
        timestamp: datetime,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Validate and potentially adjust a price (synchronous version).
        
        Args:
            price: Proposed price
            location_id: Location identifier
            timestamp: Current timestamp
            context: Additional context
            
        Returns:
            Validated (potentially adjusted) price
        """
        try:
            # Basic validation
            if price < self.min_price_usd:
                logger.warning(f"Price {price} below minimum {self.min_price_usd}, adjusting")
                return self.min_price_usd
            
            if price > self.max_price_usd:
                logger.warning(f"Price {price} above maximum {self.max_price_usd}, adjusting")
                return self.max_price_usd
            
            # Location-specific validation (simplified)
            location_type = context.get("location_type") if context else "default"
            if location_type is None:
                location_type = "default"
            max_allowed = self._get_location_max_price(location_type)
            
            if price > max_allowed:
                logger.warning(f"Price {price} exceeds location limit {max_allowed}, adjusting")
                return max_allowed
            
            return price
            
        except Exception as e:
            logger.error(f"Failed to validate price: {str(e)}")
            return min(self.max_price_usd, max(self.min_price_usd, price))
    
    async def validate_price_comprehensive(
        self, 
        price: float, 
        location_id: str, 
        timestamp: datetime,
        context: Optional[Dict[str, Any]] = None
    ) -> PriceValidationResult:
        """
        Perform comprehensive price validation with detailed results.
        
        Args:
            price: Proposed price
            location_id: Location identifier
            timestamp: Current timestamp
            context: Additional context
            
        Returns:
            Detailed validation result
        """
        try:
            original_price = price
            issues = []
            adjustments_made = []
            
            # 1. Basic range validation
            price, range_issues = self._validate_price_range(price)
            issues.extend(range_issues)
            if range_issues:
                adjustments_made.append("price_range_adjustment")
            
            # 2. Surge multiplier validation
            base_price = context.get("base_price", 8.50) if context else 8.50
            surge_multiplier = price / base_price
            validated_surge, surge_issues = await self._validate_surge_multiplier(
                surge_multiplier, location_id, timestamp
            )
            
            if validated_surge != surge_multiplier:
                price = base_price * validated_surge
                adjustments_made.append("surge_multiplier_adjustment")
            
            issues.extend(surge_issues)
            
            # 3. Price change validation
            price, change_issues = await self._validate_price_change(
                price, location_id, timestamp
            )
            issues.extend(change_issues)
            if change_issues:
                adjustments_made.append("price_change_limitation")
            
            # 4. Regulatory validation
            price, regulatory_issues = await self._validate_regulatory_constraints(
                price, location_id, timestamp
            )
            issues.extend(regulatory_issues)
            if regulatory_issues:
                adjustments_made.append("regulatory_compliance")
            
            # 5. Market validation
            price, market_issues = await self._validate_market_conditions(
                price, location_id, timestamp
            )
            issues.extend(market_issues)
            if market_issues:
                adjustments_made.append("market_adjustment")
            
            # Determine validation result
            result_type = self._determine_validation_result(
                original_price, price, issues
            )
            
            is_valid = result_type in [ValidationResult.VALID, ValidationResult.WARNING]
            
            return PriceValidationResult(
                is_valid=is_valid,
                result_type=result_type,
                validated_price=price,
                original_price=original_price,
                issues=issues,
                adjustments_made=adjustments_made,
                validation_timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"Failed to perform comprehensive price validation: {str(e)}")
            return PriceValidationResult(
                is_valid=False,
                result_type=ValidationResult.INVALID,
                validated_price=min(self.max_price_usd, max(self.min_price_usd, price)),
                original_price=price,
                issues=[ValidationIssue(
                    issue_type="validation_error",
                    severity="error",
                    message=f"Validation failed: {str(e)}",
                    suggested_action="Use fallback price"
                )],
                adjustments_made=["error_fallback"],
                validation_timestamp=timestamp
            )
    
    def _validate_price_range(self, price: float) -> Tuple[float, List[ValidationIssue]]:
        """Validate price is within acceptable range."""
        issues = []
        validated_price = price
        
        if price < self.min_price_usd:
            issues.append(ValidationIssue(
                issue_type="price_too_low",
                severity="error",
                message=f"Price {price:.2f} is below minimum {self.min_price_usd:.2f}",
                suggested_action="Increase to minimum price",
                original_value=price,
                suggested_value=self.min_price_usd
            ))
            validated_price = self.min_price_usd
        
        if price > self.max_price_usd:
            issues.append(ValidationIssue(
                issue_type="price_too_high",
                severity="error",
                message=f"Price {price:.2f} exceeds maximum {self.max_price_usd:.2f}",
                suggested_action="Reduce to maximum price",
                original_value=price,
                suggested_value=self.max_price_usd
            ))
            validated_price = self.max_price_usd
        
        return validated_price, issues
    
    async def _validate_surge_multiplier(
        self, 
        surge_multiplier: float, 
        location_id: str, 
        timestamp: datetime
    ) -> Tuple[float, List[ValidationIssue]]:
        """Validate surge multiplier constraints."""
        issues = []
        validated_surge = surge_multiplier
        
        # Basic surge limits
        if surge_multiplier < self.min_surge_multiplier:
            issues.append(ValidationIssue(
                issue_type="surge_too_low",
                severity="warning",
                message=f"Surge multiplier {surge_multiplier:.2f} is below minimum {self.min_surge_multiplier:.2f}",
                suggested_action="Increase to minimum surge",
                original_value=surge_multiplier,
                suggested_value=self.min_surge_multiplier
            ))
            validated_surge = self.min_surge_multiplier
        
        if surge_multiplier > self.max_surge_multiplier:
            issues.append(ValidationIssue(
                issue_type="surge_too_high",
                severity="error",
                message=f"Surge multiplier {surge_multiplier:.2f} exceeds maximum {self.max_surge_multiplier:.2f}",
                suggested_action="Reduce to maximum surge",
                original_value=surge_multiplier,
                suggested_value=self.max_surge_multiplier
            ))
            validated_surge = self.max_surge_multiplier
        
        # Location-specific surge limits
        try:
            location_max_surge = await self._get_location_surge_limit(location_id)
            if validated_surge > location_max_surge:
                issues.append(ValidationIssue(
                    issue_type="location_surge_limit",
                    severity="error",
                    message=f"Surge {validated_surge:.2f} exceeds location limit {location_max_surge:.2f}",
                    suggested_action="Apply location-specific surge limit",
                    original_value=validated_surge,
                    suggested_value=location_max_surge
                ))
                validated_surge = location_max_surge
        except Exception as e:
            logger.error(f"Failed to check location surge limit: {str(e)}")
        
        return validated_surge, issues
    
    async def _validate_price_change(
        self, 
        price: float, 
        location_id: str, 
        timestamp: datetime
    ) -> Tuple[float, List[ValidationIssue]]:
        """Validate price change is not too dramatic."""
        issues = []
        validated_price = price
        
        try:
            # Get recent price
            recent_price = await self._get_recent_price(location_id, timestamp)
            
            if recent_price and recent_price > 0:
                price_change_percent = (price - recent_price) / recent_price
                
                # Check for excessive increase
                if price_change_percent > self.max_price_increase_percent:
                    max_allowed_price = recent_price * (1 + self.max_price_increase_percent)
                    issues.append(ValidationIssue(
                        issue_type="excessive_price_increase",
                        severity="warning",
                        message=f"Price increase of {price_change_percent:.1%} exceeds limit of {self.max_price_increase_percent:.1%}",
                        suggested_action="Limit price increase",
                        original_value=price,
                        suggested_value=max_allowed_price
                    ))
                    validated_price = max_allowed_price
                
                # Check for excessive decrease
                elif price_change_percent < -self.max_price_decrease_percent:
                    max_allowed_price = recent_price * (1 - self.max_price_decrease_percent)
                    issues.append(ValidationIssue(
                        issue_type="excessive_price_decrease",
                        severity="warning",
                        message=f"Price decrease of {abs(price_change_percent):.1%} exceeds limit of {self.max_price_decrease_percent:.1%}",
                        suggested_action="Limit price decrease",
                        original_value=price,
                        suggested_value=max_allowed_price
                    ))
                    validated_price = max_allowed_price
        
        except Exception as e:
            logger.error(f"Failed to validate price change: {str(e)}")
        
        return validated_price, issues
    
    async def _validate_regulatory_constraints(
        self, 
        price: float, 
        location_id: str, 
        timestamp: datetime
    ) -> Tuple[float, List[ValidationIssue]]:
        """Validate regulatory constraints."""
        issues = []
        validated_price = price
        
        try:
            # Get location regulatory info
            regulatory_info = await self._get_regulatory_info(location_id)
            
            if regulatory_info:
                # Check maximum allowed price
                max_allowed = regulatory_info.get("max_price")
                if max_allowed and price > max_allowed:
                    issues.append(ValidationIssue(
                        issue_type="regulatory_price_limit",
                        severity="error",
                        message=f"Price {price:.2f} exceeds regulatory limit {max_allowed:.2f}",
                        suggested_action="Apply regulatory price limit",
                        original_value=price,
                        suggested_value=max_allowed
                    ))
                    validated_price = max_allowed
                
                # Check surge restrictions
                surge_restrictions = regulatory_info.get("surge_restrictions", {})
                if surge_restrictions.get("max_surge"):
                    # This would require base price calculation
                    pass
        
        except Exception as e:
            logger.error(f"Failed to validate regulatory constraints: {str(e)}")
        
        return validated_price, issues
    
    async def _validate_market_conditions(
        self, 
        price: float, 
        location_id: str, 
        timestamp: datetime
    ) -> Tuple[float, List[ValidationIssue]]:
        """Validate against market conditions."""
        issues = []
        validated_price = price
        
        try:
            # Get market data
            market_data = await self._get_market_data(location_id)
            
            if market_data:
                avg_market_price = market_data.get("avg_competitor_price", 0)
                
                if avg_market_price > 0:
                    price_ratio = price / avg_market_price
                    
                    # Check if significantly above market
                    if price_ratio > 2.0:  # More than 2x market average
                        suggested_price = avg_market_price * 1.8  # 80% above market
                        issues.append(ValidationIssue(
                            issue_type="significantly_above_market",
                            severity="warning",
                            message=f"Price {price:.2f} is {price_ratio:.1f}x market average {avg_market_price:.2f}",
                            suggested_action="Consider market positioning",
                            original_value=price,
                            suggested_value=suggested_price
                        ))
                        # Don't auto-adjust for market conditions, just warn
        
        except Exception as e:
            logger.error(f"Failed to validate market conditions: {str(e)}")
        
        return validated_price, issues
    
    def _determine_validation_result(
        self, 
        original_price: float, 
        validated_price: float, 
        issues: List[ValidationIssue]
    ) -> ValidationResult:
        """Determine overall validation result."""
        has_errors = any(issue.severity == "error" for issue in issues)
        has_warnings = any(issue.severity == "warning" for issue in issues)
        price_changed = abs(original_price - validated_price) > 0.01
        
        if has_errors:
            return ValidationResult.INVALID if not price_changed else ValidationResult.ADJUSTED
        elif has_warnings or price_changed:
            return ValidationResult.WARNING if not price_changed else ValidationResult.ADJUSTED
        else:
            return ValidationResult.VALID
    
    def _get_location_max_price(self, location_type: str) -> float:
        """Get maximum allowed price for location type."""
        location_limits = {
            "airport": 100.0,
            "hospital": 50.0,
            "school": 30.0,
            "emergency_zone": 25.0,
            "default": self.max_price_usd
        }
        
        return location_limits.get(location_type, location_limits["default"])
    
    async def _get_location_surge_limit(self, location_id: str) -> float:
        """Get surge limit for specific location."""
        try:
            # Query location information
            query = """
            SELECT location_type, regulatory_zone
            FROM `{project}.{dataset}.location_profiles`
            WHERE location_id = @location_id
            """
            
            results = await self.bigquery_client.execute_query(
                query, {"location_id": location_id}
            )
            
            if results:
                location_type = results[0].get("location_type", "default")
                return self.regulatory_max_surge.get(location_type, self.regulatory_max_surge["default"])
            else:
                return self.regulatory_max_surge["default"]
        
        except Exception as e:
            logger.error(f"Failed to get location surge limit: {str(e)}")
            return self.regulatory_max_surge["default"]
    
    async def _get_recent_price(self, location_id: str, timestamp: datetime) -> Optional[float]:
        """Get most recent price for location."""
        try:
            cutoff_time = timestamp - timedelta(minutes=self.price_change_window_minutes)
            
            query = """
            SELECT final_price
            FROM `{project}.{dataset}.pricing_decisions`
            WHERE location_id = @location_id
              AND timestamp >= @cutoff_time
              AND timestamp < @current_time
            ORDER BY timestamp DESC
            LIMIT 1
            """
            
            results = await self.bigquery_client.execute_query(
                query, {
                    "location_id": location_id,
                    "cutoff_time": cutoff_time.isoformat(),
                    "current_time": timestamp.isoformat()
                }
            )
            
            if results:
                return float(results[0]["final_price"])
            else:
                return None
        
        except Exception as e:
            logger.error(f"Failed to get recent price: {str(e)}")
            return None
    
    async def _get_regulatory_info(self, location_id: str) -> Optional[Dict[str, Any]]:
        """Get regulatory information for location."""
        try:
            query = """
            SELECT 
                max_allowed_price,
                surge_restrictions,
                special_regulations
            FROM `{project}.{dataset}.regulatory_constraints`
            WHERE location_id = @location_id
              OR regulatory_zone = (
                SELECT regulatory_zone 
                FROM `{project}.{dataset}.location_profiles` 
                WHERE location_id = @location_id
              )
            ORDER BY location_id IS NOT NULL DESC
            LIMIT 1
            """
            
            results = await self.bigquery_client.execute_query(
                query, {"location_id": location_id}
            )
            
            if results:
                return {
                    "max_price": results[0].get("max_allowed_price"),
                    "surge_restrictions": results[0].get("surge_restrictions", {}),
                    "special_regulations": results[0].get("special_regulations", {})
                }
            else:
                return None
        
        except Exception as e:
            logger.error(f"Failed to get regulatory info: {str(e)}")
            return None
    
    async def _get_market_data(self, location_id: str) -> Optional[Dict[str, Any]]:
        """Get market data for location."""
        try:
            query = """
            SELECT 
                AVG(final_price) as avg_competitor_price,
                COUNT(*) as competitor_count
            FROM `{project}.{dataset}.competitor_pricing`
            WHERE location_id = @location_id
              AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
              AND is_available = true
            """
            
            results = await self.bigquery_client.execute_query(
                query, {"location_id": location_id}
            )
            
            if results and results[0]["competitor_count"] > 0:
                return {
                    "avg_competitor_price": float(results[0]["avg_competitor_price"]),
                    "competitor_count": results[0]["competitor_count"]
                }
            else:
                return None
        
        except Exception as e:
            logger.error(f"Failed to get market data: {str(e)}")
            return None
    
    async def validate_bulk_prices(
        self, 
        price_requests: List[Dict[str, Any]]
    ) -> List[PriceValidationResult]:
        """Validate multiple prices in bulk."""
        try:
            validation_tasks = []
            
            for request in price_requests:
                task = asyncio.create_task(
                    self.validate_price_comprehensive(
                        price=request["price"],
                        location_id=request["location_id"],
                        timestamp=datetime.fromisoformat(request["timestamp"]),
                        context=request.get("context")
                    )
                )
                validation_tasks.append(task)
            
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Handle any exceptions
            validated_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Bulk validation failed for request {i}: {str(result)}")
                    # Create error result
                    validated_results.append(PriceValidationResult(
                        is_valid=False,
                        result_type=ValidationResult.INVALID,
                        validated_price=price_requests[i]["price"],
                        original_price=price_requests[i]["price"],
                        issues=[ValidationIssue(
                            issue_type="validation_error",
                            severity="error",
                            message=f"Validation failed: {str(result)}",
                            suggested_action="Manual review required"
                        )],
                        adjustments_made=["error_fallback"],
                        validation_timestamp=datetime.now(timezone.utc)
                    ))
                else:
                    validated_results.append(result)
            
            return validated_results
        
        except Exception as e:
            logger.error(f"Failed to validate bulk prices: {str(e)}")
            return []
    
    def is_healthy(self) -> bool:
        """Check if the price validator is healthy."""
        try:
            return (
                self.bigquery_client is not None and
                self.cache_manager is not None and
                self.min_price_usd > 0 and
                self.max_price_usd > self.min_price_usd
            )
        except Exception:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform detailed health check."""
        try:
            # Test BigQuery connection
            bigquery_healthy = await self.bigquery_client.health_check()
            
            # Test cache connection
            cache_healthy = True  # CacheManager doesn't have health_check method
            
            return {
                "status": "healthy" if bigquery_healthy and cache_healthy else "unhealthy",
                "bigquery_connection": bigquery_healthy,
                "cache_connection": cache_healthy,
                "price_range": {
                    "min_price": self.min_price_usd,
                    "max_price": self.max_price_usd
                },
                "surge_limits": {
                    "min_surge": self.min_surge_multiplier,
                    "max_surge": self.max_surge_multiplier
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
