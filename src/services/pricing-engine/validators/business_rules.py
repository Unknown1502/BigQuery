"""
Business Rules Engine for Pricing Validation
Enforces business constraints and regulatory compliance for pricing decisions.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import logging

from src.shared.utils.logging_utils import get_logger
from src.shared.clients.bigquery_client import BigQueryClient
from src.shared.config.gcp_config import GCPConfig
from src.shared.models.pricing_calculation import PricingCalculation
from src.shared.utils.cache_manager import CacheManager

logger = get_logger(__name__)


class RuleViolationType(Enum):
    """Types of business rule violations."""
    PRICE_CEILING = "price_ceiling"
    PRICE_FLOOR = "price_floor"
    SURGE_LIMIT = "surge_limit"
    REGULATORY = "regulatory"
    FAIRNESS = "fairness"
    MARKET_ABUSE = "market_abuse"


@dataclass
class BusinessRule:
    """Business rule definition."""
    rule_id: str
    name: str
    description: str
    rule_type: RuleViolationType
    is_active: bool
    priority: int
    parameters: Dict[str, Any]


@dataclass
class RuleViolation:
    """Business rule violation details."""
    rule_id: str
    violation_type: RuleViolationType
    severity: str  # 'warning', 'error', 'critical'
    message: str
    suggested_action: str
    current_value: Any
    allowed_range: Optional[Tuple[Any, Any]]


class BusinessRulesEngine:
    """
    Business rules engine for pricing validation and compliance.
    """
    
    def __init__(self):
        """Initialize the business rules engine."""
        self.bigquery_client = BigQueryClient()
        self.cache_manager = get_cache_manager()
        self.config = GCPConfig()
        self._rules_cache = {}
        self._last_rules_update = None
        
    async def validate_pricing(
        self,
        pricing_calculation: PricingCalculation,
        location_context: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> Tuple[bool, List[RuleViolation]]:
        """
        Validate pricing calculation against business rules.
        
        Args:
            pricing_calculation: The pricing calculation to validate
            location_context: Location-specific context
            market_context: Market conditions context
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        try:
            # Load active business rules
            rules = await self._get_active_rules()
            violations = []
            
            # Check each rule
            for rule in rules:
                violation = await self._check_rule(
                    rule, pricing_calculation, location_context, market_context
                )
                if violation:
                    violations.append(violation)
            
            # Determine if pricing is valid (no critical violations)
            critical_violations = [v for v in violations if v.severity == 'critical']
            is_valid = len(critical_violations) == 0
            
            # Log validation results
            if violations:
                logger.warning(
                    f"Pricing validation found {len(violations)} violations",
                    extra={
                        'location_id': location_context.get('location_id'),
                        'violations': [v.rule_id for v in violations]
                    }
                )
            
            return is_valid, violations
            
        except Exception as e:
            logger.error(f"Error in pricing validation: {str(e)}")
            # Fail safe - reject pricing on validation error
            return False, [RuleViolation(
                rule_id="system_error",
                violation_type=RuleViolationType.REGULATORY,
                severity="critical",
                message=f"Validation system error: {str(e)}",
                suggested_action="Use fallback pricing",
                current_value=None,
                allowed_range=None
            )]
    
    async def _get_active_rules(self) -> List[BusinessRule]:
        """Get active business rules from cache or database."""
        cache_key = "business_rules_active"
        
        # Check cache first
        cached_rules = await self.cache_manager.get(cache_key)
        if cached_rules:
            return cached_rules
        
        # Load from BigQuery
        query = """
        SELECT 
            rule_id,
            name,
            description,
            rule_type,
            is_active,
            priority,
            parameters
        FROM `{project}.{dataset}.business_rules`
        WHERE is_active = true
        ORDER BY priority DESC
        """.format(
            project=self.config.project_id,
            dataset=self.config.bigquery_dataset
        )
        
        try:
            results = await self.bigquery_client.query_async(query)
            rules = []
            
            for row in results:
                rule = BusinessRule(
                    rule_id=row['rule_id'],
                    name=row['name'],
                    description=row['description'],
                    rule_type=RuleViolationType(row['rule_type']),
                    is_active=row['is_active'],
                    priority=row['priority'],
                    parameters=row['parameters'] or {}
                )
                rules.append(rule)
            
            # Cache for 5 minutes
            await self.cache_manager.set(cache_key, rules, ttl=300)
            return rules
            
        except Exception as e:
            logger.error(f"Error loading business rules: {str(e)}")
            return self._get_default_rules()
    
    async def _check_rule(
        self,
        rule: BusinessRule,
        pricing: PricingCalculation,
        location_context: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> Optional[RuleViolation]:
        """Check a specific business rule against pricing calculation."""
        try:
            if rule.rule_type == RuleViolationType.PRICE_CEILING:
                return await self._check_price_ceiling(rule, pricing, location_context)
            elif rule.rule_type == RuleViolationType.PRICE_FLOOR:
                return await self._check_price_floor(rule, pricing, location_context)
            elif rule.rule_type == RuleViolationType.SURGE_LIMIT:
                return await self._check_surge_limit(rule, pricing, market_context)
            elif rule.rule_type == RuleViolationType.REGULATORY:
                return await self._check_regulatory_compliance(rule, pricing, location_context)
            elif rule.rule_type == RuleViolationType.FAIRNESS:
                return await self._check_fairness_rules(rule, pricing, location_context)
            elif rule.rule_type == RuleViolationType.MARKET_ABUSE:
                return await self._check_market_abuse(rule, pricing, market_context)
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking rule {rule.rule_id}: {str(e)}")
            return None
    
    async def _check_price_ceiling(
        self,
        rule: BusinessRule,
        pricing: PricingCalculation,
        location_context: Dict[str, Any]
    ) -> Optional[RuleViolation]:
        """Check price ceiling rules."""
        max_price = rule.parameters.get('max_price', float('inf'))
        
        if pricing.final_price > max_price:
            return RuleViolation(
                rule_id=rule.rule_id,
                violation_type=rule.rule_type,
                severity=rule.parameters.get('severity', 'error'),
                message=f"Price {pricing.final_price} exceeds maximum allowed {max_price}",
                suggested_action=f"Cap price at {max_price}",
                current_value=pricing.final_price,
                allowed_range=(0, max_price)
            )
        
        return None
    
    async def _check_price_floor(
        self,
        rule: BusinessRule,
        pricing: PricingCalculation,
        location_context: Dict[str, Any]
    ) -> Optional[RuleViolation]:
        """Check price floor rules."""
        min_price = rule.parameters.get('min_price', 0)
        
        if pricing.final_price < min_price:
            return RuleViolation(
                rule_id=rule.rule_id,
                violation_type=rule.rule_type,
                severity=rule.parameters.get('severity', 'error'),
                message=f"Price {pricing.final_price} below minimum allowed {min_price}",
                suggested_action=f"Set minimum price to {min_price}",
                current_value=pricing.final_price,
                allowed_range=(min_price, float('inf'))
            )
        
        return None
    
    async def _check_surge_limit(
        self,
        rule: BusinessRule,
        pricing: PricingCalculation,
        market_context: Dict[str, Any]
    ) -> Optional[RuleViolation]:
        """Check surge pricing limits."""
        max_surge = rule.parameters.get('max_surge_multiplier', 5.0)
        
        if pricing.surge_multiplier > max_surge:
            return RuleViolation(
                rule_id=rule.rule_id,
                violation_type=rule.rule_type,
                severity=rule.parameters.get('severity', 'error'),
                message=f"Surge multiplier {pricing.surge_multiplier} exceeds limit {max_surge}",
                suggested_action=f"Cap surge at {max_surge}x",
                current_value=pricing.surge_multiplier,
                allowed_range=(1.0, max_surge)
            )
        
        return None
    
    async def _check_regulatory_compliance(
        self,
        rule: BusinessRule,
        pricing: PricingCalculation,
        location_context: Dict[str, Any]
    ) -> Optional[RuleViolation]:
        """Check regulatory compliance rules."""
        # Example: Check local pricing regulations
        jurisdiction = location_context.get('jurisdiction')
        if jurisdiction and jurisdiction in rule.parameters.get('restricted_jurisdictions', []):
            max_allowed = rule.parameters.get('jurisdiction_limits', {}).get(jurisdiction)
            if max_allowed and pricing.final_price > max_allowed:
                return RuleViolation(
                    rule_id=rule.rule_id,
                    violation_type=rule.rule_type,
                    severity='critical',
                    message=f"Price violates {jurisdiction} regulations",
                    suggested_action=f"Apply jurisdiction-specific pricing",
                    current_value=pricing.final_price,
                    allowed_range=(0, max_allowed)
                )
        
        return None
    
    async def _check_fairness_rules(
        self,
        rule: BusinessRule,
        pricing: PricingCalculation,
        location_context: Dict[str, Any]
    ) -> Optional[RuleViolation]:
        """Check pricing fairness rules."""
        # Example: Check for discriminatory pricing
        base_price_variance = rule.parameters.get('max_base_price_variance', 0.2)
        market_base_price = location_context.get('market_average_price', pricing.base_price)
        
        if market_base_price > 0:
            variance = abs(pricing.base_price - market_base_price) / market_base_price
            if variance > base_price_variance:
                return RuleViolation(
                    rule_id=rule.rule_id,
                    violation_type=rule.rule_type,
                    severity='warning',
                    message=f"Base price varies {variance:.1%} from market average",
                    suggested_action="Review pricing strategy for fairness",
                    current_value=variance,
                    allowed_range=(0, base_price_variance)
                )
        
        return None
    
    async def _check_market_abuse(
        self,
        rule: BusinessRule,
        pricing: PricingCalculation,
        market_context: Dict[str, Any]
    ) -> Optional[RuleViolation]:
        """Check for market abuse patterns."""
        # Example: Check for excessive pricing during emergencies
        emergency_mode = market_context.get('emergency_mode', False)
        if emergency_mode:
            max_emergency_surge = rule.parameters.get('max_emergency_surge', 1.5)
            if pricing.surge_multiplier > max_emergency_surge:
                return RuleViolation(
                    rule_id=rule.rule_id,
                    violation_type=rule.rule_type,
                    severity='critical',
                    message="Excessive surge pricing during emergency",
                    suggested_action=f"Limit surge to {max_emergency_surge}x during emergencies",
                    current_value=pricing.surge_multiplier,
                    allowed_range=(1.0, max_emergency_surge)
                )
        
        return None
    
    def _get_default_rules(self) -> List[BusinessRule]:
        """Get default business rules as fallback."""
        return [
            BusinessRule(
                rule_id="default_price_ceiling",
                name="Default Price Ceiling",
                description="Maximum allowed price per ride",
                rule_type=RuleViolationType.PRICE_CEILING,
                is_active=True,
                priority=100,
                parameters={'max_price': 500.0, 'severity': 'critical'}
            ),
            BusinessRule(
                rule_id="default_surge_limit",
                name="Default Surge Limit",
                description="Maximum surge multiplier",
                rule_type=RuleViolationType.SURGE_LIMIT,
                is_active=True,
                priority=90,
                parameters={'max_surge_multiplier': 5.0, 'severity': 'error'}
            )
        ]
    
    async def add_rule(self, rule: BusinessRule) -> bool:
        """Add a new business rule."""
        try:
            query = """
            INSERT INTO `{project}.{dataset}.business_rules`
            (rule_id, name, description, rule_type, is_active, priority, parameters, created_at)
            VALUES (@rule_id, @name, @description, @rule_type, @is_active, @priority, @parameters, @created_at)
            """.format(
                project=self.config.project_id,
                dataset=self.config.bigquery_dataset
            )
            
            job_config = {
                'query_parameters': [
                    {'name': 'rule_id', 'parameterType': {'type': 'STRING'}, 'parameterValue': {'value': rule.rule_id}},
                    {'name': 'name', 'parameterType': {'type': 'STRING'}, 'parameterValue': {'value': rule.name}},
                    {'name': 'description', 'parameterType': {'type': 'STRING'}, 'parameterValue': {'value': rule.description}},
                    {'name': 'rule_type', 'parameterType': {'type': 'STRING'}, 'parameterValue': {'value': rule.rule_type.value}},
                    {'name': 'is_active', 'parameterType': {'type': 'BOOL'}, 'parameterValue': {'value': str(rule.is_active)}},
                    {'name': 'priority', 'parameterType': {'type': 'INT64'}, 'parameterValue': {'value': str(rule.priority)}},
                    {'name': 'parameters', 'parameterType': {'type': 'JSON'}, 'parameterValue': {'value': str(rule.parameters)}},
                    {'name': 'created_at', 'parameterType': {'type': 'TIMESTAMP'}, 'parameterValue': {'value': datetime.now(timezone.utc).isoformat()}}
                ]
            }
            
            await self.bigquery_client.query_async(query, job_config)
            
            # Clear cache to force reload
            await self.cache_manager.delete("business_rules_active")
            
            logger.info(f"Added business rule: {rule.rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding business rule: {str(e)}")
            return False
    
    def is_healthy(self) -> bool:
        """Check if the business rules engine is healthy."""
        try:
            return (
                self.bigquery_client is not None and
                self.cache_manager is not None
            )
        except Exception:
            return False
