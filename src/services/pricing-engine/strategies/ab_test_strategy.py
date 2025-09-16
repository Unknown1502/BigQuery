"""
A/B Testing Strategy - Manages pricing experiments and user assignments.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import hashlib
import random

from src.shared.utils.logging_utils import get_logger
from src.shared.clients.bigquery_client import BigQueryClient
from src.shared.config.gcp_config import GCPConfig
from src.shared.utils.cache_utils import CacheManager


logger = get_logger(__name__)


class ExperimentStatus(Enum):
    """Experiment status values."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ExperimentType(Enum):
    """Types of pricing experiments."""
    SURGE_MULTIPLIER = "surge_multiplier"
    BASE_PRICE = "base_price"
    DISCOUNT = "discount"
    PREMIUM = "premium"
    DYNAMIC_ADJUSTMENT = "dynamic_adjustment"


@dataclass
class ExperimentVariant:
    """Data class for experiment variant."""
    variant_id: str
    name: str
    description: str
    traffic_allocation: float  # Percentage of traffic (0.0 to 1.0)
    pricing_adjustment: Dict[str, Any]  # Pricing parameters
    is_control: bool = False


@dataclass
class PricingExperiment:
    """Data class for pricing experiment."""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    status: ExperimentStatus
    variants: List[ExperimentVariant]
    start_date: datetime
    end_date: datetime
    target_locations: List[str]
    target_user_segments: List[str]
    success_metrics: List[str]
    created_by: str
    created_at: datetime


@dataclass
class ExperimentAssignment:
    """Data class for user experiment assignment."""
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime
    location_id: str
    user_segment: Optional[str] = None


class ABTestStrategy:
    """
    Manages A/B testing for pricing strategies and user assignments.
    """
    
    def __init__(self):
        """Initialize the A/B test strategy."""
        self.bigquery_client = BigQueryClient()
        self.cache_manager = CacheManager()
        
        # Assignment configuration
        self.assignment_cache_ttl = 3600  # 1 hour
        self.experiment_cache_ttl = 300   # 5 minutes
        
        # Hash salt for consistent user assignments
        self.assignment_salt = "pricing_experiment_2024"
        
        logger.info("A/B test strategy initialized")
    
    async def assign_user_to_experiment(
        self, 
        user_id: str, 
        location_id: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ExperimentAssignment]:
        """
        Assign a user to an active pricing experiment.
        
        Args:
            user_id: User identifier
            location_id: Location identifier
            user_context: Additional user context
            
        Returns:
            Experiment assignment if applicable, None otherwise
        """
        try:
            # Check cache first
            cache_key = f"experiment_assignment_{user_id}_{location_id}"
            cached_assignment = await self.cache_manager.get(cache_key)
            if cached_assignment:
                return ExperimentAssignment(**cached_assignment)
            
            # Get active experiments for location
            active_experiments = await self._get_active_experiments(location_id)
            
            if not active_experiments:
                return None
            
            # Find applicable experiment
            applicable_experiment = await self._find_applicable_experiment(
                active_experiments, user_id, location_id, user_context
            )
            
            if not applicable_experiment:
                return None
            
            # Assign user to variant
            assignment = await self._assign_user_to_variant(
                user_id, location_id, applicable_experiment, user_context
            )
            
            if assignment:
                # Cache the assignment
                await self.cache_manager.set(
                    cache_key, 
                    assignment.__dict__, 
                    ttl=self.assignment_cache_ttl
                )
                
                # Log assignment to BigQuery
                await self._log_experiment_assignment(assignment)
                
                logger.info(
                    f"User {user_id} assigned to experiment {assignment.experiment_id}, variant {assignment.variant_id}",
                    extra={
                        "user_id": user_id,
                        "location_id": location_id,
                        "experiment_id": assignment.experiment_id,
                        "variant_id": assignment.variant_id
                    }
                )
            
            return assignment
            
        except Exception as e:
            logger.error(f"Failed to assign user to experiment: {str(e)}")
            return None
    
    async def get_experiment_pricing_adjustment(
        self, 
        assignment: ExperimentAssignment
    ) -> Dict[str, Any]:
        """
        Get pricing adjustment for an experiment assignment.
        
        Args:
            assignment: Experiment assignment
            
        Returns:
            Pricing adjustment parameters
        """
        try:
            # Get experiment details
            experiment = await self._get_experiment_by_id(assignment.experiment_id)
            
            if not experiment:
                return {}
            
            # Find the assigned variant
            variant = next(
                (v for v in experiment.variants if v.variant_id == assignment.variant_id),
                None
            )
            
            if not variant:
                return {}
            
            return variant.pricing_adjustment
            
        except Exception as e:
            logger.error(f"Failed to get experiment pricing adjustment: {str(e)}")
            return {}
    
    async def create_experiment(
        self, 
        experiment_data: Dict[str, Any],
        created_by: str
    ) -> str:
        """
        Create a new pricing experiment.
        
        Args:
            experiment_data: Experiment configuration
            created_by: User who created the experiment
            
        Returns:
            Experiment ID
        """
        try:
            experiment_id = f"exp_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"
            
            # Validate experiment data
            self._validate_experiment_data(experiment_data)
            
            # Create experiment object
            experiment = PricingExperiment(
                experiment_id=experiment_id,
                name=experiment_data["name"],
                description=experiment_data["description"],
                experiment_type=ExperimentType(experiment_data["experiment_type"]),
                status=ExperimentStatus.DRAFT,
                variants=[
                    ExperimentVariant(**variant_data) 
                    for variant_data in experiment_data["variants"]
                ],
                start_date=datetime.fromisoformat(experiment_data["start_date"]),
                end_date=datetime.fromisoformat(experiment_data["end_date"]),
                target_locations=experiment_data.get("target_locations", []),
                target_user_segments=experiment_data.get("target_user_segments", []),
                success_metrics=experiment_data.get("success_metrics", []),
                created_by=created_by,
                created_at=datetime.now(timezone.utc)
            )
            
            # Save to BigQuery
            await self._save_experiment(experiment)
            
            logger.info(
                f"Created experiment {experiment_id}",
                extra={
                    "experiment_id": experiment_id,
                    "name": experiment.name,
                    "created_by": created_by
                }
            )
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {str(e)}")
            raise
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an experiment (change status to active).
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update experiment status
            query = """
            UPDATE `{project}.{dataset}.pricing_experiments`
            SET status = 'active',
                updated_at = CURRENT_TIMESTAMP()
            WHERE experiment_id = @experiment_id
              AND status = 'draft'
            """
            
            await self.bigquery_client.execute_query(
                query, {"experiment_id": experiment_id}
            )
            
            # Clear cache
            await self._clear_experiment_cache(experiment_id)
            
            logger.info(f"Started experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment_id}: {str(e)}")
            return False
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """
        Stop an experiment (change status to completed).
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update experiment status
            query = """
            UPDATE `{project}.{dataset}.pricing_experiments`
            SET status = 'completed',
                updated_at = CURRENT_TIMESTAMP()
            WHERE experiment_id = @experiment_id
              AND status = 'active'
            """
            
            await self.bigquery_client.execute_query(
                query, {"experiment_id": experiment_id}
            )
            
            # Clear cache
            await self._clear_experiment_cache(experiment_id)
            
            logger.info(f"Stopped experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop experiment {experiment_id}: {str(e)}")
            return False
    
    async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get results and metrics for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment results and metrics
        """
        try:
            # Get experiment details
            experiment = await self._get_experiment_by_id(experiment_id)
            
            if not experiment:
                return {}
            
            # Get assignment statistics
            assignment_stats = await self._get_assignment_statistics(experiment_id)
            
            # Get performance metrics
            performance_metrics = await self._get_performance_metrics(experiment_id)
            
            # Calculate statistical significance
            significance_results = await self._calculate_statistical_significance(
                experiment_id, performance_metrics
            )
            
            return {
                "experiment_id": experiment_id,
                "experiment_name": experiment.name,
                "status": experiment.status.value,
                "start_date": experiment.start_date.isoformat(),
                "end_date": experiment.end_date.isoformat(),
                "assignment_statistics": assignment_stats,
                "performance_metrics": performance_metrics,
                "statistical_significance": significance_results,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment results: {str(e)}")
            return {}
    
    async def _get_active_experiments(self, location_id: str) -> List[PricingExperiment]:
        """Get active experiments for a location."""
        try:
            cache_key = f"active_experiments_{location_id}"
            cached_experiments = await self.cache_manager.get(cache_key)
            if cached_experiments:
                return [PricingExperiment(**exp) for exp in cached_experiments]
            
            query = """
            SELECT *
            FROM `{project}.{dataset}.pricing_experiments`
            WHERE status = 'active'
              AND start_date <= CURRENT_TIMESTAMP()
              AND end_date >= CURRENT_TIMESTAMP()
              AND (
                ARRAY_LENGTH(target_locations) = 0 
                OR @location_id IN UNNEST(target_locations)
              )
            ORDER BY created_at DESC
            """
            
            results = await self.bigquery_client.execute_query(
                query, {"location_id": location_id}
            )
            
            experiments = []
            for row in results:
                # Parse variants
                variants = [
                    ExperimentVariant(**variant) 
                    for variant in row.get("variants", [])
                ]
                
                experiment = PricingExperiment(
                    experiment_id=row["experiment_id"],
                    name=row["name"],
                    description=row["description"],
                    experiment_type=ExperimentType(row["experiment_type"]),
                    status=ExperimentStatus(row["status"]),
                    variants=variants,
                    start_date=row["start_date"],
                    end_date=row["end_date"],
                    target_locations=row.get("target_locations", []),
                    target_user_segments=row.get("target_user_segments", []),
                    success_metrics=row.get("success_metrics", []),
                    created_by=row["created_by"],
                    created_at=row["created_at"]
                )
                experiments.append(experiment)
            
            # Cache results
            await self.cache_manager.set(
                cache_key,
                [exp.__dict__ for exp in experiments],
                ttl=self.experiment_cache_ttl
            )
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to get active experiments: {str(e)}")
            return []
    
    async def _find_applicable_experiment(
        self,
        experiments: List[PricingExperiment],
        user_id: str,
        location_id: str,
        user_context: Optional[Dict[str, Any]]
    ) -> Optional[PricingExperiment]:
        """Find the most applicable experiment for a user."""
        try:
            user_segment = user_context.get("segment") if user_context else None
            
            for experiment in experiments:
                # Check user segment targeting
                if experiment.target_user_segments:
                    if not user_segment or user_segment not in experiment.target_user_segments:
                        continue
                
                # Check if user is already assigned to this experiment
                existing_assignment = await self._get_existing_assignment(
                    user_id, experiment.experiment_id
                )
                
                if existing_assignment:
                    return experiment
                
                # Check traffic allocation
                if self._should_include_user_in_experiment(user_id, experiment):
                    return experiment
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find applicable experiment: {str(e)}")
            return None
    
    def _should_include_user_in_experiment(
        self, 
        user_id: str, 
        experiment: PricingExperiment
    ) -> bool:
        """Determine if user should be included in experiment based on traffic allocation."""
        try:
            # Calculate total traffic allocation
            total_allocation = sum(variant.traffic_allocation for variant in experiment.variants)
            
            if total_allocation <= 0:
                return False
            
            # Use consistent hashing for user assignment
            hash_input = f"{user_id}_{experiment.experiment_id}_{self.assignment_salt}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            user_bucket = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
            
            return user_bucket < total_allocation
            
        except Exception as e:
            logger.error(f"Failed to determine experiment inclusion: {str(e)}")
            return False
    
    async def _assign_user_to_variant(
        self,
        user_id: str,
        location_id: str,
        experiment: PricingExperiment,
        user_context: Optional[Dict[str, Any]]
    ) -> Optional[ExperimentAssignment]:
        """Assign user to a specific variant within an experiment."""
        try:
            # Use consistent hashing for variant assignment
            hash_input = f"{user_id}_{experiment.experiment_id}_variant_{self.assignment_salt}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            variant_bucket = (hash_value % 10000) / 10000.0
            
            # Find variant based on traffic allocation
            cumulative_allocation = 0.0
            selected_variant = None
            
            for variant in experiment.variants:
                cumulative_allocation += variant.traffic_allocation
                if variant_bucket < cumulative_allocation:
                    selected_variant = variant
                    break
            
            if not selected_variant:
                # Fallback to control variant or first variant
                selected_variant = next(
                    (v for v in experiment.variants if v.is_control),
                    experiment.variants[0] if experiment.variants else None
                )
            
            if not selected_variant:
                return None
            
            return ExperimentAssignment(
                user_id=user_id,
                experiment_id=experiment.experiment_id,
                variant_id=selected_variant.variant_id,
                assigned_at=datetime.now(timezone.utc),
                location_id=location_id,
                user_segment=user_context.get("segment") if user_context else None
            )
            
        except Exception as e:
            logger.error(f"Failed to assign user to variant: {str(e)}")
            return None
    
    async def _get_existing_assignment(
        self, 
        user_id: str, 
        experiment_id: str
    ) -> Optional[ExperimentAssignment]:
        """Get existing assignment for user and experiment."""
        try:
            query = """
            SELECT *
            FROM `{project}.{dataset}.experiment_assignments`
            WHERE user_id = @user_id
              AND experiment_id = @experiment_id
            ORDER BY assigned_at DESC
            LIMIT 1
            """
            
            results = await self.bigquery_client.execute_query(
                query, {
                    "user_id": user_id,
                    "experiment_id": experiment_id
                }
            )
            
            if results:
                row = results[0]
                return ExperimentAssignment(
                    user_id=row["user_id"],
                    experiment_id=row["experiment_id"],
                    variant_id=row["variant_id"],
                    assigned_at=row["assigned_at"],
                    location_id=row["location_id"],
                    user_segment=row.get("user_segment")
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get existing assignment: {str(e)}")
            return None
    
    async def _log_experiment_assignment(self, assignment: ExperimentAssignment) -> None:
        """Log experiment assignment to BigQuery."""
        try:
            row = {
                "user_id": assignment.user_id,
                "experiment_id": assignment.experiment_id,
                "variant_id": assignment.variant_id,
                "assigned_at": assignment.assigned_at.isoformat(),
                "location_id": assignment.location_id,
                "user_segment": assignment.user_segment
            }
            
            await self.bigquery_client.buffered_stream_insert(
                "experiment_assignments", row
            )
            
        except Exception as e:
            logger.error(f"Failed to log experiment assignment: {str(e)}")
    
    async def _get_experiment_by_id(self, experiment_id: str) -> Optional[PricingExperiment]:
        """Get experiment by ID."""
        try:
            query = """
            SELECT *
            FROM `{project}.{dataset}.pricing_experiments`
            WHERE experiment_id = @experiment_id
            """
            
            results = await self.bigquery_client.execute_query(
                query, {"experiment_id": experiment_id}
            )
            
            if results:
                row = results[0]
                variants = [
                    ExperimentVariant(**variant) 
                    for variant in row.get("variants", [])
                ]
                
                return PricingExperiment(
                    experiment_id=row["experiment_id"],
                    name=row["name"],
                    description=row["description"],
                    experiment_type=ExperimentType(row["experiment_type"]),
                    status=ExperimentStatus(row["status"]),
                    variants=variants,
                    start_date=row["start_date"],
                    end_date=row["end_date"],
                    target_locations=row.get("target_locations", []),
                    target_user_segments=row.get("target_user_segments", []),
                    success_metrics=row.get("success_metrics", []),
                    created_by=row["created_by"],
                    created_at=row["created_at"]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get experiment by ID: {str(e)}")
            return None
    
    async def _save_experiment(self, experiment: PricingExperiment) -> None:
        """Save experiment to BigQuery."""
        try:
            row = {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "description": experiment.description,
                "experiment_type": experiment.experiment_type.value,
                "status": experiment.status.value,
                "variants": [variant.__dict__ for variant in experiment.variants],
                "start_date": experiment.start_date.isoformat(),
                "end_date": experiment.end_date.isoformat(),
                "target_locations": experiment.target_locations,
                "target_user_segments": experiment.target_user_segments,
                "success_metrics": experiment.success_metrics,
                "created_by": experiment.created_by,
                "created_at": experiment.created_at.isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            await self.bigquery_client.buffered_stream_insert(
                "pricing_experiments", row
            )
            
        except Exception as e:
            logger.error(f"Failed to save experiment: {str(e)}")
            raise
    
    def _validate_experiment_data(self, experiment_data: Dict[str, Any]) -> None:
        """Validate experiment data."""
        required_fields = ["name", "description", "experiment_type", "variants", "start_date", "end_date"]
        
        for field in required_fields:
            if field not in experiment_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate variants
        variants = experiment_data["variants"]
        if not variants:
            raise ValueError("At least one variant is required")
        
        total_allocation = sum(variant.get("traffic_allocation", 0) for variant in variants)
        if total_allocation > 1.0:
            raise ValueError("Total traffic allocation cannot exceed 100%")
        
        # Validate dates
        start_date = datetime.fromisoformat(experiment_data["start_date"])
        end_date = datetime.fromisoformat(experiment_data["end_date"])
        
        if end_date <= start_date:
            raise ValueError("End date must be after start date")
    
    async def _get_assignment_statistics(self, experiment_id: str) -> Dict[str, Any]:
        """Get assignment statistics for an experiment."""
        try:
            query = """
            SELECT 
                variant_id,
                COUNT(*) as assignment_count,
                COUNT(DISTINCT user_id) as unique_users
            FROM `{project}.{dataset}.experiment_assignments`
            WHERE experiment_id = @experiment_id
            GROUP BY variant_id
            """
            
            results = await self.bigquery_client.execute_query(
                query, {"experiment_id": experiment_id}
            )
            
            stats = {}
            for row in results:
                stats[row["variant_id"]] = {
                    "assignment_count": row["assignment_count"],
                    "unique_users": row["unique_users"]
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get assignment statistics: {str(e)}")
            return {}
    
    async def _get_performance_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """Get performance metrics for an experiment."""
        try:
            # This would be implemented based on specific metrics
            # For now, return placeholder structure
            return {
                "conversion_rate": {},
                "average_price": {},
                "revenue_per_user": {},
                "user_satisfaction": {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            return {}
    
    async def _calculate_statistical_significance(
        self, 
        experiment_id: str, 
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate statistical significance for experiment results."""
        try:
            # Placeholder for statistical significance calculation
            # Would implement proper statistical tests (t-test, chi-square, etc.)
            return {
                "is_significant": False,
                "confidence_level": 0.95,
                "p_value": 0.5,
                "effect_size": 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate statistical significance: {str(e)}")
            return {}
    
    async def _clear_experiment_cache(self, experiment_id: str) -> None:
        """Clear experiment-related cache entries."""
        try:
            # Clear experiment cache
            await self.cache_manager.delete(f"experiment_{experiment_id}")
            
            # Clear active experiments cache (would need to iterate through locations)
            # This is a simplified version
            
        except Exception as e:
            logger.error(f"Failed to clear experiment cache: {str(e)}")
    
    def is_healthy(self) -> bool:
        """Check if the A/B test strategy is healthy."""
        try:
            return (
                self.bigquery_client is not None and
                self.cache_manager is not None and
                len(self.assignment_salt) > 0
            )
        except Exception:
            return False
