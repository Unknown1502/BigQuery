"""
Global configuration settings for the Dynamic Pricing Intelligence System.
Manages environment-specific configurations and service settings.
"""

import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """BigQuery and database configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    
    project_id: str = Field(default="test-project", description="GCP Project ID")
    dataset_id: str = Field(default="ride_intelligence", description="BigQuery Dataset ID")
    location: str = Field(default="US", description="BigQuery Location")
    
    # Connection settings
    max_connections: int = Field(default=10, description="Maximum BigQuery connections")
    query_timeout_seconds: int = Field(default=300, description="Query timeout in seconds")
    
    # Table settings
    streaming_buffer_size: int = Field(default=1000, description="Streaming buffer size")
    batch_size: int = Field(default=100, description="Batch processing size")
    
    # BigQuery specific attributes
    bigquery_dataset: str = Field(default="ride_intelligence", description="BigQuery dataset name")


class PubSubSettings(BaseSettings):
    """Pub/Sub configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    
    project_id: str = Field(default="test-project", description="GCP Project ID")
    
    # Topic names
    street_events_topic: str = Field(default="street-events", description="Street events topic")
    pricing_events_topic: str = Field(default="pricing-events", description="Pricing events topic")
    image_analysis_topic: str = Field(default="image-analysis", description="Image analysis topic")
    
    # Subscription names
    pricing_subscription: str = Field(default="pricing-subscription", description="Pricing subscription")
    image_processing_subscription: str = Field(default="image-processing-subscription", description="Image processing subscription")
    analytics_subscription: str = Field(default="analytics-subscription", description="Analytics subscription")
    
    # Performance settings
    max_messages: int = Field(default=100, description="Maximum messages per pull")
    ack_deadline_seconds: int = Field(default=60, description="Acknowledgment deadline")
    flow_control_max_messages: int = Field(default=1000, description="Flow control max messages")


class CloudStorageSettings(BaseSettings):
    """Cloud Storage configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    
    project_id: str = Field(default="test-project", description="GCP Project ID")
    
    # Bucket names
    street_imagery_bucket: str = Field(default="street-imagery-bucket", description="Street imagery bucket")
    model_artifacts_bucket: str = Field(default="model-artifacts-bucket", description="Model artifacts bucket")
    data_lake_bucket: str = Field(default="data-lake-bucket", description="Data lake bucket")
    backup_bucket: str = Field(default="backup-bucket", description="Backup bucket")
    
    # Performance settings
    upload_chunk_size: int = Field(default=8388608, description="Upload chunk size (8MB)")
    download_chunk_size: int = Field(default=8388608, description="Download chunk size (8MB)")
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts")


class APISettings(BaseSettings):
    """API configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    secret_key: str = Field(default="test-secret-key", description="API secret key")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=1000, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    cors_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], description="CORS allowed methods")


class PricingEngineSettings(BaseSettings):
    """Pricing engine configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    
    # Base pricing
    base_price: float = Field(default=10.0, description="Base price")
    currency: str = Field(default="USD", description="Currency")
    
    # Surge pricing
    min_surge_multiplier: float = Field(default=0.8, description="Minimum surge multiplier")
    max_surge_multiplier: float = Field(default=3.0, description="Maximum surge multiplier")
    surge_threshold: float = Field(default=0.7, description="Surge activation threshold")
    
    # Optimization
    optimization_window_hours: int = Field(default=24, description="Optimization window in hours")
    price_update_frequency_minutes: int = Field(default=5, description="Price update frequency")
    
    # A/B testing
    ab_test_enabled: bool = Field(default=True, description="Enable A/B testing")
    default_traffic_split: float = Field(default=0.5, description="Default A/B test traffic split")


class ImageProcessingSettings(BaseSettings):
    """Image processing configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    
    # Processing settings
    max_image_size_mb: int = Field(default=10, description="Maximum image size in MB")
    supported_formats: List[str] = Field(default=["jpg", "jpeg", "png"], description="Supported image formats")
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold for detections")
    
    # Model settings
    crowd_detection_model: str = Field(default="yolov8n", description="Crowd detection model")
    scene_classification_model: str = Field(default="resnet50", description="Scene classification model")
    
    # Performance settings
    batch_size: int = Field(default=8, description="Processing batch size")
    max_concurrent_jobs: int = Field(default=4, description="Maximum concurrent processing jobs")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    structured_logging: bool = Field(default=True, description="Enable structured logging")
    
    # Metrics
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    
    # Tracing
    tracing_enabled: bool = Field(default=True, description="Enable distributed tracing")
    tracing_sample_rate: float = Field(default=0.1, description="Tracing sample rate")
    
    # Health checks
    health_check_interval_seconds: int = Field(default=30, description="Health check interval")
    health_check_timeout_seconds: int = Field(default=5, description="Health check timeout")


class CacheSettings(BaseSettings):
    """Cache configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    
    # Redis settings
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    
    # Cache TTL settings
    default_ttl_seconds: int = Field(default=3600, description="Default cache TTL")
    pricing_ttl_seconds: int = Field(default=300, description="Pricing cache TTL")
    location_ttl_seconds: int = Field(default=7200, description="Location cache TTL")
    
    # Performance settings
    max_connections: int = Field(default=20, description="Maximum Redis connections")
    connection_timeout_seconds: int = Field(default=5, description="Connection timeout")


class GCPConfig(BaseSettings):
    """GCP-specific configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    
    project_id: str = Field(default="test-project", description="GCP Project ID")
    region: str = Field(default="us-central1", description="GCP Region")
    
    # Vertex AI endpoints
    crowd_detection_endpoint: str = Field(default="projects/test-project/locations/us-central1/endpoints/crowd-detection", description="Crowd detection endpoint")
    scene_classification_endpoint: str = Field(default="projects/test-project/locations/us-central1/endpoints/scene-classification", description="Scene classification endpoint")
    accessibility_analysis_endpoint: str = Field(default="projects/test-project/locations/us-central1/endpoints/accessibility-analysis", description="Accessibility analysis endpoint")
    
    # Storage buckets
    street_imagery_bucket: str = Field(default="street-imagery-bucket", description="Street imagery bucket")
    
    # BigQuery
    bigquery_dataset: str = Field(default="ride_intelligence", description="BigQuery dataset")


class MLConfig(BaseSettings):
    """Machine Learning configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)
    
    # Model paths
    crowd_detection_model_path: str = Field(default="models/crowd_detection.h5", description="Crowd detection model path")
    density_estimation_model_path: str = Field(default="models/density_estimation.h5", description="Density estimation model path")
    scene_classification_model_path: str = Field(default="models/scene_classification.h5", description="Scene classification model path")
    object_detection_model_path: str = Field(default="models/object_detection.h5", description="Object detection model path")
    
    # Model parameters
    crowd_detection_input_size: tuple = Field(default=(640, 640), description="Crowd detection input size")
    scene_classification_input_size: tuple = Field(default=(224, 224), description="Scene classification input size")
    
    # Confidence thresholds
    crowd_detection_confidence_threshold: float = Field(default=0.5, description="Crowd detection confidence threshold")
    object_detection_confidence_threshold: float = Field(default=0.5, description="Object detection confidence threshold")
    
    # Processing parameters
    density_to_count_ratio: float = Field(default=1.5, description="Density to count conversion ratio")


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    service_name: str = Field(default="pricing-intelligence", description="Service name")
    service_version: str = Field(default="1.0.0", description="Service version")
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    pubsub: PubSubSettings = Field(default_factory=PubSubSettings)
    cloud_storage: CloudStorageSettings = Field(default_factory=CloudStorageSettings)
    api: APISettings = Field(default_factory=APISettings)
    pricing_engine: PricingEngineSettings = Field(default_factory=PricingEngineSettings)
    image_processing: ImageProcessingSettings = Field(default_factory=ImageProcessingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    gcp: GCPConfig = Field(default_factory=GCPConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment.lower() in ["test", "testing"]
    
    def get_bigquery_table_id(self, table_name: str) -> str:
        """Get fully qualified BigQuery table ID."""
        return f"{self.database.project_id}.{self.database.dataset_id}.{table_name}"
    
    def get_pubsub_topic_path(self, topic_name: str) -> str:
        """Get fully qualified Pub/Sub topic path."""
        return f"projects/{self.pubsub.project_id}/topics/{topic_name}"
    
    def get_pubsub_subscription_path(self, subscription_name: str) -> str:
        """Get fully qualified Pub/Sub subscription path."""
        return f"projects/{self.pubsub.project_id}/subscriptions/{subscription_name}"
    
    def get_gcs_bucket_path(self, bucket_name: str, object_path: str = "") -> str:
        """Get fully qualified Cloud Storage path."""
        if object_path:
            return f"gs://{bucket_name}/{object_path}"
        return f"gs://{bucket_name}"


# Global settings instance
settings = Settings()


# Environment-specific configuration overrides
def get_environment_settings() -> Dict:
    """Get environment-specific configuration overrides."""
    
    if settings.is_production:
        return {
            "log_level": "INFO",
            "debug": False,
            "metrics_enabled": True,
            "tracing_enabled": True,
            "tracing_sample_rate": 0.01,  # Lower sampling in production
        }
    elif settings.is_development:
        return {
            "log_level": "DEBUG",
            "debug": True,
            "metrics_enabled": True,
            "tracing_enabled": True,
            "tracing_sample_rate": 1.0,  # Full sampling in development
        }
    elif settings.is_testing:
        return {
            "log_level": "WARNING",
            "debug": False,
            "metrics_enabled": False,
            "tracing_enabled": False,
        }
    else:
        return {}


def validate_settings() -> None:
    """Validate critical settings and raise errors if misconfigured."""
    
    # Validate required GCP settings
    if not settings.database.project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is required")
    
    if not settings.api.secret_key:
        raise ValueError("API_SECRET_KEY environment variable is required")
    
    # Validate pricing settings
    if settings.pricing_engine.min_surge_multiplier >= settings.pricing_engine.max_surge_multiplier:
        raise ValueError("PRICING_MIN_SURGE_MULTIPLIER must be less than PRICING_MAX_SURGE_MULTIPLIER")
    
    # Validate image processing settings
    if settings.image_processing.confidence_threshold < 0.0 or settings.image_processing.confidence_threshold > 1.0:
        raise ValueError("IMAGE_PROCESSING_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
    
    # Validate monitoring settings
    if settings.monitoring.tracing_sample_rate < 0.0 or settings.monitoring.tracing_sample_rate > 1.0:
        raise ValueError("TRACING_SAMPLE_RATE must be between 0.0 and 1.0")


# Utility functions for backward compatibility
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def setup_logging():
    """Setup logging configuration."""
    import logging
    
    # Configure logging level
    log_level = getattr(logging, settings.monitoring.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def handle_exceptions(func):
    """Decorator for exception handling."""
    from functools import wraps
    from ..utils.error_handling import PricingIntelligenceError, ErrorCategory, ErrorSeverity
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, PricingIntelligenceError):
                raise
            
            # Convert to application error
            raise PricingIntelligenceError(
                message=f"Function '{func.__name__}' failed: {str(e)}",
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.MEDIUM,
                cause=e
            )
    
    return wrapper


# Validate settings on import
try:
    validate_settings()
except ValueError as e:
    # In testing or when environment variables are not set, use defaults
    if not settings.is_testing:
        import warnings
        warnings.warn(f"Settings validation warning: {e}")
