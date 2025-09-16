"""
Centralized logging utilities for the Dynamic Pricing Intelligence System.
Provides structured logging with correlation IDs, performance metrics, and cloud integration.
"""

import json
import logging
import sys
import time
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from functools import wraps

import structlog
from google.cloud import logging as cloud_logging

from ..config.settings import settings


# Context variables for request tracking
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
location_id_context: ContextVar[Optional[str]] = ContextVar('location_id', default=None)


class CloudLoggingHandler(logging.Handler):
    """Custom handler for Google Cloud Logging integration."""
    
    def __init__(self):
        super().__init__()
        if not settings.is_testing:
            try:
                self.cloud_client = cloud_logging.Client()
                self.cloud_handler = self.cloud_client.get_default_handler()
            except Exception:
                self.cloud_client = None
                self.cloud_handler = None
        else:
            self.cloud_client = None
            self.cloud_handler = None
    
    def emit(self, record):
        """Emit log record to Cloud Logging."""
        if self.cloud_handler:
            try:
                self.cloud_handler.emit(record)
            except Exception:
                # Fallback to console if cloud logging fails
                pass


def add_context_fields(logger, method_name, event_dict):
    """Add context fields to log events."""
    # Add request context
    request_id = request_id_context.get()
    if request_id:
        event_dict['request_id'] = request_id
    
    user_id = user_id_context.get()
    if user_id:
        event_dict['user_id'] = user_id
    
    location_id = location_id_context.get()
    if location_id:
        event_dict['location_id'] = location_id
    
    # Add service context
    event_dict['service'] = settings.service_name
    event_dict['version'] = settings.service_version
    event_dict['environment'] = settings.environment
    
    return event_dict


def add_timestamp(logger, method_name, event_dict):
    """Add ISO timestamp to log events."""
    event_dict['timestamp'] = datetime.now(timezone.utc).isoformat()
    return event_dict


def configure_logging():
    """Configure structured logging for the application."""
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        add_context_fields,
        add_timestamp,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    
    if settings.monitoring.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.monitoring.log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.monitoring.log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.monitoring.log_level.upper()))
    
    if settings.monitoring.log_format == "json":
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add cloud logging handler in production
    if settings.is_production:
        cloud_handler = CloudLoggingHandler()
        root_logger.addHandler(cloud_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    location_id: Optional[str] = None
) -> str:
    """
    Set request context for logging.
    
    Args:
        request_id: Request identifier (generated if not provided)
        user_id: User identifier
        location_id: Location identifier
        
    Returns:
        Request ID that was set
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    request_id_context.set(request_id)
    
    if user_id:
        user_id_context.set(user_id)
    
    if location_id:
        location_id_context.set(location_id)
    
    return request_id


def clear_request_context():
    """Clear request context variables."""
    request_id_context.set(None)
    user_id_context.set(None)
    location_id_context.set(None)


def log_performance(operation_name: str):
    """
    Decorator to log performance metrics for functions.
    
    Args:
        operation_name: Name of the operation being measured
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            logger.info(
                "Operation started",
                operation=operation_name,
                function=func.__name__
            )
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    "Operation completed successfully",
                    operation=operation_name,
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2)
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    "Operation failed",
                    operation=operation_name,
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    error=str(e),
                    error_type=type(e).__name__
                )
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            logger.info(
                "Operation started",
                operation=operation_name,
                function=func.__name__
            )
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    "Operation completed successfully",
                    operation=operation_name,
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2)
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    "Operation failed",
                    operation=operation_name,
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    error=str(e),
                    error_type=type(e).__name__
                )
                
                raise
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_size_bytes: Optional[int] = None,
    response_size_bytes: Optional[int] = None,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None
):
    """
    Log API request details.
    
    Args:
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        request_size_bytes: Request body size
        response_size_bytes: Response body size
        user_agent: User agent string
        ip_address: Client IP address
    """
    logger = get_logger("api")
    
    log_data = {
        "event_type": "api_request",
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2)
    }
    
    if request_size_bytes is not None:
        log_data["request_size_bytes"] = request_size_bytes
    
    if response_size_bytes is not None:
        log_data["response_size_bytes"] = response_size_bytes
    
    if user_agent:
        log_data["user_agent"] = user_agent
    
    if ip_address:
        log_data["ip_address"] = ip_address
    
    # Log level based on status code
    if status_code >= 500:
        logger.error("API request completed", **log_data)
    elif status_code >= 400:
        logger.warning("API request completed", **log_data)
    else:
        logger.info("API request completed", **log_data)


def log_ml_prediction(
    model_name: str,
    model_version: str,
    input_features: Dict[str, Any],
    prediction_result: Any,
    confidence_score: Optional[float] = None,
    processing_time_ms: Optional[float] = None
):
    """
    Log ML model prediction details.
    
    Args:
        model_name: Name of the ML model
        model_version: Version of the ML model
        input_features: Input features used for prediction
        prediction_result: Model prediction result
        confidence_score: Prediction confidence score
        processing_time_ms: Processing time in milliseconds
    """
    logger = get_logger("ml")
    
    log_data = {
        "event_type": "ml_prediction",
        "model_name": model_name,
        "model_version": model_version,
        "input_feature_count": len(input_features),
        "prediction_result": prediction_result
    }
    
    if confidence_score is not None:
        log_data["confidence_score"] = confidence_score
    
    if processing_time_ms is not None:
        log_data["processing_time_ms"] = round(processing_time_ms, 2)
    
    # Log input features (be careful with sensitive data)
    if settings.is_development:
        log_data["input_features"] = input_features
    
    logger.info("ML prediction completed", **log_data)


def log_pricing_decision(
    location_id: str,
    base_price: float,
    surge_multiplier: float,
    final_price: float,
    factors: Dict[str, Any],
    confidence_score: float,
    processing_time_ms: float
):
    """
    Log pricing decision details.
    
    Args:
        location_id: Location identifier
        base_price: Base price before surge
        surge_multiplier: Surge multiplier applied
        final_price: Final calculated price
        factors: Factors that influenced pricing
        confidence_score: Confidence in pricing decision
        processing_time_ms: Processing time in milliseconds
    """
    logger = get_logger("pricing")
    
    logger.info(
        "Pricing decision made",
        event_type="pricing_decision",
        location_id=location_id,
        base_price=base_price,
        surge_multiplier=surge_multiplier,
        final_price=final_price,
        confidence_score=confidence_score,
        processing_time_ms=round(processing_time_ms, 2),
        factors=factors
    )


def log_image_analysis(
    image_id: str,
    analysis_type: str,
    results: Dict[str, Any],
    processing_time_ms: float,
    model_versions: Dict[str, str]
):
    """
    Log image analysis results.
    
    Args:
        image_id: Image identifier
        analysis_type: Type of analysis performed
        results: Analysis results
        processing_time_ms: Processing time in milliseconds
        model_versions: Versions of models used
    """
    logger = get_logger("image_analysis")
    
    logger.info(
        "Image analysis completed",
        event_type="image_analysis",
        image_id=image_id,
        analysis_type=analysis_type,
        processing_time_ms=round(processing_time_ms, 2),
        model_versions=model_versions,
        results_summary={
            "crowd_count": results.get("crowd_analysis", {}).get("crowd_count"),
            "activities_detected": len(results.get("activity_analysis", [])),
            "accessibility_score": results.get("accessibility_analysis", {}).get("accessibility_score")
        }
    )


def log_business_metric(
    metric_name: str,
    metric_value: Union[int, float],
    metric_unit: str,
    dimensions: Optional[Dict[str, str]] = None
):
    """
    Log business metrics for monitoring and alerting.
    
    Args:
        metric_name: Name of the business metric
        metric_value: Metric value
        metric_unit: Unit of measurement
        dimensions: Additional dimensions for the metric
    """
    logger = get_logger("metrics")
    
    log_data = {
        "event_type": "business_metric",
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_unit": metric_unit
    }
    
    if dimensions:
        log_data["dimensions"] = dimensions
    
    logger.info("Business metric recorded", **log_data)


def setup_logging():
    """
    Setup logging configuration (alias for configure_logging).
    This function is imported by many files.
    """
    configure_logging()


def handle_exceptions(func):
    """
    Decorator to handle exceptions with logging.
    
    Args:
        func: Function to wrap with exception handling
        
    Returns:
        Wrapped function with exception handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = get_logger(func.__module__)
            logger.error(
                "Unhandled exception in function",
                function=func.__name__,
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc()
            )
            raise
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger = get_logger(func.__module__)
            logger.error(
                "Unhandled exception in async function",
                function=func.__name__,
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc()
            )
            raise
    
    # Return appropriate wrapper based on function type
    if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
        return async_wrapper
    else:
        return wrapper


# Initialize logging configuration
configure_logging()
