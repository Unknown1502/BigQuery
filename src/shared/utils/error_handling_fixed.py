"""
Centralized error handling utilities for the Dynamic Pricing Intelligence System.
Provides custom exceptions, error classification, and GCP error handling.
"""

import traceback
from typing import Any, Dict, Optional, Type, Union
from enum import Enum
from datetime import datetime, timezone
from functools import wraps

from google.api_core import exceptions as gcp_exceptions
from google.cloud.exceptions import GoogleCloudError

from .logging_utils import get_logger


logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    RESOURCE_NOT_FOUND = "resource_not_found"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    ML_MODEL = "ml_model"
    IMAGE_PROCESSING = "image_processing"
    PRICING_ENGINE = "pricing_engine"
    CONFIGURATION = "configuration"
    INTERNAL = "internal"


class PricingIntelligenceError(Exception):
    """Base exception class for all application-specific errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now(timezone.utc)
        
        # Log the error
        self._log_error()
    
    def _log_error(self):
        """Log the error with appropriate level based on severity."""
        log_data = {
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.cause:
            log_data["cause"] = str(self.cause)
            log_data["cause_type"] = type(self.cause).__name__
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(self.message, extra=log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(self.message, extra=log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(self.message, extra=log_data)
        else:
            logger.info(self.message, extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }


# Specific error classes for different domains
class DataIngestionError(PricingIntelligenceError):
    """Error in data ingestion processes."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            **kwargs
        )


class RetryableError(PricingIntelligenceError):
    """Error that can be retried."""
    
    def __init__(self, message: str, retry_after_seconds: int = 60, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.retry_after_seconds = retry_after_seconds


class ValidationError(PricingIntelligenceError):
    """Data validation error."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )
        self.field_name = field_name


class ConfigurationError(PricingIntelligenceError):
    """Configuration error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class MLModelError(PricingIntelligenceError):
    """Machine learning model error."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.ML_MODEL,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.model_name = model_name


def handle_exceptions(func):
    """Decorator for handling exceptions in functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PricingIntelligenceError:
            raise  # Re-raise application errors
        except Exception as e:
            # Convert to application error
            raise PricingIntelligenceError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.HIGH,
                cause=e
            ) from e
    return wrapper


def get_http_status_code(error: PricingIntelligenceError) -> int:
    """
    Get appropriate HTTP status code for an error.
    
    Args:
        error: Application error
        
    Returns:
        HTTP status code
    """
    status_map = {
        ErrorCategory.VALIDATION: 400,
        ErrorCategory.AUTHENTICATION: 401,
        ErrorCategory.AUTHORIZATION: 403,
        ErrorCategory.RESOURCE_NOT_FOUND: 404,
        ErrorCategory.RATE_LIMIT: 429,
        ErrorCategory.EXTERNAL_SERVICE: 502,
        ErrorCategory.DATABASE: 503,
        ErrorCategory.ML_MODEL: 503,
        ErrorCategory.IMAGE_PROCESSING: 422,
        ErrorCategory.PRICING_ENGINE: 500,
        ErrorCategory.CONFIGURATION: 500,
        ErrorCategory.INTERNAL: 500
    }
    
    return status_map.get(error.category, 500)


class ErrorHandler:
    """Context manager for comprehensive error handling."""
    
    def __init__(
        self,
        operation_name: str,
        reraise: bool = True,
        default_return: Any = None
    ):
        self.operation_name = operation_name
        self.reraise = reraise
        self.default_return = default_return
        self.error: Optional[Exception] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            
            if isinstance(exc_val, PricingIntelligenceError):
                if self.reraise:
                    return False  # Reraise the exception
                return True  # Suppress the exception
            
            # Convert to application error
            app_error = PricingIntelligenceError(
                message=f"Operation '{self.operation_name}' failed: {str(exc_val)}",
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.MEDIUM,
                details={
                    "operation": self.operation_name,
                    "traceback": traceback.format_exc()
                },
                cause=exc_val
            )
            
            self.error = app_error
            
            if self.reraise:
                raise app_error from exc_val
            
            return True  # Suppress the exception
        
        return False
    
    def get_result(self):
        """Get the default return value if an error occurred."""
        if self.error:
            return self.default_return
        return None


def handle_api_error(func):
    """Decorator specifically for API endpoint error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PricingIntelligenceError as e:
            # Return structured error response
            return {
                "error": e.to_dict(),
                "status_code": get_http_status_code(e)
            }
        except Exception as e:
            # Convert unexpected errors
            app_error = PricingIntelligenceError(
                message=f"API error in {func.__name__}: {str(e)}",
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.HIGH,
                cause=e
            )
            return {
                "error": app_error.to_dict(),
                "status_code": 500
            }
    return wrapper


# Utility functions for common error scenarios
def get_settings():
    """Get application settings - utility function for imports."""
    from ..config.settings import settings
    return settings


def setup_logging():
    """Setup logging - utility function for imports."""
    from .logging_utils import configure_logging
    return configure_logging()
