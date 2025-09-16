"""
Google Cloud Platform utility functions for the Dynamic Pricing Intelligence System.
Provides common GCP operations and error handling.
"""

import json
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone

from google.api_core import exceptions as gcp_exceptions
from google.cloud.exceptions import GoogleCloudError

from .logging_utils import get_logger
from .error_handling import PricingIntelligenceError, ErrorCategory, ErrorSeverity


logger = get_logger(__name__)


def handle_gcp_error(func):
    """
    Decorator to handle GCP-specific errors and convert them to application errors.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with GCP error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except gcp_exceptions.NotFound as e:
            raise PricingIntelligenceError(
                message=f"GCP resource not found: {str(e)}",
                category=ErrorCategory.RESOURCE_NOT_FOUND,
                severity=ErrorSeverity.MEDIUM,
                cause=e
            ) from e
        except gcp_exceptions.PermissionDenied as e:
            raise PricingIntelligenceError(
                message=f"GCP permission denied: {str(e)}",
                category=ErrorCategory.AUTHORIZATION,
                severity=ErrorSeverity.HIGH,
                cause=e
            ) from e
        except gcp_exceptions.Unauthenticated as e:
            raise PricingIntelligenceError(
                message=f"GCP authentication failed: {str(e)}",
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.HIGH,
                cause=e
            ) from e
        except gcp_exceptions.ResourceExhausted as e:
            raise PricingIntelligenceError(
                message=f"GCP quota exceeded: {str(e)}",
                category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.HIGH,
                cause=e
            ) from e
        except gcp_exceptions.DeadlineExceeded as e:
            raise PricingIntelligenceError(
                message=f"GCP operation timeout: {str(e)}",
                category=ErrorCategory.EXTERNAL_SERVICE,
                severity=ErrorSeverity.MEDIUM,
                cause=e
            ) from e
        except gcp_exceptions.ServiceUnavailable as e:
            raise PricingIntelligenceError(
                message=f"GCP service unavailable: {str(e)}",
                category=ErrorCategory.EXTERNAL_SERVICE,
                severity=ErrorSeverity.HIGH,
                cause=e
            ) from e
        except GoogleCloudError as e:
            raise PricingIntelligenceError(
                message=f"GCP error: {str(e)}",
                category=ErrorCategory.EXTERNAL_SERVICE,
                severity=ErrorSeverity.MEDIUM,
                cause=e
            ) from e
        except Exception as e:
            raise PricingIntelligenceError(
                message=f"Unexpected error in GCP operation: {str(e)}",
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.HIGH,
                cause=e
            ) from e
    
    return wrapper


def format_gcp_timestamp(timestamp: Union[datetime, str, None]) -> Optional[str]:
    """
    Format timestamp for GCP services.
    
    Args:
        timestamp: Timestamp to format
        
    Returns:
        Formatted timestamp string or None
    """
    if timestamp is None:
        return None
    
    if isinstance(timestamp, str):
        return timestamp
    
    if isinstance(timestamp, datetime):
        # Ensure timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.isoformat()
    
    return str(timestamp)


def parse_gcp_timestamp(timestamp_str: Optional[str]) -> Optional[datetime]:
    """
    Parse GCP timestamp string to datetime.
    
    Args:
        timestamp_str: Timestamp string from GCP
        
    Returns:
        Parsed datetime or None
    """
    if not timestamp_str:
        return None
    
    try:
        # Handle various GCP timestamp formats
        if timestamp_str.endswith('Z'):
            return datetime.fromisoformat(timestamp_str[:-1] + '+00:00')
        else:
            return datetime.fromisoformat(timestamp_str)
    except ValueError:
        logger.warning(f"Failed to parse GCP timestamp: {timestamp_str}")
        return None


def build_gcp_labels(
    service_name: str,
    environment: str,
    additional_labels: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Build standardized GCP resource labels.
    
    Args:
        service_name: Name of the service
        environment: Environment (dev, staging, prod)
        additional_labels: Additional labels to include
        
    Returns:
        Dictionary of GCP labels
    """
    labels = {
        'service': service_name.lower().replace('_', '-'),
        'environment': environment.lower(),
        'managed-by': 'dynamic-pricing-system',
        'created-at': datetime.now(timezone.utc).strftime('%Y-%m-%d')
    }
    
    if additional_labels:
        # Ensure label values are valid for GCP
        for key, value in additional_labels.items():
            # GCP label values must be lowercase and contain only letters, numbers, and hyphens
            clean_key = key.lower().replace('_', '-')
            clean_value = str(value).lower().replace('_', '-').replace(' ', '-')
            labels[clean_key] = clean_value
    
    return labels


def validate_gcp_resource_name(name: str, resource_type: str = "resource") -> str:
    """
    Validate and clean GCP resource name.
    
    Args:
        name: Resource name to validate
        resource_type: Type of resource for error messages
        
    Returns:
        Cleaned resource name
        
    Raises:
        ValueError: If name is invalid
    """
    if not name:
        raise ValueError(f"{resource_type} name cannot be empty")
    
    # Remove invalid characters and convert to lowercase
    clean_name = name.lower().replace('_', '-').replace(' ', '-')
    
    # Remove consecutive hyphens
    while '--' in clean_name:
        clean_name = clean_name.replace('--', '-')
    
    # Remove leading/trailing hyphens
    clean_name = clean_name.strip('-')
    
    # Validate length
    if len(clean_name) < 1:
        raise ValueError(f"{resource_type} name too short after cleaning")
    if len(clean_name) > 63:
        raise ValueError(f"{resource_type} name too long (max 63 characters)")
    
    # Validate format (must start with letter, contain only letters, numbers, hyphens)
    if not clean_name[0].isalpha():
        clean_name = f"res-{clean_name}"
    
    # Final validation
    import re
    if not re.match(r'^[a-z][a-z0-9-]*[a-z0-9]$', clean_name):
        raise ValueError(f"Invalid {resource_type} name format: {clean_name}")
    
    return clean_name


def extract_gcp_error_details(error: Exception) -> Dict[str, Any]:
    """
    Extract detailed information from GCP errors.
    
    Args:
        error: GCP exception
        
    Returns:
        Dictionary with error details
    """
    details = {
        'error_type': type(error).__name__,
        'message': str(error)
    }
    
    # Extract additional details from GCP exceptions
    if hasattr(error, 'code'):
        details['error_code'] = error.code
    
    if hasattr(error, 'details'):
        details['error_details'] = error.details
    
    if hasattr(error, 'response'):
        response = error.response
        if hasattr(response, 'status_code'):
            details['http_status'] = response.status_code
        if hasattr(response, 'reason'):
            details['http_reason'] = response.reason
    
    return details


def format_bigquery_table_id(
    project_id: str,
    dataset_id: str,
    table_id: str
) -> str:
    """
    Format BigQuery table ID in standard format.
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        
    Returns:
        Formatted table ID
    """
    return f"{project_id}.{dataset_id}.{table_id}"


def format_cloud_storage_path(
    bucket_name: str,
    object_path: str = ""
) -> str:
    """
    Format Cloud Storage path in gs:// format.
    
    Args:
        bucket_name: Cloud Storage bucket name
        object_path: Object path within bucket
        
    Returns:
        Formatted Cloud Storage path
    """
    if object_path:
        return f"gs://{bucket_name}/{object_path.lstrip('/')}"
    return f"gs://{bucket_name}"


def parse_cloud_storage_path(gs_path: str) -> Dict[str, str]:
    """
    Parse Cloud Storage gs:// path into components.
    
    Args:
        gs_path: Cloud Storage path in gs:// format
        
    Returns:
        Dictionary with bucket and object path
        
    Raises:
        ValueError: If path format is invalid
    """
    if not gs_path.startswith('gs://'):
        raise ValueError("Cloud Storage path must start with gs://")
    
    path_parts = gs_path[5:].split('/', 1)  # Remove gs:// prefix
    
    if len(path_parts) == 1:
        return {'bucket': path_parts[0], 'object_path': ''}
    else:
        return {'bucket': path_parts[0], 'object_path': path_parts[1]}


def build_pubsub_topic_path(
    project_id: str,
    topic_name: str
) -> str:
    """
    Build Pub/Sub topic path.
    
    Args:
        project_id: GCP project ID
        topic_name: Pub/Sub topic name
        
    Returns:
        Full topic path
    """
    return f"projects/{project_id}/topics/{topic_name}"


def build_pubsub_subscription_path(
    project_id: str,
    subscription_name: str
) -> str:
    """
    Build Pub/Sub subscription path.
    
    Args:
        project_id: GCP project ID
        subscription_name: Pub/Sub subscription name
        
    Returns:
        Full subscription path
    """
    return f"projects/{project_id}/subscriptions/{subscription_name}"


def retry_gcp_operation(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    retry_on: tuple = (gcp_exceptions.ServiceUnavailable, gcp_exceptions.DeadlineExceeded)
):
    """
    Decorator for retrying GCP operations with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff multiplier for retry delays
        retry_on: Tuple of exception types to retry on
        
    Returns:
        Decorator function
    """
    import time
    import random
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Calculate delay with jitter
                    delay = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                    
                    logger.warning(
                        f"GCP operation failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    
                    time.sleep(delay)
            
            # All retries exhausted
            raise PricingIntelligenceError(
                message=f"GCP operation failed after {max_retries + 1} attempts: {str(last_exception)}",
                category=ErrorCategory.EXTERNAL_SERVICE,
                severity=ErrorSeverity.HIGH,
                cause=last_exception
            ) from last_exception
        
        return wrapper
    return decorator
