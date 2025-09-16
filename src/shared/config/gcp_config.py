"""
Google Cloud Platform Configuration
Centralized GCP service configurations and client initialization
"""

import os
import logging
from typing import Dict, Any, Optional
from google.cloud import bigquery, storage, pubsub_v1, aiplatform
from google.oauth2 import service_account
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.shared.config.settings_fixed import settings
from src.shared.utils.logging_utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

class GCPConfig:
    """
    Centralized Google Cloud Platform configuration and client management
    """
    
    def __init__(self):
        self.project_id = settings.gcp.project_id
        self.region = settings.gcp.region
        self.credentials = self._get_credentials()
        
        # Service configurations
        self.bigquery_config = self._get_bigquery_config()
        self.storage_config = self._get_storage_config()
        self.pubsub_config = self._get_pubsub_config()
        self.vertex_ai_config = self._get_vertex_ai_config()
        
        # Add missing attributes that are accessed in other files
        self.temp_bucket = f"{self.project_id}-temp-data"
        self.street_imagery_bucket = f"{self.project_id}-street-imagery"
        self.bigquery_dataset = getattr(settings, 'bigquery_dataset', 'ride_intelligence')
        
        # Vertex AI endpoints
        self.crowd_detection_endpoint = f"projects/{self.project_id}/locations/{self.region}/endpoints/crowd-detection"
        self.scene_classification_endpoint = f"projects/{self.project_id}/locations/{self.region}/endpoints/scene-classification"
        self.accessibility_analysis_endpoint = f"projects/{self.project_id}/locations/{self.region}/endpoints/accessibility-analysis"
        
        logger.info(f"GCP configuration initialized for project {self.project_id}")
    
    def _get_credentials(self) -> Optional[service_account.Credentials]:
        """Get GCP credentials from service account key or default"""
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if credentials_path and os.path.exists(credentials_path):
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=[
                        'https://www.googleapis.com/auth/cloud-platform',
                        'https://www.googleapis.com/auth/bigquery',
                        'https://www.googleapis.com/auth/devstorage.full_control',
                        'https://www.googleapis.com/auth/pubsub'
                    ]
                )
                logger.info("Using service account credentials")
                return credentials
            except Exception as e:
                logger.warning(f"Could not load service account credentials: {e}")
        
        logger.info("Using default credentials")
        return None
    
    def _get_bigquery_config(self) -> Dict[str, Any]:
        """Get BigQuery configuration"""
        return {
            'project_id': self.project_id,
            'location': getattr(settings, 'bigquery_location', 'US'),
            'dataset': getattr(settings, 'bigquery_dataset', 'ride_intelligence'),
            'job_config': {
                'use_query_cache': True,
                'use_legacy_sql': False,
                'maximum_bytes_billed': 1000000000,  # 1GB limit
                'job_timeout_ms': 300000,  # 5 minutes
                'labels': {
                    'environment': getattr(settings, 'environment', 'development'),
                    'service': 'pricing-intelligence'
                }
            },
            'client_config': {
                'default_query_job_config': bigquery.QueryJobConfig(
                    use_query_cache=True,
                    use_legacy_sql=False
                )
            }
        }
    
    def _get_storage_config(self) -> Dict[str, Any]:
        """Get Cloud Storage configuration"""
        return {
            'project_id': self.project_id,
            'buckets': {
                'street_imagery': f"{self.project_id}-street-imagery",
                'ml_models': f"{self.project_id}-ml-models",
                'data_processing': f"{self.project_id}-data-processing",
                'backups': f"{self.project_id}-backups",
                'analytics': f"{self.project_id}-analytics"
            },
            'default_location': self.region,
            'lifecycle_policies': {
                'street_imagery': [
                    {
                        'action': {'type': 'SetStorageClass', 'storageClass': 'COLDLINE'},
                        'condition': {'age': 90}
                    }
                ],
                'data_processing': [
                    {
                        'action': {'type': 'Delete'},
                        'condition': {'age': 7}
                    }
                ],
                'backups': [
                    {
                        'action': {'type': 'SetStorageClass', 'storageClass': 'ARCHIVE'},
                        'condition': {'age': 30}
                    },
                    {
                        'action': {'type': 'Delete'},
                        'condition': {'age': 365}
                    }
                ]
            }
        }
    
    def _get_pubsub_config(self) -> Dict[str, Any]:
        """Get Pub/Sub configuration"""
        return {
            'project_id': self.project_id,
            'topics': {
                'realtime_events': 'realtime-events',
                'image_processing': 'image-processing',
                'pricing_updates': 'pricing-updates',
                'model_training': 'model-training'
            },
            'subscriptions': {
                'realtime_events': 'realtime-events-subscription',
                'image_processing': 'image-processing-subscription',
                'pricing_updates': 'pricing-updates-subscription'
            },
            'publisher_config': {
                'max_messages': 1000,
                'max_latency': 0.1,  # 100ms
                'max_bytes': 1024 * 1024,  # 1MB
            },
            'subscriber_config': {
                'max_messages': 100,
                'ack_deadline_seconds': 600,
                'message_retention_duration': 604800,  # 7 days
                'flow_control': {
                    'max_messages': 1000,
                    'max_bytes': 1024 * 1024 * 100  # 100MB
                }
            }
        }
    
    def _get_vertex_ai_config(self) -> Dict[str, Any]:
        """Get Vertex AI configuration"""
        return {
            'project_id': self.project_id,
            'location': self.region,
            'staging_bucket': f"gs://{self.project_id}-ml-models",
            'endpoints': {
                'crowd_detection': f"projects/{self.project_id}/locations/{self.region}/endpoints/crowd-detection-endpoint",
                'activity_classification': f"projects/{self.project_id}/locations/{self.region}/endpoints/activity-classification-endpoint",
                'demand_forecasting': f"projects/{self.project_id}/locations/{self.region}/endpoints/demand-forecasting-endpoint",
                'price_optimization': f"projects/{self.project_id}/locations/{self.region}/endpoints/price-optimization-endpoint"
            },
            'model_configs': {
                'crowd_detection': {
                    'model_type': 'yolo_v5',
                    'input_size': [640, 640],
                    'confidence_threshold': 0.5,
                    'nms_threshold': 0.4
                },
                'activity_classification': {
                    'model_type': 'resnet50',
                    'input_size': [224, 224],
                    'num_classes': 20,
                    'confidence_threshold': 0.3
                },
                'demand_forecasting': {
                    'model_type': 'lstm_ensemble',
                    'sequence_length': 24,
                    'forecast_horizon': 24,
                    'features': ['historical_demand', 'weather', 'events', 'visual_features']
                },
                'price_optimization': {
                    'model_type': 'multi_objective_optimizer',
                    'objectives': ['revenue', 'customer_satisfaction', 'market_share'],
                    'constraints': {
                        'min_multiplier': 0.8,
                        'max_multiplier': 3.0
                    }
                }
            },
            'training_config': {
                'machine_type': 'n1-standard-4',
                'accelerator_type': 'NVIDIA_TESLA_T4',
                'accelerator_count': 1,
                'disk_type': 'pd-ssd',
                'disk_size_gb': 100
            }
        }
    
    def get_bigquery_client(self) -> bigquery.Client:
        """Get configured BigQuery client"""
        return bigquery.Client(
            project=self.project_id,
            credentials=self.credentials,
            location=self.bigquery_config['location']
        )
    
    def get_storage_client(self) -> storage.Client:
        """Get configured Cloud Storage client"""
        return storage.Client(
            project=self.project_id,
            credentials=self.credentials
        )
    
    def get_pubsub_publisher(self) -> pubsub_v1.PublisherClient:
        """Get configured Pub/Sub publisher client"""
        return pubsub_v1.PublisherClient(credentials=self.credentials)
    
    def get_pubsub_subscriber(self) -> pubsub_v1.SubscriberClient:
        """Get configured Pub/Sub subscriber client"""
        return pubsub_v1.SubscriberClient(credentials=self.credentials)
    
    def initialize_vertex_ai(self):
        """Initialize Vertex AI with project configuration"""
        aiplatform.init(
            project=self.project_id,
            location=self.region,
            credentials=self.credentials,
            staging_bucket=self.vertex_ai_config['staging_bucket']
        )
    
    def get_topic_path(self, topic_name: str) -> str:
        """Get full topic path"""
        topic_id = self.pubsub_config['topics'].get(topic_name, topic_name)
        return f"projects/{self.project_id}/topics/{topic_id}"
    
    def get_subscription_path(self, subscription_name: str) -> str:
        """Get full subscription path"""
        subscription_id = self.pubsub_config['subscriptions'].get(subscription_name, subscription_name)
        return f"projects/{self.project_id}/subscriptions/{subscription_id}"
    
    def get_bucket_name(self, bucket_type: str) -> str:
        """Get bucket name for specific type"""
        return self.storage_config['buckets'].get(bucket_type, f"{self.project_id}-{bucket_type}")
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate GCP configuration and connectivity"""
        validation_results = {}
        
        # Test BigQuery connectivity
        try:
            client = self.get_bigquery_client()
            list(client.list_datasets(max_results=1))
            validation_results['bigquery'] = True
        except Exception as e:
            logger.error(f"BigQuery validation failed: {e}")
            validation_results['bigquery'] = False
        
        # Test Cloud Storage connectivity
        try:
            client = self.get_storage_client()
            list(client.list_buckets(max_results=1))
            validation_results['storage'] = True
        except Exception as e:
            logger.error(f"Storage validation failed: {e}")
            validation_results['storage'] = False
        
        # Test Pub/Sub connectivity
        try:
            client = self.get_pubsub_publisher()
            client.list_topics(request={"project": f"projects/{self.project_id}"})
            validation_results['pubsub'] = True
        except Exception as e:
            logger.error(f"Pub/Sub validation failed: {e}")
            validation_results['pubsub'] = False
        
        # Test Vertex AI connectivity
        try:
            self.initialize_vertex_ai()
            validation_results['vertex_ai'] = True
        except Exception as e:
            logger.error(f"Vertex AI validation failed: {e}")
            validation_results['vertex_ai'] = False
        
        return validation_results

# Global configuration instance
_gcp_config = None

def get_gcp_config() -> GCPConfig:
    """Get singleton GCP configuration instance"""
    global _gcp_config
    if _gcp_config is None:
        _gcp_config = GCPConfig()
    return _gcp_config
