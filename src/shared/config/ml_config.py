"""
Machine Learning Configuration
Centralized ML model configurations, hyperparameters, and training settings
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.shared.config.settings_fixed import get_settings
from src.shared.utils.logging_utils import setup_logging

# Setup logging
try:
    setup_logging()
    logger = logging.getLogger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

try:
    settings = get_settings()
except Exception:
    # Fallback settings object
    class FallbackSettings:
        project_id: str = 'test-project'
        region: str = 'us-central1'
    settings = FallbackSettings()

@dataclass
class ModelConfig:
    """Configuration for individual ML models"""
    name: str
    version: str
    model_type: str
    framework: str
    input_shape: List[int]
    output_shape: List[int]
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    postprocessing: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    performance_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class TrainingConfig:
    """Training configuration for ML models"""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    optimizer: str = 'adam'
    loss_function: str = 'mse'
    metrics: List[str] = field(default_factory=lambda: ['accuracy', 'precision', 'recall'])
    callbacks: List[str] = field(default_factory=lambda: ['early_stopping', 'model_checkpoint'])
    data_augmentation: bool = False
    regularization: Dict[str, float] = field(default_factory=dict)

class MLConfig:
    """
    Centralized Machine Learning configuration management
    """
    
    def __init__(self):
        self.project_id = getattr(settings, 'project_id', 'test-project')
        self.region = getattr(settings, 'region', 'us-central1')
        
        # Model configurations
        self.models = self._initialize_model_configs()
        
        # Training configurations
        self.training_configs = self._initialize_training_configs()
        
        # Feature engineering configurations
        self.feature_configs = self._initialize_feature_configs()
        
        # Model serving configurations
        self.serving_configs = self._initialize_serving_configs()
        
        # Monitoring and evaluation configurations
        self.monitoring_configs = self._initialize_monitoring_configs()
        
        # Add missing attributes that are accessed in other files
        self.crowd_detection_model_path: str = f"gs://{self.project_id}-ml-models/crowd_detection/model.pkl"
        self.density_estimation_model_path: str = f"gs://{self.project_id}-ml-models/density_estimation/model.pkl"
        self.scene_classification_model_path: str = f"gs://{self.project_id}-ml-models/scene_classification/model.pkl"
        self.object_detection_model_path: str = f"gs://{self.project_id}-ml-models/object_detection/model.pkl"
        
        # Input sizes
        self.crowd_detection_input_size: tuple = (640, 640)
        self.scene_classification_input_size: tuple = (224, 224)
        
        # Confidence thresholds
        self.crowd_detection_confidence_threshold: float = 0.5
        self.object_detection_confidence_threshold: float = 0.5
        
        # Other model parameters
        self.density_to_count_ratio: float = 0.75
        
        logger.info("ML configuration initialized")
    
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize configurations for all ML models"""
        return {
            'crowd_detection': ModelConfig(
                name='crowd_detection',
                version='v2.1',
                model_type='object_detection',
                framework='tensorflow',
                input_shape=[640, 640, 3],
                output_shape=[100, 6],  # [max_detections, (x, y, w, h, confidence, class)]
                preprocessing={
                    'resize_method': 'bilinear',
                    'normalization': 'imagenet',
                    'augmentation': {
                        'horizontal_flip': True,
                        'rotation_range': 15,
                        'brightness_range': [0.8, 1.2],
                        'contrast_range': [0.8, 1.2]
                    }
                },
                postprocessing={
                    'nms_threshold': 0.4,
                    'confidence_threshold': 0.5,
                    'max_detections': 100
                },
                hyperparameters={
                    'backbone': 'efficientdet_d2',
                    'anchor_scales': [1.0, 1.26, 1.59],
                    'aspect_ratios': [0.5, 1.0, 2.0],
                    'num_classes': 1,  # person class only
                    'focal_loss_alpha': 0.25,
                    'focal_loss_gamma': 2.0
                },
                training_config={
                    'batch_size': 16,
                    'learning_rate': 0.0001,
                    'epochs': 200,
                    'warmup_epochs': 5,
                    'cosine_decay': True
                },
                deployment_config={
                    'machine_type': 'n1-standard-4',
                    'accelerator_type': 'NVIDIA_TESLA_T4',
                    'min_replica_count': 1,
                    'max_replica_count': 10,
                    'target_utilization': 70
                },
                performance_thresholds={
                    'map_50': 0.85,
                    'precision': 0.90,
                    'recall': 0.85,
                    'inference_time_ms': 200
                }
            ),
            
            'activity_classification': ModelConfig(
                name='activity_classification',
                version='v1.8',
                model_type='classification',
                framework='tensorflow',
                input_shape=[224, 224, 3],
                output_shape=[20],  # 20 activity classes
                preprocessing={
                    'resize_method': 'bilinear',
                    'normalization': 'imagenet',
                    'center_crop': True,
                    'augmentation': {
                        'horizontal_flip': True,
                        'rotation_range': 10,
                        'zoom_range': 0.1,
                        'shear_range': 0.1
                    }
                },
                postprocessing={
                    'softmax_temperature': 1.0,
                    'top_k': 5,
                    'confidence_threshold': 0.3
                },
                hyperparameters={
                    'backbone': 'efficientnet_b3',
                    'dropout_rate': 0.3,
                    'label_smoothing': 0.1,
                    'mixup_alpha': 0.2
                },
                training_config={
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'epochs': 150,
                    'fine_tuning_epochs': 50,
                    'freeze_backbone_epochs': 20
                },
                deployment_config={
                    'machine_type': 'n1-standard-2',
                    'min_replica_count': 1,
                    'max_replica_count': 5,
                    'target_utilization': 60
                },
                performance_thresholds={
                    'accuracy': 0.92,
                    'top_5_accuracy': 0.98,
                    'f1_score': 0.90,
                    'inference_time_ms': 50
                }
            ),
            
            'demand_forecasting': ModelConfig(
                name='demand_forecasting',
                version='v3.2',
                model_type='time_series',
                framework='tensorflow',
                input_shape=[24, 15],  # 24 hours, 15 features
                output_shape=[24, 3],  # 24 hours forecast with confidence intervals
                preprocessing={
                    'normalization': 'z_score',
                    'seasonal_decomposition': True,
                    'feature_engineering': {
                        'lag_features': [1, 2, 3, 6, 12, 24],
                        'rolling_features': [3, 6, 12],
                        'cyclical_features': ['hour', 'day_of_week', 'month']
                    }
                },
                postprocessing={
                    'denormalization': True,
                    'confidence_intervals': True,
                    'anomaly_detection': True
                },
                hyperparameters={
                    'lstm_units': [128, 64, 32],
                    'attention_heads': 8,
                    'dropout_rate': 0.2,
                    'recurrent_dropout': 0.1,
                    'ensemble_size': 5
                },
                training_config={
                    'batch_size': 64,
                    'learning_rate': 0.0005,
                    'epochs': 300,
                    'sequence_length': 24,
                    'forecast_horizon': 24
                },
                deployment_config={
                    'machine_type': 'n1-highmem-2',
                    'min_replica_count': 2,
                    'max_replica_count': 8,
                    'target_utilization': 75
                },
                performance_thresholds={
                    'mape': 0.15,  # Mean Absolute Percentage Error
                    'rmse': 5.0,   # Root Mean Square Error
                    'mae': 3.0,    # Mean Absolute Error
                    'inference_time_ms': 100
                }
            ),
            
            'price_optimization': ModelConfig(
                name='price_optimization',
                version='v2.5',
                model_type='reinforcement_learning',
                framework='tensorflow',
                input_shape=[50],  # Combined feature vector
                output_shape=[1],  # Optimal price multiplier
                preprocessing={
                    'feature_scaling': 'robust',
                    'feature_selection': True,
                    'dimensionality_reduction': False
                },
                postprocessing={
                    'constraint_enforcement': True,
                    'business_rules_validation': True,
                    'explainability_generation': True
                },
                hyperparameters={
                    'network_architecture': [256, 128, 64, 32],
                    'activation': 'relu',
                    'optimizer': 'adam',
                    'learning_rate_schedule': 'exponential_decay',
                    'reward_function': 'multi_objective'
                },
                training_config={
                    'batch_size': 128,
                    'learning_rate': 0.0001,
                    'episodes': 10000,
                    'replay_buffer_size': 100000,
                    'target_update_frequency': 1000
                },
                deployment_config={
                    'machine_type': 'n1-standard-4',
                    'min_replica_count': 2,
                    'max_replica_count': 6,
                    'target_utilization': 80
                },
                performance_thresholds={
                    'revenue_improvement': 0.15,
                    'customer_satisfaction': 0.85,
                    'convergence_episodes': 5000,
                    'inference_time_ms': 30
                }
            ),
            
            'location_embeddings': ModelConfig(
                name='location_embeddings',
                version='v1.4',
                model_type='embedding',
                framework='tensorflow',
                input_shape=[100],  # Location feature vector
                output_shape=[64],  # Embedding dimension
                preprocessing={
                    'text_preprocessing': {
                        'tokenization': 'bert',
                        'max_sequence_length': 512,
                        'vocabulary_size': 30000
                    },
                    'categorical_encoding': 'target_encoding',
                    'numerical_scaling': 'standard'
                },
                postprocessing={
                    'similarity_computation': 'cosine',
                    'clustering': 'kmeans',
                    'dimensionality_reduction': 'umap'
                },
                hyperparameters={
                    'embedding_dim': 64,
                    'hidden_layers': [128, 64],
                    'dropout_rate': 0.1,
                    'l2_regularization': 0.001
                },
                training_config={
                    'batch_size': 256,
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'negative_sampling_rate': 5
                },
                deployment_config={
                    'machine_type': 'n1-standard-2',
                    'min_replica_count': 1,
                    'max_replica_count': 3,
                    'target_utilization': 50
                },
                performance_thresholds={
                    'embedding_quality': 0.85,
                    'clustering_silhouette': 0.7,
                    'similarity_accuracy': 0.90,
                    'inference_time_ms': 20
                }
            )
        }
    
    def _initialize_training_configs(self) -> Dict[str, TrainingConfig]:
        """Initialize training configurations"""
        return {
            'default': TrainingConfig(),
            'computer_vision': TrainingConfig(
                batch_size=16,
                learning_rate=0.0001,
                epochs=200,
                validation_split=0.15,
                early_stopping_patience=15,
                optimizer='adamw',
                loss_function='focal_loss',
                metrics=['map', 'precision', 'recall'],
                data_augmentation=True,
                regularization={'l2': 0.0001, 'dropout': 0.3}
            ),
            'time_series': TrainingConfig(
                batch_size=64,
                learning_rate=0.0005,
                epochs=300,
                validation_split=0.2,
                early_stopping_patience=20,
                optimizer='adam',
                loss_function='huber',
                metrics=['mae', 'mape', 'rmse'],
                callbacks=['reduce_lr_on_plateau', 'early_stopping']
            ),
            'reinforcement_learning': TrainingConfig(
                batch_size=128,
                learning_rate=0.0001,
                epochs=10000,
                optimizer='adam',
                loss_function='mse',
                metrics=['reward', 'q_value', 'policy_loss'],
                callbacks=['tensorboard', 'model_checkpoint']
            )
        }
    
    def _initialize_feature_configs(self) -> Dict[str, Any]:
        """Initialize feature engineering configurations"""
        return {
            'visual_features': {
                'crowd_density': {
                    'extraction_method': 'object_detection',
                    'aggregation': ['count', 'density_per_sqm', 'spatial_distribution'],
                    'temporal_features': ['trend', 'seasonality', 'anomaly_score']
                },
                'activity_features': {
                    'extraction_method': 'classification',
                    'categories': [
                        'commuting', 'shopping', 'dining', 'entertainment',
                        'business_meeting', 'tourism', 'events', 'construction',
                        'emergency', 'delivery', 'maintenance', 'protest'
                    ],
                    'aggregation': ['dominant_activity', 'activity_diversity', 'intensity_score']
                },
                'accessibility_features': {
                    'extraction_method': 'scene_analysis',
                    'factors': ['parking_availability', 'traffic_congestion', 'road_conditions', 'barriers'],
                    'scoring': 'weighted_average'
                }
            },
            'contextual_features': {
                'temporal': {
                    'time_of_day': {'encoding': 'cyclical', 'bins': 24},
                    'day_of_week': {'encoding': 'one_hot'},
                    'month': {'encoding': 'cyclical', 'bins': 12},
                    'season': {'encoding': 'one_hot'},
                    'holiday': {'encoding': 'binary', 'source': 'calendar_api'}
                },
                'weather': {
                    'temperature': {'normalization': 'z_score', 'outlier_handling': 'clip'},
                    'precipitation': {'transformation': 'log1p'},
                    'wind_speed': {'normalization': 'min_max'},
                    'visibility': {'normalization': 'min_max'},
                    'weather_condition': {'encoding': 'target_encoding'}
                },
                'events': {
                    'event_type': {'encoding': 'embedding', 'dim': 16},
                    'event_size': {'normalization': 'log', 'bins': 10},
                    'distance_to_event': {'transformation': 'inverse', 'max_distance': 5000}
                }
            },
            'location_features': {
                'static': {
                    'business_district': {'encoding': 'target_encoding'},
                    'demographic_profile': {'encoding': 'embedding', 'dim': 32},
                    'poi_density': {'normalization': 'log1p'},
                    'transport_accessibility': {'scoring': 'composite_index'}
                },
                'dynamic': {
                    'traffic_flow': {'smoothing': 'exponential', 'window': 15},
                    'competitor_presence': {'aggregation': 'weighted_average'},
                    'supply_availability': {'normalization': 'z_score'}
                }
            }
        }
    
    def _initialize_serving_configs(self) -> Dict[str, Any]:
        """Initialize model serving configurations"""
        return {
            'batch_prediction': {
                'max_batch_size': 1000,
                'timeout_seconds': 300,
                'retry_policy': {
                    'max_retries': 3,
                    'backoff_multiplier': 2.0,
                    'initial_delay_seconds': 1
                }
            },
            'online_prediction': {
                'max_latency_ms': 100,
                'auto_scaling': {
                    'min_replicas': 1,
                    'max_replicas': 10,
                    'target_cpu_utilization': 70,
                    'scale_up_cooldown': 60,
                    'scale_down_cooldown': 300
                },
                'caching': {
                    'enabled': True,
                    'ttl_seconds': 300,
                    'max_cache_size': 10000
                }
            },
            'model_versioning': {
                'strategy': 'blue_green',
                'rollback_threshold': 0.05,  # 5% error rate increase
                'canary_traffic_percentage': 10,
                'monitoring_window_minutes': 30
            }
        }
    
    def _initialize_monitoring_configs(self) -> Dict[str, Any]:
        """Initialize model monitoring configurations"""
        return {
            'performance_monitoring': {
                'metrics': {
                    'accuracy': {'threshold': 0.90, 'alert_on_drop': 0.05},
                    'latency': {'threshold_ms': 100, 'percentile': 95},
                    'throughput': {'min_rps': 10, 'max_rps': 1000},
                    'error_rate': {'threshold': 0.01, 'window_minutes': 5}
                },
                'data_drift': {
                    'detection_method': 'ks_test',
                    'significance_level': 0.05,
                    'monitoring_features': 'all',
                    'alert_threshold': 0.1
                },
                'model_drift': {
                    'detection_method': 'psi',  # Population Stability Index
                    'threshold': 0.2,
                    'monitoring_window_days': 7,
                    'baseline_window_days': 30
                }
            },
            'business_metrics': {
                'revenue_impact': {
                    'baseline_period_days': 30,
                    'minimum_improvement': 0.05,
                    'alert_on_decline': 0.02
                },
                'customer_satisfaction': {
                    'target_score': 4.5,
                    'minimum_score': 4.0,
                    'survey_response_rate': 0.1
                },
                'operational_efficiency': {
                    'driver_utilization': {'target': 0.8, 'minimum': 0.7},
                    'wait_time_reduction': {'target': 0.3, 'minimum': 0.1}
                }
            },
            'alerting': {
                'channels': ['email', 'slack', 'pagerduty'],
                'severity_levels': {
                    'critical': {'response_time_minutes': 15},
                    'high': {'response_time_minutes': 60},
                    'medium': {'response_time_minutes': 240},
                    'low': {'response_time_hours': 24}
                },
                'escalation_policy': {
                    'primary_oncall': 'ml_team',
                    'secondary_oncall': 'engineering_team',
                    'escalation_delay_minutes': 30
                }
            }
        }
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model"""
        return self.models.get(model_name)
    
    def get_training_config(self, config_type: str = 'default') -> TrainingConfig:
        """Get training configuration"""
        return self.training_configs.get(config_type, self.training_configs['default'])
    
    def get_feature_config(self, feature_type: str) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self.feature_configs.get(feature_type, {})
    
    def update_model_config(self, model_name: str, updates: Dict[str, Any]) -> bool:
        """Update model configuration"""
        if model_name not in self.models:
            return False
        
        model_config = self.models[model_name]
        
        # Update specific fields
        for key, value in updates.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        
        logger.info(f"Updated configuration for model {model_name}")
        return True
    
    def validate_model_config(self, model_name: str) -> Dict[str, bool]:
        """Validate model configuration"""
        if model_name not in self.models:
            return {'exists': False}
        
        config = self.models[model_name]
        validation_results = {'exists': True}
        
        # Validate required fields
        required_fields = ['name', 'version', 'model_type', 'framework', 'input_shape', 'output_shape']
        for field in required_fields:
            validation_results[f'has_{field}'] = hasattr(config, field) and getattr(config, field) is not None
        
        # Validate performance thresholds
        validation_results['has_performance_thresholds'] = bool(config.performance_thresholds)
        
        # Validate deployment config
        validation_results['has_deployment_config'] = bool(config.deployment_config)
        
        return validation_results
    
    def export_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        if model_name:
            if model_name in self.models:
                default_training = self.training_configs.get('default')
                return {
                    'model': self.models[model_name].__dict__,
                    'training': default_training.__dict__ if default_training else {},
                    'features': self.feature_configs,
                    'serving': self.serving_configs,
                    'monitoring': self.monitoring_configs
                }
            return {}
        
        return {
            'models': {name: config.__dict__ for name, config in self.models.items()},
            'training': {name: config.__dict__ for name, config in self.training_configs.items()},
            'features': self.feature_configs,
            'serving': self.serving_configs,
            'monitoring': self.monitoring_configs
        }

# Global ML configuration instance
_ml_config = None

def get_ml_config() -> MLConfig:
    """Get singleton ML configuration instance"""
    global _ml_config
    if _ml_config is None:
        _ml_config = MLConfig()
    return _ml_config
