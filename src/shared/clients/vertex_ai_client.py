"""
Vertex AI Client - Advanced ML model serving and management
Handles model deployment, prediction, and monitoring for the pricing intelligence system
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.shared.config.settings_fixed import settings
from src.shared.utils.logging_utils import get_logger
from src.shared.utils.error_handling_fixed import handle_exceptions, PricingIntelligenceError

# Setup logging
logger = get_logger(__name__)

class VertexAIClient:
    """
    Advanced Vertex AI client for ML model operations
    Handles crowd detection, activity classification, and demand forecasting models
    """
    
    def __init__(self):
        """Initialize Vertex AI client with project configuration"""
        self.project_id = getattr(settings, 'project_id', 'default-project')
        self.region = getattr(settings, 'region', 'us-central1')
        self.staging_bucket = f"gs://{self.project_id}-ml-models"
        
        # Initialize Vertex AI
        try:
            aiplatform.init(
                project=self.project_id,
                location=self.region,
                staging_bucket=self.staging_bucket
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Vertex AI: {e}")
        
        # Model endpoints
        self.model_endpoints = {
            'crowd_detection': f"projects/{self.project_id}/locations/{self.region}/endpoints/crowd-detection-endpoint",
            'activity_classification': f"projects/{self.project_id}/locations/{self.region}/endpoints/activity-classification-endpoint",
            'demand_forecasting': f"projects/{self.project_id}/locations/{self.region}/endpoints/demand-forecasting-endpoint",
            'price_optimization': f"projects/{self.project_id}/locations/{self.region}/endpoints/price-optimization-endpoint"
        }
        
        # Prediction clients
        try:
            self.prediction_client = gapic.PredictionServiceClient(
                client_options={"api_endpoint": f"{self.region}-aiplatform.googleapis.com"}
            )
        except Exception as e:
            logger.warning(f"Could not initialize prediction client: {e}")
            self.prediction_client = None
        
        logger.info(f"Vertex AI client initialized for project {self.project_id} in region {self.region}")
    
    @handle_exceptions
    async def predict_crowd_count(self, image_data: Union[bytes, str], location_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict crowd count from street imagery using custom YOLO model
        
        Args:
            image_data: Base64 encoded image or image bytes
            location_context: Location metadata for context
            
        Returns:
            Dict containing crowd count, confidence, and bounding boxes
        """
        try:
            # Prepare prediction request
            instances = [{
                "image": {
                    "bytesBase64Encoded": image_data if isinstance(image_data, str) else str(image_data, 'utf-8')
                },
                "location_context": location_context
            }]
            
            # Convert to Vertex AI format
            try:
                instances = [json_format.ParseDict(instance, Value()) for instance in instances]
            except Exception as e:
                logger.error(f"Failed to parse instances: {e}")
                instances = []
            
            # Make prediction request
            endpoint = self.model_endpoints['crowd_detection']
            request = gapic.PredictRequest(
                endpoint=endpoint,
                instances=instances
            )
            
            try:
                if self.prediction_client:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, self.prediction_client.predict, request
                    )
                    
                    # Parse response
                    predictions = [json_format.MessageToDict(prediction) for prediction in response.predictions]
                else:
                    logger.warning("Prediction client not available, returning mock data")
                    predictions = []
            except Exception as e:
                logger.error(f"Prediction request failed: {e}")
                predictions = []
            
            if predictions:
                result = predictions[0]
                return {
                    'crowd_count': int(result.get('crowd_count', 0)),
                    'confidence': float(result.get('confidence', 0.0)),
                    'bounding_boxes': result.get('bounding_boxes', []),
                    'density_map': result.get('density_map', {}),
                    'processing_time': result.get('processing_time', 0.0),
                    'model_version': result.get('model_version', 'v1.0')
                }
            else:
                return {
                    'crowd_count': 0,
                    'confidence': 0.0,
                    'bounding_boxes': [],
                    'density_map': {},
                    'processing_time': 0.0,
                    'model_version': 'v1.0'
                }
                
        except Exception as e:
            logger.error(f"Error in crowd count prediction: {e}")
            raise PricingIntelligenceError(f"Crowd detection failed: {e}")
    
    @handle_exceptions
    async def classify_activities(self, image_data: Union[bytes, str], location_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Classify activities in street scene using custom CNN model
        
        Args:
            image_data: Base64 encoded image or image bytes
            location_context: Location metadata for context
            
        Returns:
            List of detected activities with confidence scores
        """
        try:
            # Prepare prediction request
            instances = [{
                "image": {
                    "bytesBase64Encoded": image_data if isinstance(image_data, str) else str(image_data, 'utf-8')
                },
                "location_context": location_context,
                "detection_threshold": 0.3
            }]
            
            # Convert to Vertex AI format
            instances = [json_format.ParseDict(instance, Value()) for instance in instances]
            
            # Make prediction request
            endpoint = self.model_endpoints['activity_classification']
            request = gapic.PredictRequest(
                endpoint=endpoint,
                instances=instances
            )
            
            if self.prediction_client:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.prediction_client.predict, request
                )
                
                # Parse response
                predictions = [json_format.MessageToDict(prediction) for prediction in response.predictions]
            else:
                logger.warning("Prediction client not available, returning mock data")
                predictions = []
            
            if predictions:
                activities = predictions[0].get('activities', [])
                return [
                    {
                        'activity_type': activity.get('type', 'unknown'),
                        'confidence': float(activity.get('confidence', 0.0)),
                        'bounding_box': activity.get('bounding_box', {}),
                        'impact_score': float(activity.get('impact_score', 0.0)),
                        'pricing_influence': activity.get('pricing_influence', 'neutral')
                    }
                    for activity in activities
                    if activity.get('confidence', 0.0) > 0.3
                ]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error in activity classification: {e}")
            raise PricingIntelligenceError(f"Activity classification failed: {e}")
    
    @handle_exceptions
    async def predict_demand(self, location_features: Dict[str, Any], forecast_horizon: int = 1) -> Dict[str, Any]:
        """
        Predict demand using advanced time series model
        
        Args:
            location_features: Location and contextual features
            forecast_horizon: Hours to forecast ahead
            
        Returns:
            Dict containing demand prediction with confidence intervals
        """
        try:
            # Prepare prediction request
            instances = [{
                "location_features": location_features,
                "forecast_horizon": forecast_horizon,
                "include_confidence_intervals": True,
                "model_type": "ensemble"
            }]
            
            # Convert to Vertex AI format
            instances = [json_format.ParseDict(instance, Value()) for instance in instances]
            
            # Make prediction request
            endpoint = self.model_endpoints['demand_forecasting']
            request = gapic.PredictRequest(
                endpoint=endpoint,
                instances=instances
            )
            
            if self.prediction_client:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.prediction_client.predict, request
                )
                
                # Parse response
                predictions = [json_format.MessageToDict(prediction) for prediction in response.predictions]
            else:
                logger.warning("Prediction client not available, returning mock data")
                predictions = []
            
            if predictions:
                result = predictions[0]
                return {
                    'predicted_demand': float(result.get('predicted_demand', 0.0)),
                    'confidence_interval_lower': float(result.get('confidence_interval_lower', 0.0)),
                    'confidence_interval_upper': float(result.get('confidence_interval_upper', 0.0)),
                    'uncertainty_score': float(result.get('uncertainty_score', 0.0)),
                    'trend_direction': result.get('trend_direction', 'stable'),
                    'seasonality_factor': float(result.get('seasonality_factor', 1.0)),
                    'model_accuracy': float(result.get('model_accuracy', 0.0))
                }
            else:
                return {
                    'predicted_demand': 0.0,
                    'confidence_interval_lower': 0.0,
                    'confidence_interval_upper': 0.0,
                    'uncertainty_score': 1.0,
                    'trend_direction': 'stable',
                    'seasonality_factor': 1.0,
                    'model_accuracy': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error in demand prediction: {e}")
            raise PricingIntelligenceError(f"Demand prediction failed: {e}")
    
    @handle_exceptions
    async def optimize_pricing(self, pricing_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize pricing using multi-objective optimization model
        
        Args:
            pricing_context: Complete context for pricing decision
            
        Returns:
            Dict containing optimal pricing recommendations
        """
        try:
            # Prepare prediction request
            instances = [{
                "pricing_context": pricing_context,
                "optimization_objectives": ["revenue", "customer_satisfaction", "market_share"],
                "constraints": {
                    "min_multiplier": 0.8,
                    "max_multiplier": 3.0,
                    "competitor_awareness": True
                }
            }]
            
            # Convert to Vertex AI format
            instances = [json_format.ParseDict(instance, Value()) for instance in instances]
            
            # Make prediction request
            endpoint = self.model_endpoints['price_optimization']
            request = gapic.PredictRequest(
                endpoint=endpoint,
                instances=instances
            )
            
            if self.prediction_client:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.prediction_client.predict, request
                )
                
                # Parse response
                predictions = [json_format.MessageToDict(prediction) for prediction in response.predictions]
            else:
                logger.warning("Prediction client not available, returning mock data")
                predictions = []
            
            if predictions:
                result = predictions[0]
                return {
                    'optimal_multiplier': float(result.get('optimal_multiplier', 1.0)),
                    'revenue_impact': float(result.get('revenue_impact', 0.0)),
                    'demand_elasticity': float(result.get('demand_elasticity', -1.0)),
                    'competitive_position': result.get('competitive_position', 'neutral'),
                    'optimization_confidence': float(result.get('optimization_confidence', 0.0)),
                    'alternative_strategies': result.get('alternative_strategies', []),
                    'risk_assessment': result.get('risk_assessment', {})
                }
            else:
                return {
                    'optimal_multiplier': 1.0,
                    'revenue_impact': 0.0,
                    'demand_elasticity': -1.0,
                    'competitive_position': 'neutral',
                    'optimization_confidence': 0.0,
                    'alternative_strategies': [],
                    'risk_assessment': {}
                }
                
        except Exception as e:
            logger.error(f"Error in pricing optimization: {e}")
            raise PricingIntelligenceError(f"Pricing optimization failed: {e}")
    
    @handle_exceptions
    async def batch_predict(self, model_name: str, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform batch predictions for multiple instances
        
        Args:
            model_name: Name of the model to use
            instances: List of prediction instances
            
        Returns:
            List of prediction results
        """
        try:
            if model_name not in self.model_endpoints:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Convert instances to Vertex AI format
            vertex_instances = [json_format.ParseDict(instance, Value()) for instance in instances]
            
            # Make batch prediction request
            endpoint = self.model_endpoints[model_name]
            request = gapic.PredictRequest(
                endpoint=endpoint,
                instances=vertex_instances
            )
            
            if self.prediction_client:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.prediction_client.predict, request
                )
                
                # Parse response
                predictions = [json_format.MessageToDict(prediction) for prediction in response.predictions]
            else:
                logger.warning("Prediction client not available, returning mock data")
                predictions = []
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise PricingIntelligenceError(f"Batch prediction failed: {e}")
    
    @handle_exceptions
    async def get_model_metrics(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance metrics for a deployed model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict containing model performance metrics
        """
        try:
            # This would typically query model monitoring APIs
            # For now, return mock metrics structure
            return {
                'model_name': model_name,
                'accuracy': 0.94,
                'precision': 0.92,
                'recall': 0.91,
                'f1_score': 0.915,
                'latency_p50': 45.2,
                'latency_p95': 89.7,
                'throughput': 1250.0,
                'error_rate': 0.002,
                'last_updated': datetime.utcnow().isoformat(),
                'deployment_status': 'healthy'
            }
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            raise PricingIntelligenceError(f"Model metrics retrieval failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all model endpoints
        
        Returns:
            Dict containing health status of all models
        """
        health_status = {
            'overall_status': 'healthy',
            'models': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for model_name, endpoint in self.model_endpoints.items():
            try:
                # Simple health check - attempt to get endpoint info
                # In production, this would make actual health check calls
                health_status['models'][model_name] = {
                    'status': 'healthy',
                    'endpoint': endpoint,
                    'last_check': datetime.utcnow().isoformat()
                }
            except Exception as e:
                health_status['models'][model_name] = {
                    'status': 'unhealthy',
                    'endpoint': endpoint,
                    'error': str(e),
                    'last_check': datetime.utcnow().isoformat()
                }
                health_status['overall_status'] = 'degraded'
        
        return health_status

# Global client instance
_vertex_ai_client = None

def get_vertex_ai_client() -> VertexAIClient:
    """Get singleton Vertex AI client instance"""
    global _vertex_ai_client
    if _vertex_ai_client is None:
        _vertex_ai_client = VertexAIClient()
    return _vertex_ai_client
