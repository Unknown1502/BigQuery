"""
Google Cloud Platform Utilities for Image Processing
Integration utilities for GCP services used in visual intelligence pipeline
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import logging
from dataclasses import dataclass, asdict

from google.cloud import storage
from google.cloud import pubsub_v1
from google.cloud import bigquery
from google.cloud import aiplatform
import numpy as np

from src.shared.utils.logging_utils import get_logger
from src.shared.config.gcp_config import GCPConfig
from src.shared.clients.bigquery_client import BigQueryClient
from src.shared.clients.cloud_storage_client import CloudStorageClient

logger = get_logger(__name__)

@dataclass
class ImageProcessingResult:
    """Standardized result structure for image processing"""
    image_uri: str
    location_id: str
    processing_timestamp: datetime
    crowd_analysis: Dict[str, Any]
    scene_analysis: Dict[str, Any]
    accessibility_analysis: Dict[str, Any]
    processing_time_ms: float
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]

class GCPImageProcessor:
    """
    GCP integration utilities for image processing pipeline
    """
    
    def __init__(self, config: GCPConfig):
        self.config = config
        self.storage_client = CloudStorageClient(config)
        self.bigquery_client = BigQueryClient(config)
        self.publisher = pubsub_v1.PublisherClient()
        
        # Initialize Vertex AI
        aiplatform.init(
            project=config.project_id,
            location=config.region
        )
        
    async def process_image_from_gcs(
        self, 
        image_uri: str, 
        location_id: str,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> ImageProcessingResult:
        """Process image from Google Cloud Storage"""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Download image from GCS
            image_data = await self._download_image_async(image_uri)
            
            # Process image through ML pipeline
            crowd_result = await self._analyze_crowd_async(image_data)
            scene_result = await self._analyze_scene_async(image_data)
            accessibility_result = await self._analyze_accessibility_async(image_data)
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Compile results
            result = ImageProcessingResult(
                image_uri=image_uri,
                location_id=location_id,
                processing_timestamp=start_time,
                crowd_analysis=crowd_result,
                scene_analysis=scene_result,
                accessibility_analysis=accessibility_result,
                processing_time_ms=processing_time,
                confidence_scores={
                    'crowd_confidence': crowd_result.get('confidence_score', 0.0),
                    'scene_confidence': scene_result.get('confidence_score', 0.0),
                    'accessibility_confidence': accessibility_result.get('confidence_score', 0.0)
                },
                metadata={
                    'processing_version': '1.0',
                    'model_versions': {
                        'crowd_detection': 'v2.1',
                        'scene_classification': 'v1.8',
                        'accessibility_analysis': 'v1.3'
                    }
                }
            )
            
            # Store results in BigQuery
            await self._store_results_async(result)
            
            # Publish to Pub/Sub for downstream processing
            await self._publish_results_async(result)
            
            logger.info(f"Successfully processed image {image_uri} in {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_uri}: {e}")
            raise
    
    async def _download_image_async(self, image_uri: str) -> bytes:
        """Download image from GCS asynchronously"""
        try:
            # Parse GCS URI
            if not image_uri.startswith('gs://'):
                raise ValueError(f"Invalid GCS URI: {image_uri}")
            
            bucket_name = image_uri.split('/')[2]
            blob_name = '/'.join(image_uri.split('/')[3:])
            
            # Download image data
            image_data = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.storage_client.download_blob_as_bytes,
                bucket_name,
                blob_name
            )
            
            return image_data
            
        except Exception as e:
            logger.error(f"Failed to download image {image_uri}: {e}")
            raise
    
    async def _analyze_crowd_async(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze crowd using Vertex AI endpoint"""
        try:
            # Prepare request for Vertex AI
            instances = [{
                'image': {
                    'bytesBase64Encoded': self._encode_image_base64(image_data)
                }
            }]
            
            # Call Vertex AI endpoint
            endpoint = aiplatform.Endpoint(self.config.crowd_detection_endpoint)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                endpoint.predict,
                instances
            )
            
            # Parse response
            predictions = response.predictions[0] if response.predictions else {}
            
            return {
                'crowd_count': predictions.get('crowd_count', 0),
                'crowd_density': predictions.get('crowd_density', 0.0),
                'confidence_score': predictions.get('confidence', 0.0),
                'bounding_boxes': predictions.get('bounding_boxes', []),
                'density_heatmap_uri': predictions.get('density_heatmap_uri', '')
            }
            
        except Exception as e:
            logger.error(f"Error in crowd analysis: {e}")
            return {
                'crowd_count': 0,
                'crowd_density': 0.0,
                'confidence_score': 0.0,
                'bounding_boxes': [],
                'error': str(e)
            }
    
    async def _analyze_scene_async(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze scene using Vertex AI endpoint"""
        try:
            # Prepare request for Vertex AI
            instances = [{
                'image': {
                    'bytesBase64Encoded': self._encode_image_base64(image_data)
                }
            }]
            
            # Call Vertex AI endpoint
            endpoint = aiplatform.Endpoint(self.config.scene_classification_endpoint)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                endpoint.predict,
                instances
            )
            
            # Parse response
            predictions = response.predictions[0] if response.predictions else {}
            
            return {
                'scene_type': predictions.get('scene_type', 'unknown'),
                'confidence_score': predictions.get('confidence', 0.0),
                'scene_probabilities': predictions.get('scene_probabilities', {}),
                'detected_objects': predictions.get('detected_objects', []),
                'business_indicators': predictions.get('business_indicators', {}),
                'residential_indicators': predictions.get('residential_indicators', {}),
                'entertainment_indicators': predictions.get('entertainment_indicators', {}),
                'transport_indicators': predictions.get('transport_indicators', {}),
                'time_context': predictions.get('time_context', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error in scene analysis: {e}")
            return {
                'scene_type': 'unknown',
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    async def _analyze_accessibility_async(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze accessibility using Vertex AI endpoint"""
        try:
            # Prepare request for Vertex AI
            instances = [{
                'image': {
                    'bytesBase64Encoded': self._encode_image_base64(image_data)
                }
            }]
            
            # Call Vertex AI endpoint
            endpoint = aiplatform.Endpoint(self.config.accessibility_analysis_endpoint)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                endpoint.predict,
                instances
            )
            
            # Parse response
            predictions = response.predictions[0] if response.predictions else {}
            
            return {
                'accessibility_score': predictions.get('accessibility_score', 0.0),
                'pickup_feasibility': predictions.get('pickup_feasibility', 0.0),
                'dropoff_feasibility': predictions.get('dropoff_feasibility', 0.0),
                'confidence_score': predictions.get('confidence', 0.0),
                'obstacles': predictions.get('obstacles', []),
                'infrastructure_quality': predictions.get('infrastructure_quality', 0.0),
                'safety_score': predictions.get('safety_score', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error in accessibility analysis: {e}")
            return {
                'accessibility_score': 0.0,
                'pickup_feasibility': 0.0,
                'dropoff_feasibility': 0.0,
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    def _encode_image_base64(self, image_data: bytes) -> str:
        """Encode image data to base64 for Vertex AI"""
        import base64
        return base64.b64encode(image_data).decode('utf-8')
    
    async def _store_results_async(self, result: ImageProcessingResult) -> None:
        """Store processing results in BigQuery"""
        try:
            # Prepare row for BigQuery
            row = {
                'analysis_id': f"{result.location_id}_{int(result.processing_timestamp.timestamp())}",
                'image_uri': result.image_uri,
                'location_id': result.location_id,
                'analysis_timestamp': result.processing_timestamp.isoformat(),
                
                # Crowd analysis
                'crowd_count': result.crowd_analysis.get('crowd_count', 0),
                'crowd_density_per_sqm': result.crowd_analysis.get('crowd_density', 0.0),
                'crowd_distribution': json.dumps(result.crowd_analysis.get('bounding_boxes', [])),
                
                # Activity analysis
                'dominant_activity': result.scene_analysis.get('scene_type', 'unknown'),
                'activity_scores': json.dumps(result.scene_analysis.get('scene_probabilities', {})),
                'activity_diversity_score': self._calculate_diversity_score(
                    result.scene_analysis.get('scene_probabilities', {})
                ),
                
                # Infrastructure analysis
                'accessibility_score': result.accessibility_analysis.get('accessibility_score', 0.0),
                'infrastructure_quality': result.accessibility_analysis.get('infrastructure_quality', 0.0),
                'safety_score': result.accessibility_analysis.get('safety_score', 0.0),
                
                # Processing metadata
                'processing_time_ms': result.processing_time_ms,
                'model_version': result.metadata.get('processing_version', '1.0'),
                'confidence_score': np.mean(list(result.confidence_scores.values())),
                
                # Audit fields
                'created_at': datetime.now(timezone.utc).isoformat(),
                'created_by': 'image_processor_service'
            }
            
            # Insert into BigQuery
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.bigquery_client.insert_rows,
                'ride_intelligence.image_analysis_results',
                [row]
            )
            
            logger.debug(f"Stored results for image {result.image_uri} in BigQuery")
            
        except Exception as e:
            logger.error(f"Failed to store results in BigQuery: {e}")
            # Don't raise - this shouldn't fail the entire processing
    
    async def _publish_results_async(self, result: ImageProcessingResult) -> None:
        """Publish processing results to Pub/Sub"""
        try:
            # Prepare message
            message_data = {
                'image_uri': result.image_uri,
                'location_id': result.location_id,
                'processing_timestamp': result.processing_timestamp.isoformat(),
                'crowd_count': result.crowd_analysis.get('crowd_count', 0),
                'scene_type': result.scene_analysis.get('scene_type', 'unknown'),
                'accessibility_score': result.accessibility_analysis.get('accessibility_score', 0.0),
                'confidence_scores': result.confidence_scores,
                'processing_time_ms': result.processing_time_ms
            }
            
            # Publish to image processing results topic
            topic_path = self.publisher.topic_path(
                self.config.project_id, 
                'image-processing-results'
            )
            
            message_json = json.dumps(message_data).encode('utf-8')
            
            # Publish message
            future = self.publisher.publish(
                topic_path, 
                message_json,
                location_id=result.location_id,
                processing_timestamp=result.processing_timestamp.isoformat()
            )
            
            # Wait for publish to complete
            await asyncio.get_event_loop().run_in_executor(None, future.result)
            
            logger.debug(f"Published results for image {result.image_uri} to Pub/Sub")
            
        except Exception as e:
            logger.error(f"Failed to publish results to Pub/Sub: {e}")
            # Don't raise - this shouldn't fail the entire processing
    
    def _calculate_diversity_score(self, probabilities: Dict[str, float]) -> float:
        """Calculate diversity score from probability distribution"""
        if not probabilities:
            return 0.0
        
        # Calculate entropy as diversity measure
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(probabilities)) if len(probabilities) > 1 else 1.0
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    async def batch_process_images(
        self, 
        image_uris: List[str], 
        location_ids: List[str],
        max_concurrent: int = 10
    ) -> List[ImageProcessingResult]:
        """Process multiple images concurrently"""
        
        if len(image_uris) != len(location_ids):
            raise ValueError("Number of image URIs must match number of location IDs")
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(image_uri: str, location_id: str):
            async with semaphore:
                return await self.process_image_from_gcs(image_uri, location_id)
        
        # Process all images concurrently
        tasks = [
            process_with_semaphore(uri, loc_id) 
            for uri, loc_id in zip(image_uris, location_ids)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process image {image_uris[i]}: {result}")
            else:
                successful_results.append(result)
        
        logger.info(f"Successfully processed {len(successful_results)}/{len(image_uris)} images")
        
        return successful_results
    
    async def cleanup_old_processing_data(self, retention_days: int = 30) -> None:
        """Clean up old processing data from GCS and BigQuery"""
        try:
            cutoff_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=retention_days)
            
            # Clean up BigQuery data
            cleanup_query = f"""
            DELETE FROM `{self.config.project_id}.ride_intelligence.image_analysis_results`
            WHERE analysis_timestamp < '{cutoff_date.isoformat()}'
            """
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.bigquery_client.execute_query,
                cleanup_query
            )
            
            # Clean up temporary GCS files (if any)
            temp_bucket = f"{self.config.project_id}-temp-processing"
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.storage_client.cleanup_old_blobs,
                temp_bucket,
                retention_days
            )
            
            logger.info(f"Cleaned up processing data older than {retention_days} days")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing performance metrics"""
        try:
            # Query BigQuery for recent processing metrics
            metrics_query = f"""
            SELECT 
                COUNT(*) as total_processed,
                AVG(processing_time_ms) as avg_processing_time,
                AVG(confidence_score) as avg_confidence,
                AVG(crowd_count) as avg_crowd_count,
                COUNT(DISTINCT location_id) as unique_locations
            FROM `{self.config.project_id}.ride_intelligence.image_analysis_results`
            WHERE analysis_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
            """
            
            results = self.bigquery_client.execute_query(metrics_query)
            
            if results:
                return dict(results[0])
            else:
                return {
                    'total_processed': 0,
                    'avg_processing_time': 0.0,
                    'avg_confidence': 0.0,
                    'avg_crowd_count': 0.0,
                    'unique_locations': 0
                }
                
        except Exception as e:
            logger.error(f"Error getting processing metrics: {e}")
            return {}
