"""
Image Processing Service - Core component of the Visual Intelligence Engine.
Processes street imagery for crowd detection, activity classification, and accessibility analysis.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from google.cloud import pubsub_v1, storage
from google.cloud.pubsub_v1.types import PubsubMessage

from ...shared.config.settings import settings
from ...shared.utils.logging_utils import get_logger, log_performance, log_image_analysis
from ...shared.utils.error_handling import (
    ImageProcessingError, safe_execute_async, ErrorHandler
)
from ...shared.clients.bigquery_client import get_bigquery_client
from .processors.crowd_detector import CrowdDetector
from .processors.activity_classifier import ActivityClassifier
from .processors.accessibility_analyzer import AccessibilityAnalyzer
from .utils.image_utils import ImageProcessor
from .utils.gcp_utils import GCPImageHandler


logger = get_logger(__name__)


class ImageProcessingService:
    """
    Main image processing service that orchestrates visual intelligence analysis.
    """
    
    def __init__(self):
        """Initialize the image processing service with all components."""
        self.subscriber = pubsub_v1.SubscriberClient()
        self.publisher = pubsub_v1.PublisherClient()
        self.storage_client = storage.Client()
        self.bigquery_client = get_bigquery_client()
        
        # Initialize processors
        self.crowd_detector = CrowdDetector()
        self.activity_classifier = ActivityClassifier()
        self.accessibility_analyzer = AccessibilityAnalyzer()
        self.image_processor = ImageProcessor()
        self.gcp_handler = GCPImageHandler()
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(
            max_workers=settings.image_processing.batch_size
        )
        
        # Subscription paths
        self.subscription_path = settings.pubsub.get_pubsub_subscription_path(
            settings.pubsub.image_processing_subscription
        )
        self.output_topic_path = settings.pubsub.get_pubsub_topic_path(
            settings.pubsub.image_analysis_topic
        )
        
        logger.info(
            "Image processing service initialized",
            extra={
                "subscription_path": self.subscription_path,
                "output_topic_path": self.output_topic_path,
                "gpu_enabled": settings.image_processing.gpu_enabled
            }
        )
    
    @log_performance("image_processing_pipeline")
    async def process_image_message(self, message: PubsubMessage) -> None:
        """
        Process a single image message from Pub/Sub.
        
        Args:
            message: Pub/Sub message containing image processing request
        """
        try:
            # Parse message data
            message_data = json.loads(message.data.decode('utf-8'))
            
            # Validate message structure
            if not self._validate_message(message_data):
                logger.warning(
                    "Invalid message structure, skipping",
                    extra={"message_id": message.message_id}
                )
                message.ack()
                return
            
            # Extract image and location information
            image_ref = message_data.get('image_data', {})
            location_context = message_data.get('geolocation', {})
            event_id = message_data.get('event_id')
            
            logger.info(
                "Processing image message",
                extra={
                    "event_id": event_id,
                    "location_id": location_context.get('location_id'),
                    "message_id": message.message_id
                }
            )
            
            # Process the image
            analysis_results = await self._process_single_image(
                image_ref, location_context, event_id
            )
            
            # Publish results
            await self._publish_analysis_results(
                event_id, image_ref, location_context, analysis_results
            )
            
            # Stream results to BigQuery
            await self._stream_to_bigquery(
                event_id, image_ref, location_context, analysis_results
            )
            
            # Acknowledge message
            message.ack()
            
            logger.info(
                "Image processing completed successfully",
                extra={
                    "event_id": event_id,
                    "message_id": message.message_id
                }
            )
            
        except Exception as e:
            logger.error(
                f"Failed to process image message: {str(e)}",
                extra={
                    "message_id": message.message_id,
                    "error": str(e)
                }
            )
            # Nack the message to retry
            message.nack()
    
    async def _process_single_image(
        self,
        image_ref: Dict[str, Any],
        location_context: Dict[str, Any],
        event_id: str
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete analysis pipeline.
        
        Args:
            image_ref: Image reference information
            location_context: Location context data
            event_id: Event identifier
            
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        
        try:
            # Download and preprocess image
            image_array = await self._download_and_preprocess_image(image_ref)
            
            # Run all analysis components in parallel
            crowd_task = asyncio.create_task(
                self._run_crowd_analysis(image_array, location_context)
            )
            activity_task = asyncio.create_task(
                self._run_activity_analysis(image_array, location_context)
            )
            accessibility_task = asyncio.create_task(
                self._run_accessibility_analysis(image_array, location_context)
            )
            traffic_task = asyncio.create_task(
                self._run_traffic_analysis(image_array, location_context)
            )
            infrastructure_task = asyncio.create_task(
                self._run_infrastructure_analysis(image_array, location_context)
            )
            
            # Wait for all analyses to complete
            crowd_results, activity_results, accessibility_results, \
            traffic_results, infrastructure_results = await asyncio.gather(
                crowd_task, activity_task, accessibility_task,
                traffic_task, infrastructure_task
            )
            
            # Combine results
            analysis_results = {
                "crowd_analysis": crowd_results,
                "activity_classification": activity_results,
                "accessibility_analysis": accessibility_results,
                "traffic_analysis": traffic_results,
                "infrastructure_analysis": infrastructure_results,
                "weather_impact": self._analyze_weather_impact(image_array)
            }
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Add processing metadata
            analysis_results["processing_metadata"] = {
                "processing_start_time": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
                "processing_end_time": datetime.now(timezone.utc).isoformat(),
                "processing_duration_ms": processing_time_ms,
                "model_versions": self._get_model_versions(),
                "confidence_scores": self._calculate_confidence_scores(analysis_results)
            }
            
            # Log analysis completion
            log_image_analysis(
                image_id=event_id,
                analysis_type="complete_street_scene",
                results=analysis_results,
                processing_time_ms=processing_time_ms,
                model_versions=self._get_model_versions()
            )
            
            return analysis_results
            
        except Exception as e:
            raise ImageProcessingError(
                operation="complete_image_analysis",
                message=f"Failed to process image: {str(e)}",
                image_id=event_id,
                cause=e
            )
    
    async def _download_and_preprocess_image(
        self, image_ref: Dict[str, Any]
    ) -> np.ndarray:
        """
        Download image from Cloud Storage and preprocess it.
        
        Args:
            image_ref: Image reference information
            
        Returns:
            Preprocessed image array
        """
        try:
            # Download image
            image_url = image_ref.get('image_url')
            if not image_url:
                raise ImageProcessingError(
                    operation="image_download",
                    message="No image URL provided in image reference"
                )
            
            image_data = await self.gcp_handler.download_image(image_url)
            
            # Preprocess image
            image_array = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.image_processor.preprocess_image,
                image_data
            )
            
            return image_array
            
        except Exception as e:
            raise ImageProcessingError(
                operation="image_preprocessing",
                message=f"Failed to download and preprocess image: {str(e)}",
                cause=e
            )
    
    async def _run_crowd_analysis(
        self, image_array: np.ndarray, location_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run crowd detection and density analysis."""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.crowd_detector.analyze_crowd,
                image_array,
                location_context
            )
        except Exception as e:
            logger.error(f"Crowd analysis failed: {str(e)}")
            return {
                "crowd_count": 0,
                "crowd_density_score": 0.0,
                "age_distribution": {"children": 0.0, "adults": 0.0, "elderly": 0.0},
                "error": str(e)
            }
    
    async def _run_activity_analysis(
        self, image_array: np.ndarray, location_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run activity classification analysis."""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.activity_classifier.classify_activities,
                image_array,
                location_context
            )
        except Exception as e:
            logger.error(f"Activity analysis failed: {str(e)}")
            return []
    
    async def _run_accessibility_analysis(
        self, image_array: np.ndarray, location_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run accessibility analysis."""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.accessibility_analyzer.analyze_accessibility,
                image_array,
                location_context
            )
        except Exception as e:
            logger.error(f"Accessibility analysis failed: {str(e)}")
            return {
                "is_accessible": True,
                "accessibility_score": 0.5,
                "blocking_factors": [],
                "pickup_zones_available": 0,
                "error": str(e)
            }
    
    async def _run_traffic_analysis(
        self, image_array: np.ndarray, location_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run traffic analysis."""
        try:
            # Use crowd detector for vehicle detection as well
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.crowd_detector.analyze_traffic,
                image_array,
                location_context
            )
        except Exception as e:
            logger.error(f"Traffic analysis failed: {str(e)}")
            return {
                "congestion_level": "unknown",
                "vehicle_count": 0,
                "vehicle_types": {"cars": 0, "trucks": 0, "buses": 0, "motorcycles": 0, "bicycles": 0},
                "flow_rate": "unknown",
                "parking_availability": "unknown",
                "error": str(e)
            }
    
    async def _run_infrastructure_analysis(
        self, image_array: np.ndarray, location_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run infrastructure analysis."""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.accessibility_analyzer.analyze_infrastructure,
                image_array,
                location_context
            )
        except Exception as e:
            logger.error(f"Infrastructure analysis failed: {str(e)}")
            return {
                "construction_present": False,
                "road_quality": "unknown",
                "lighting_quality": "unknown",
                "safety_score": 0.5,
                "signage_visibility": "unknown",
                "error": str(e)
            }
    
    def _analyze_weather_impact(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Analyze weather conditions from image."""
        try:
            # Simple weather analysis based on image characteristics
            brightness = np.mean(image_array)
            contrast = np.std(image_array)
            
            # Determine lighting conditions
            if brightness > 150:
                lighting_conditions = "daylight"
            elif brightness > 100:
                lighting_conditions = "twilight"
            elif brightness > 50:
                lighting_conditions = "artificial"
            else:
                lighting_conditions = "dark"
            
            # Estimate visibility based on contrast
            visibility_score = min(1.0, contrast / 50.0)
            
            # Simple precipitation detection (very basic)
            precipitation_visible = contrast < 20 and brightness < 100
            
            return {
                "visibility_score": visibility_score,
                "precipitation_visible": precipitation_visible,
                "lighting_conditions": lighting_conditions,
                "weather_severity": "clear" if visibility_score > 0.7 else "moderate"
            }
            
        except Exception as e:
            logger.error(f"Weather analysis failed: {str(e)}")
            return {
                "visibility_score": 0.5,
                "precipitation_visible": False,
                "lighting_conditions": "unknown",
                "weather_severity": "unknown"
            }
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of all models used."""
        return {
            "crowd_detection_model": self.crowd_detector.get_model_version(),
            "activity_classification_model": self.activity_classifier.get_model_version(),
            "object_detection_model": "yolo_v5_6.0"
        }
    
    def _calculate_confidence_scores(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall confidence scores for analysis components."""
        confidence_scores = {}
        
        # Crowd analysis confidence
        crowd_analysis = analysis_results.get("crowd_analysis", {})
        if "error" not in crowd_analysis:
            confidence_scores["crowd_analysis_confidence"] = 0.85
        else:
            confidence_scores["crowd_analysis_confidence"] = 0.0
        
        # Activity analysis confidence
        activity_analysis = analysis_results.get("activity_classification", [])
        if activity_analysis and len(activity_analysis) > 0:
            avg_confidence = np.mean([act.get("confidence", 0.0) for act in activity_analysis])
            confidence_scores["activity_analysis_confidence"] = avg_confidence
        else:
            confidence_scores["activity_analysis_confidence"] = 0.0
        
        # Overall confidence
        confidence_scores["overall_confidence"] = np.mean(list(confidence_scores.values()))
        
        return confidence_scores
    
    async def _publish_analysis_results(
        self,
        event_id: str,
        image_ref: Dict[str, Any],
        location_context: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> None:
        """Publish analysis results to Pub/Sub topic."""
        try:
            message_data = {
                "analysis_id": f"{event_id}_analysis",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "image_reference": {
                    "image_url": image_ref.get("image_url"),
                    "image_hash": image_ref.get("image_hash"),
                    "original_event_id": event_id
                },
                "location_context": location_context,
                "analysis_results": analysis_results
            }
            
            # Publish message
            future = self.publisher.publish(
                self.output_topic_path,
                json.dumps(message_data).encode('utf-8')
            )
            
            # Wait for publish to complete
            await asyncio.get_event_loop().run_in_executor(
                None, future.result
            )
            
            logger.info(
                "Analysis results published successfully",
                extra={"event_id": event_id, "analysis_id": message_data["analysis_id"]}
            )
            
        except Exception as e:
            logger.error(
                f"Failed to publish analysis results: {str(e)}",
                extra={"event_id": event_id}
            )
            raise
    
    async def _stream_to_bigquery(
        self,
        event_id: str,
        image_ref: Dict[str, Any],
        location_context: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> None:
        """Stream analysis results to BigQuery."""
        try:
            # Prepare row for BigQuery
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "location_id": location_context.get("location_id"),
                "image_ref": json.dumps(image_ref),
                "social_signals": json.dumps({
                    "crowd_density_score": analysis_results.get("crowd_analysis", {}).get("crowd_density_score", 0.0),
                    "accessibility_score": analysis_results.get("accessibility_analysis", {}).get("accessibility_score", 0.5),
                    "event_impact_score": self._calculate_event_impact_score(analysis_results),
                    "traffic_level": analysis_results.get("traffic_analysis", {}).get("congestion_level", "unknown")
                }),
                "event_type": "image_analysis_completed"
            }
            
            # Stream to BigQuery
            await self.bigquery_client.buffered_stream_insert("realtime_events", row)
            
        except Exception as e:
            logger.error(
                f"Failed to stream to BigQuery: {str(e)}",
                extra={"event_id": event_id}
            )
            # Don't raise - this is not critical for the main pipeline
    
    def _calculate_event_impact_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall event impact score from analysis results."""
        try:
            activity_analysis = analysis_results.get("activity_classification", [])
            
            if not activity_analysis:
                return 0.0
            
            # Calculate weighted impact based on activities
            impact_weights = {
                "construction": 0.8,
                "protest": 0.9,
                "event": 0.7,
                "emergency": 1.0,
                "outdoor_dining": 0.3,
                "shopping": 0.2,
                "delivery": 0.1,
                "nightlife": 0.4
            }
            
            total_impact = 0.0
            total_weight = 0.0
            
            for activity in activity_analysis:
                activity_type = activity.get("activity_type", "")
                intensity = activity.get("intensity_score", 0.0)
                confidence = activity.get("confidence", 0.0)
                
                weight = impact_weights.get(activity_type, 0.1)
                impact = (intensity / 10.0) * confidence * weight
                
                total_impact += impact
                total_weight += weight
            
            if total_weight > 0:
                return min(1.0, total_impact / total_weight)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to calculate event impact score: {str(e)}")
            return 0.0
    
    def _validate_message(self, message_data: Dict[str, Any]) -> bool:
        """Validate incoming message structure."""
        required_fields = ["event_id", "event_type", "geolocation"]
        
        for field in required_fields:
            if field not in message_data:
                return False
        
        # Check if it's an image event
        if message_data.get("event_type") != "image_captured":
            return False
        
        # Check for image data
        image_data = message_data.get("image_data", {})
        if not image_data.get("image_url"):
            return False
        
        return True
    
    async def start_processing(self) -> None:
        """Start the image processing service."""
        logger.info("Starting image processing service")
        
        # Configure flow control
        flow_control = pubsub_v1.types.FlowControl(
            max_messages=settings.pubsub.flow_control_max_messages
        )
        
        # Start pulling messages
        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path,
            callback=self.process_image_message,
            flow_control=flow_control
        )
        
        logger.info(f"Listening for messages on {self.subscription_path}")
        
        try:
            # Keep the service running
            await asyncio.get_event_loop().run_in_executor(
                None, streaming_pull_future.result
            )
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            logger.info("Image processing service stopped")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the service."""
        try:
            # Check model loading
            models_healthy = (
                self.crowd_detector.is_healthy() and
                self.activity_classifier.is_healthy() and
                self.accessibility_analyzer.is_healthy()
            )
            
            # Check GCP connections
            gcp_healthy = await self.gcp_handler.health_check()
            
            return {
                "status": "healthy" if models_healthy and gcp_healthy else "unhealthy",
                "models_loaded": models_healthy,
                "gcp_connection": gcp_healthy,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


@asynccontextmanager
async def image_processing_service():
    """Context manager for image processing service."""
    service = ImageProcessingService()
    try:
        yield service
    finally:
        # Cleanup resources
        service.executor.shutdown(wait=True)


async def main():
    """Main entry point for the image processing service."""
    async with image_processing_service() as service:
        await service.start_processing()


if __name__ == "__main__":
    asyncio.run(main())
