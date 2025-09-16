"""
Data Ingestion Pipeline - Apache Beam pipeline for real-time data processing.
Handles multimodal data streams including images, weather, traffic, and competitor data.
"""

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from apache_beam.transforms import window
from apache_beam.io import ReadFromPubSub, WriteToBigQuery
from apache_beam.io.gcp.pubsub import WriteStringsToPubSub
import apache_beam.transforms.combiners as combiners

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.shared.config.settings_fixed import settings
from src.shared.utils.logging_utils import get_logger
from src.shared.utils.error_handling import DataIngestionError, ValidationError
from src.shared.clients.bigquery_client import get_bigquery_client
from .connectors.street_camera_connector import StreetCameraConnector
from .connectors.weather_connector import WeatherConnector
from .connectors.traffic_connector import TrafficConnector
from .connectors.competitor_connector import CompetitorConnector
from .transformers.image_transformer import ImageTransformer
from .transformers.weather_transformer import WeatherTransformer
from .transformers.location_transformer import LocationTransformer


logger = get_logger(__name__)


class DataIngestionPipeline:
    """
    Main data ingestion pipeline orchestrator.
    
    Processes multiple data streams in real-time:
    - Street camera imagery for visual intelligence
    - Weather data for environmental context
    - Traffic data for congestion analysis
    - Competitor pricing for market intelligence
    """
    
    def __init__(self, pipeline_options: PipelineOptions):
        self.pipeline_options = pipeline_options
        try:
            self.bigquery_client = get_bigquery_client()
        except TypeError:
            # Handle case where get_bigquery_client doesn't accept parameters
            from src.shared.clients.bigquery_client import BigQueryClient
            self.bigquery_client = BigQueryClient()
        
        # Initialize connectors
        self.street_camera_connector = StreetCameraConnector()
        self.weather_connector = WeatherConnector()
        self.traffic_connector = TrafficConnector()
        self.competitor_connector = CompetitorConnector()
        
        # Initialize transformers
        self.image_transformer = ImageTransformer()
        self.weather_transformer = WeatherTransformer()
        self.location_transformer = LocationTransformer()
        
        logger.info("Data ingestion pipeline initialized")
    
    def create_pipeline(self) -> beam.Pipeline:
        """Create and configure the Apache Beam pipeline."""
        pipeline = beam.Pipeline(options=self.pipeline_options)
        
        # Define Pub/Sub topics
        street_imagery_topic = f"projects/{settings.gcp.project_id}/topics/street-imagery"
        weather_topic = f"projects/{settings.gcp.project_id}/topics/weather-data"
        traffic_topic = f"projects/{settings.gcp.project_id}/topics/traffic-data"
        competitor_topic = f"projects/{settings.gcp.project_id}/topics/competitor-pricing"
        
        # Process street imagery stream
        street_imagery_stream = (
            pipeline
            | "Read Street Imagery" >> ReadFromPubSub(topic=street_imagery_topic)
            | "Parse Street Imagery" >> beam.Map(self._parse_street_imagery_message)
            | "Transform Street Imagery" >> beam.ParDo(StreetImageryTransform())
            | "Window Street Imagery" >> beam.WindowInto(
                window.FixedWindows(60)  # 1-minute windows
            )
            | "Aggregate Street Analysis" >> beam.CombinePerKey(
                combiners.MeanCombineFn()
            )
            | "Format Street Results" >> beam.Map(self._format_street_analysis_result)
        )
        
        # Process weather data stream
        weather_stream = (
            pipeline
            | "Read Weather Data" >> ReadFromPubSub(topic=weather_topic)
            | "Parse Weather Data" >> beam.Map(self._parse_weather_message)
            | "Transform Weather Data" >> beam.ParDo(WeatherTransform())
            | "Window Weather Data" >> beam.WindowInto(window.FixedWindows(300))  # 5-minute windows
            | "Aggregate Weather Data" >> beam.CombinePerKey(WeatherAggregator())
            | "Format Weather Results" >> beam.Map(self._format_weather_result)
        )
        
        # Process traffic data stream
        traffic_stream = (
            pipeline
            | "Read Traffic Data" >> ReadFromPubSub(topic=traffic_topic)
            | "Parse Traffic Data" >> beam.Map(self._parse_traffic_message)
            | "Transform Traffic Data" >> beam.ParDo(TrafficTransform())
            | "Window Traffic Data" >> beam.WindowInto(window.FixedWindows(120))  # 2-minute windows
            | "Aggregate Traffic Data" >> beam.CombinePerKey(TrafficAggregator())
            | "Format Traffic Results" >> beam.Map(self._format_traffic_result)
        )
        
        # Process competitor pricing stream
        competitor_stream = (
            pipeline
            | "Read Competitor Data" >> ReadFromPubSub(topic=competitor_topic)
            | "Parse Competitor Data" >> beam.Map(self._parse_competitor_message)
            | "Transform Competitor Data" >> beam.ParDo(CompetitorTransform())
            | "Window Competitor Data" >> beam.WindowInto(window.FixedWindows(600))  # 10-minute windows
            | "Aggregate Competitor Data" >> beam.CombinePerKey(CompetitorAggregator())
            | "Format Competitor Results" >> beam.Map(self._format_competitor_result)
        )
        
        # Combine all streams for unified processing
        unified_stream = (
            (street_imagery_stream, weather_stream, traffic_stream, competitor_stream)
            | "Flatten Streams" >> beam.Flatten()
            | "Add Processing Timestamp" >> beam.Map(self._add_processing_timestamp)
            | "Enrich with Location Data" >> beam.ParDo(LocationEnrichmentTransform())
            | "Validate Data Quality" >> beam.ParDo(DataQualityTransform())
        )
        
        # Write to BigQuery for analytics
        unified_stream | "Write to BigQuery" >> WriteToBigQuery(
            table=f"{settings.gcp.project_id}:{settings.database.bigquery_dataset}.realtime_events",
            schema=self._get_bigquery_schema(),
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            method='STREAMING_INSERTS'
        )
        
        # Trigger real-time pricing calculations
        pricing_triggers = (
            unified_stream
            | "Filter Pricing Triggers" >> beam.Filter(self._should_trigger_pricing)
            | "Create Pricing Messages" >> beam.Map(self._create_pricing_message)
            | "Publish Pricing Triggers" >> WriteStringsToPubSub(
                f"projects/{settings.gcp.project_id}/topics/pricing-triggers"
            )
        )
        
        return pipeline
    
    def _parse_street_imagery_message(self, message: bytes) -> Dict[str, Any]:
        """Parse street imagery Pub/Sub message."""
        try:
            data = json.loads(message.decode('utf-8'))
            return {
                'message_type': 'street_imagery',
                'location_id': data['location_id'],
                'image_url': data['image_url'],
                'timestamp': data['timestamp'],
                'camera_id': data.get('camera_id'),
                'metadata': data.get('metadata', {})
            }
        except Exception as e:
            logger.error(f"Failed to parse street imagery message: {str(e)}")
            raise DataIngestionError(f"Invalid street imagery message format: {str(e)}")
    
    def _parse_weather_message(self, message: bytes) -> Dict[str, Any]:
        """Parse weather data Pub/Sub message."""
        try:
            data = json.loads(message.decode('utf-8'))
            return {
                'message_type': 'weather',
                'location_id': data['location_id'],
                'temperature': data['temperature'],
                'humidity': data['humidity'],
                'precipitation': data.get('precipitation', 0),
                'wind_speed': data.get('wind_speed', 0),
                'visibility': data.get('visibility', 10),
                'timestamp': data['timestamp']
            }
        except Exception as e:
            logger.error(f"Failed to parse weather message: {str(e)}")
            raise DataIngestionError(f"Invalid weather message format: {str(e)}")
    
    def _parse_traffic_message(self, message: bytes) -> Dict[str, Any]:
        """Parse traffic data Pub/Sub message."""
        try:
            data = json.loads(message.decode('utf-8'))
            return {
                'message_type': 'traffic',
                'location_id': data['location_id'],
                'congestion_level': data['congestion_level'],
                'average_speed': data['average_speed'],
                'incident_count': data.get('incident_count', 0),
                'road_closures': data.get('road_closures', []),
                'timestamp': data['timestamp']
            }
        except Exception as e:
            logger.error(f"Failed to parse traffic message: {str(e)}")
            raise DataIngestionError(f"Invalid traffic message format: {str(e)}")
    
    def _parse_competitor_message(self, message: bytes) -> Dict[str, Any]:
        """Parse competitor pricing Pub/Sub message."""
        try:
            data = json.loads(message.decode('utf-8'))
            return {
                'message_type': 'competitor',
                'location_id': data['location_id'],
                'competitor_name': data['competitor_name'],
                'base_price': data['base_price'],
                'surge_multiplier': data.get('surge_multiplier', 1.0),
                'estimated_wait_time': data.get('estimated_wait_time'),
                'timestamp': data['timestamp']
            }
        except Exception as e:
            logger.error(f"Failed to parse competitor message: {str(e)}")
            raise DataIngestionError(f"Invalid competitor message format: {str(e)}")
    
    def _format_street_analysis_result(self, kv_pair: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Format street analysis aggregation result."""
        location_id, analysis_data = kv_pair
        return {
            'event_type': 'street_analysis',
            'location_id': location_id,
            'crowd_density': analysis_data.get('avg_crowd_count', 0),
            'accessibility_score': analysis_data.get('avg_accessibility_score', 1.0),
            'activity_level': analysis_data.get('avg_activity_level', 0.5),
            'visual_factors': analysis_data.get('visual_factors', []),
            'confidence_score': analysis_data.get('avg_confidence', 0.8),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _format_weather_result(self, kv_pair: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Format weather aggregation result."""
        location_id, weather_data = kv_pair
        return {
            'event_type': 'weather_update',
            'location_id': location_id,
            'temperature': weather_data.get('avg_temperature'),
            'humidity': weather_data.get('avg_humidity'),
            'precipitation': weather_data.get('total_precipitation', 0),
            'weather_impact_score': weather_data.get('impact_score', 1.0),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _format_traffic_result(self, kv_pair: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Format traffic aggregation result."""
        location_id, traffic_data = kv_pair
        return {
            'event_type': 'traffic_update',
            'location_id': location_id,
            'congestion_level': traffic_data.get('avg_congestion_level'),
            'average_speed': traffic_data.get('avg_speed'),
            'incident_count': traffic_data.get('total_incidents', 0),
            'traffic_impact_score': traffic_data.get('impact_score', 1.0),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _format_competitor_result(self, kv_pair: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Format competitor pricing aggregation result."""
        location_id, competitor_data = kv_pair
        return {
            'event_type': 'competitor_update',
            'location_id': location_id,
            'avg_competitor_price': competitor_data.get('avg_price'),
            'min_competitor_price': competitor_data.get('min_price'),
            'max_competitor_price': competitor_data.get('max_price'),
            'competitor_count': competitor_data.get('competitor_count', 0),
            'market_position': competitor_data.get('market_position', 'unknown'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _add_processing_timestamp(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing timestamp to element."""
        element['processing_timestamp'] = datetime.now(timezone.utc).isoformat()
        element['processing_latency_ms'] = (
            datetime.now(timezone.utc) - 
            datetime.fromisoformat(element['timestamp'].replace('Z', '+00:00'))
        ).total_seconds() * 1000
        return element
    
    def _should_trigger_pricing(self, element: Dict[str, Any]) -> bool:
        """Determine if element should trigger pricing recalculation."""
        # Trigger pricing for significant changes
        trigger_conditions = [
            element.get('event_type') == 'street_analysis' and element.get('crowd_density', 0) > 20,
            element.get('event_type') == 'traffic_update' and element.get('congestion_level', 0) > 0.7,
            element.get('event_type') == 'weather_update' and element.get('weather_impact_score', 1.0) != 1.0,
            element.get('event_type') == 'competitor_update'
        ]
        return any(trigger_conditions)
    
    def _create_pricing_message(self, element: Dict[str, Any]) -> str:
        """Create pricing trigger message."""
        pricing_message = {
            'location_id': element['location_id'],
            'trigger_type': element['event_type'],
            'trigger_data': element,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'priority': self._calculate_priority(element)
        }
        return json.dumps(pricing_message)
    
    def _calculate_priority(self, element: Dict[str, Any]) -> str:
        """Calculate priority for pricing trigger."""
        if element.get('event_type') == 'street_analysis':
            crowd_density = element.get('crowd_density', 0)
            if crowd_density > 50:
                return 'high'
            elif crowd_density > 20:
                return 'medium'
        elif element.get('event_type') == 'traffic_update':
            congestion = element.get('congestion_level', 0)
            if congestion > 0.8:
                return 'high'
            elif congestion > 0.5:
                return 'medium'
        elif element.get('event_type') == 'competitor_update':
            return 'medium'
        
        return 'low'
    
    def _get_bigquery_schema(self) -> str:
        """Get BigQuery schema for realtime_events table."""
        return """
        [
            {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
            {"name": "processing_timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
            {"name": "processing_latency_ms", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "event_type", "type": "STRING", "mode": "REQUIRED"},
            {"name": "location_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "crowd_density", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "accessibility_score", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "activity_level", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "visual_factors", "type": "STRING", "mode": "REPEATED"},
            {"name": "confidence_score", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "temperature", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "humidity", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "precipitation", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "weather_impact_score", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "congestion_level", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "average_speed", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "incident_count", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "traffic_impact_score", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "avg_competitor_price", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "min_competitor_price", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "max_competitor_price", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "competitor_count", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "market_position", "type": "STRING", "mode": "NULLABLE"}
        ]
        """


class StreetImageryTransform(beam.DoFn):
    """Transform street imagery data for analysis."""
    
    def __init__(self):
        self.image_transformer = None
    
    def setup(self):
        """Initialize image transformer."""
        self.image_transformer = ImageTransformer()
    
    def process(self, element: Dict[str, Any]):
        """Process street imagery element."""
        try:
            # Analyze image using computer vision
            if self.image_transformer and hasattr(self.image_transformer, 'analyze_street_scene'):
                analysis_result = self.image_transformer.analyze_street_scene(
                    element['image_url'],
                    element['location_id']
                )
            else:
                analysis_result = {}
            
            yield (element['location_id'], {
                'crowd_count': analysis_result.get('crowd_count', 0),
                'accessibility_score': analysis_result.get('accessibility_score', 1.0),
                'activity_level': analysis_result.get('activity_level', 0.5),
                'visual_factors': analysis_result.get('visual_factors', []),
                'confidence': analysis_result.get('confidence', 0.8),
                'timestamp': element['timestamp']
            })
            
        except Exception as e:
            logger.error(f"Street imagery transform failed: {str(e)}")
            # Yield default values on error
            yield (element['location_id'], {
                'crowd_count': 0,
                'accessibility_score': 1.0,
                'activity_level': 0.5,
                'visual_factors': [],
                'confidence': 0.0,
                'timestamp': element['timestamp']
            })


class WeatherTransform(beam.DoFn):
    """Transform weather data."""
    
    def __init__(self):
        self.weather_transformer = None
    
    def setup(self):
        """Initialize weather transformer."""
        self.weather_transformer = WeatherTransformer()
    
    def process(self, element: Dict[str, Any]):
        """Process weather element."""
        try:
            # Calculate weather impact on demand
            if self.weather_transformer and hasattr(self.weather_transformer, 'calculate_impact_score'):
                impact_score = self.weather_transformer.calculate_impact_score(
                    temperature=element['temperature'],
                    precipitation=element['precipitation'],
                    wind_speed=element.get('wind_speed', 0)
                )
            else:
                impact_score = 1.0
            
            yield (element['location_id'], {
                'temperature': element['temperature'],
                'humidity': element['humidity'],
                'precipitation': element['precipitation'],
                'impact_score': impact_score,
                'timestamp': element['timestamp']
            })
            
        except Exception as e:
            logger.error(f"Weather transform failed: {str(e)}")
            yield (element['location_id'], {
                'temperature': element.get('temperature', 20.0),
                'humidity': element.get('humidity', 50.0),
                'precipitation': element.get('precipitation', 0.0),
                'impact_score': 1.0,
                'timestamp': element['timestamp']
            })


class TrafficTransform(beam.DoFn):
    """Transform traffic data."""
    
    def process(self, element: Dict[str, Any]):
        """Process traffic element."""
        try:
            # Calculate traffic impact score
            congestion = element['congestion_level']
            incidents = element.get('incident_count', 0)
            
            impact_score = min(2.0, 1.0 + (congestion * 0.5) + (incidents * 0.1))
            
            yield (element['location_id'], {
                'congestion_level': congestion,
                'average_speed': element['average_speed'],
                'incident_count': incidents,
                'impact_score': impact_score,
                'timestamp': element['timestamp']
            })
            
        except Exception as e:
            logger.error(f"Traffic transform failed: {str(e)}")
            yield (element['location_id'], {
                'congestion_level': 0.0,
                'average_speed': 50.0,
                'incident_count': 0,
                'impact_score': 1.0,
                'timestamp': element['timestamp']
            })


class CompetitorTransform(beam.DoFn):
    """Transform competitor pricing data."""
    
    def process(self, element: Dict[str, Any]):
        """Process competitor element."""
        try:
            total_price = element['base_price'] * element.get('surge_multiplier', 1.0)
            
            yield (element['location_id'], {
                'competitor_name': element['competitor_name'],
                'total_price': total_price,
                'base_price': element['base_price'],
                'surge_multiplier': element.get('surge_multiplier', 1.0),
                'timestamp': element['timestamp']
            })
            
        except Exception as e:
            logger.error(f"Competitor transform failed: {str(e)}")


class WeatherAggregator(beam.CombineFn):
    """Aggregate weather data over time windows."""
    
    def create_accumulator(self):
        return {'temperatures': [], 'humidities': [], 'precipitations': [], 'impacts': []}
    
    def add_input(self, accumulator, input_element):
        _, weather_data = input_element
        accumulator['temperatures'].append(weather_data['temperature'])
        accumulator['humidities'].append(weather_data['humidity'])
        accumulator['precipitations'].append(weather_data['precipitation'])
        accumulator['impacts'].append(weather_data['impact_score'])
        return accumulator
    
    def merge_accumulators(self, accumulators):
        merged = {'temperatures': [], 'humidities': [], 'precipitations': [], 'impacts': []}
        for acc in accumulators:
            merged['temperatures'].extend(acc['temperatures'])
            merged['humidities'].extend(acc['humidities'])
            merged['precipitations'].extend(acc['precipitations'])
            merged['impacts'].extend(acc['impacts'])
        return merged
    
    def extract_output(self, accumulator):
        if not accumulator['temperatures']:
            return {}
        
        return {
            'avg_temperature': sum(accumulator['temperatures']) / len(accumulator['temperatures']),
            'avg_humidity': sum(accumulator['humidities']) / len(accumulator['humidities']),
            'total_precipitation': sum(accumulator['precipitations']),
            'impact_score': sum(accumulator['impacts']) / len(accumulator['impacts'])
        }


class TrafficAggregator(beam.CombineFn):
    """Aggregate traffic data over time windows."""
    
    def create_accumulator(self):
        return {'congestions': [], 'speeds': [], 'incidents': [], 'impacts': []}
    
    def add_input(self, accumulator, input_element):
        _, traffic_data = input_element
        accumulator['congestions'].append(traffic_data['congestion_level'])
        accumulator['speeds'].append(traffic_data['average_speed'])
        accumulator['incidents'].append(traffic_data['incident_count'])
        accumulator['impacts'].append(traffic_data['impact_score'])
        return accumulator
    
    def merge_accumulators(self, accumulators):
        merged = {'congestions': [], 'speeds': [], 'incidents': [], 'impacts': []}
        for acc in accumulators:
            merged['congestions'].extend(acc['congestions'])
            merged['speeds'].extend(acc['speeds'])
            merged['incidents'].extend(acc['incidents'])
            merged['impacts'].extend(acc['impacts'])
        return merged
    
    def extract_output(self, accumulator):
        if not accumulator['congestions']:
            return {}
        
        return {
            'avg_congestion_level': sum(accumulator['congestions']) / len(accumulator['congestions']),
            'avg_speed': sum(accumulator['speeds']) / len(accumulator['speeds']),
            'total_incidents': sum(accumulator['incidents']),
            'impact_score': sum(accumulator['impacts']) / len(accumulator['impacts'])
        }


class CompetitorAggregator(beam.CombineFn):
    """Aggregate competitor pricing data over time windows."""
    
    def create_accumulator(self):
        return {'prices': [], 'competitors': set()}
    
    def add_input(self, accumulator, input_element):
        _, competitor_data = input_element
        accumulator['prices'].append(competitor_data['total_price'])
        accumulator['competitors'].add(competitor_data['competitor_name'])
        return accumulator
    
    def merge_accumulators(self, accumulators):
        merged = {'prices': [], 'competitors': set()}
        for acc in accumulators:
            merged['prices'].extend(acc['prices'])
            merged['competitors'].update(acc['competitors'])
        return merged
    
    def extract_output(self, accumulator):
        if not accumulator['prices']:
            return {}
        
        prices = accumulator['prices']
        return {
            'avg_price': sum(prices) / len(prices),
            'min_price': min(prices),
            'max_price': max(prices),
            'competitor_count': len(accumulator['competitors']),
            'market_position': 'competitive' if len(prices) > 2 else 'limited'
        }


class LocationEnrichmentTransform(beam.DoFn):
    """Enrich data with location metadata."""
    
    def __init__(self):
        self.location_transformer = None
    
    def setup(self):
        """Initialize location transformer."""
        self.location_transformer = LocationTransformer()
    
    def process(self, element: Dict[str, Any]):
        """Enrich element with location data."""
        try:
            # Add location metadata
            if self.location_transformer and hasattr(self.location_transformer, 'get_location_metadata'):
                location_data = self.location_transformer.get_location_metadata(
                    element['location_id']
                )
            else:
                location_data = {}
            
            element.update({
                'location_type': location_data.get('type', 'unknown'),
                'business_district': location_data.get('business_district', False),
                'population_density': location_data.get('population_density', 'medium'),
                'transportation_hubs': location_data.get('transportation_hubs', [])
            })
            
            yield element
            
        except Exception as e:
            logger.error(f"Location enrichment failed: {str(e)}")
            yield element


class DataQualityTransform(beam.DoFn):
    """Validate and ensure data quality."""
    
    def process(self, element: Dict[str, Any]):
        """Validate data quality and filter invalid records."""
        try:
            # Basic validation
            if not element.get('location_id'):
                logger.warning("Dropping record without location_id")
                return
            
            if not element.get('timestamp'):
                logger.warning("Dropping record without timestamp")
                return
            
            # Validate timestamp is recent (within last hour)
            try:
                event_time = datetime.fromisoformat(element['timestamp'].replace('Z', '+00:00'))
                time_diff = datetime.now(timezone.utc) - event_time
                if time_diff.total_seconds() > 3600:  # 1 hour
                    logger.warning(f"Dropping old record: {time_diff.total_seconds()}s old")
                    return
            except Exception:
                logger.warning("Invalid timestamp format")
                return
            
            # Add data quality score
            element['data_quality_score'] = self._calculate_quality_score(element)
            
            yield element
            
        except Exception as e:
            logger.error(f"Data quality validation failed: {str(e)}")
    
    def _calculate_quality_score(self, element: Dict[str, Any]) -> float:
        """Calculate data quality score (0-1)."""
        score = 1.0
        
        # Penalize missing optional fields
        optional_fields = ['confidence_score', 'processing_latency_ms']
        for field in optional_fields:
            if field not in element:
                score -= 0.1
        
        # Penalize high processing latency
        latency = element.get('processing_latency_ms', 0)
        if latency > 5000:  # 5 seconds
            score -= 0.2
        elif latency > 1000:  # 1 second
            score -= 0.1
        
        return max(0.0, score)


def run_pipeline():
    """Run the data ingestion pipeline."""
    # Configure pipeline options
    pipeline_options = PipelineOptions([
        '--project=' + settings.gcp.project_id,
        '--region=' + settings.gcp.region,
        '--runner=DataflowRunner',
        '--temp_location=gs://' + getattr(settings.gcp, 'temp_bucket', f'{settings.gcp.project_id}-temp') + '/temp',
        '--staging_location=gs://' + getattr(settings.gcp, 'temp_bucket', f'{settings.gcp.project_id}-temp') + '/staging',
        '--job_name=dynamic-pricing-data-ingestion',
        '--max_num_workers=10',
        '--autoscaling_algorithm=THROUGHPUT_BASED',
        '--enable_streaming_engine',
        '--use_public_ips=false'
    ])
    
    # Set streaming mode
    pipeline_options.view_as(StandardOptions).streaming = True
    
    # Create and run pipeline
    ingestion_pipeline = DataIngestionPipeline(pipeline_options)
    pipeline = ingestion_pipeline.create_pipeline()
    
    logger.info("Starting data ingestion pipeline")
    result = pipeline.run()
    
    if hasattr(result, 'wait_until_finish'):
        result.wait_until_finish()
    
    logger.info("Data ingestion pipeline completed")


if __name__ == "__main__":
    run_pipeline()
