"""
Dynamic Pricing Intelligence - Stream Processor Service
Apache Beam pipeline for real-time data processing and analytics
"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms import window
from apache_beam.io import ReadFromPubSub, WriteToBigQuery
import json
import logging
from datetime import datetime
from typing import Dict, Any, Iterator
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.shared.config.settings import get_settings
from src.shared.utils.logging_utils import setup_logging
from src.shared.clients.bigquery_client import BigQueryClient

# Setup logging
logger = setup_logging(__name__)
settings = get_settings()

class ParsePubSubMessage(beam.DoFn):
    """Parse Pub/Sub messages and extract relevant data"""
    
    def process(self, element: bytes) -> Iterator[Dict[str, Any]]:
        try:
            message = json.loads(element.decode('utf-8'))
            
            # Add processing timestamp
            message['processing_timestamp'] = datetime.utcnow().isoformat()
            
            # Validate required fields
            required_fields = ['timestamp', 'location_id', 'event_type']
            if all(field in message for field in required_fields):
                yield message
            else:
                logger.warning(f"Message missing required fields: {message}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

class EnrichLocationData(beam.DoFn):
    """Enrich messages with location metadata"""
    
    def __init__(self):
        self.bigquery_client = None
    
    def setup(self):
        """Initialize BigQuery client"""
        self.bigquery_client = BigQueryClient()
    
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        try:
            location_id = element.get('location_id')
            
            if location_id and self.bigquery_client:
                # Query location metadata
                query = f"""
                SELECT 
                    location_name,
                    business_district,
                    demographic_profile,
                    avg_base_demand,
                    historical_surge_patterns
                FROM `{settings.project_id}.{settings.bigquery_dataset}.location_profiles`
                WHERE location_id = '{location_id}'
                LIMIT 1
                """
                
                results = list(self.bigquery_client.query(query))
                if results:
                    location_data = dict(results[0])
                    element['location_metadata'] = location_data
            
            yield element
            
        except Exception as e:
            logger.error(f"Error enriching location data: {e}")
            yield element

class CalculateRealTimeMetrics(beam.DoFn):
    """Calculate real-time metrics from streaming data"""
    
    def process(self, elements: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        try:
            elements_list = list(elements)
            
            if not elements_list:
                return
            
            # Group by location
            location_groups = {}
            for element in elements_list:
                location_id = element.get('location_id')
                if location_id:
                    if location_id not in location_groups:
                        location_groups[location_id] = []
                    location_groups[location_id].append(element)
            
            # Calculate metrics for each location
            for location_id, location_events in location_groups.items():
                metrics = {
                    'location_id': location_id,
                    'window_start': min(e['timestamp'] for e in location_events),
                    'window_end': max(e['timestamp'] for e in location_events),
                    'event_count': len(location_events),
                    'event_types': list(set(e['event_type'] for e in location_events)),
                    'processing_timestamp': datetime.utcnow().isoformat()
                }
                
                # Calculate specific metrics by event type
                event_type_counts = {}
                for event in location_events:
                    event_type = event['event_type']
                    event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
                
                metrics['event_type_counts'] = event_type_counts
                
                # Calculate demand indicators
                demand_events = [e for e in location_events if e['event_type'] in ['ride_request', 'ride_completed']]
                supply_events = [e for e in location_events if e['event_type'] in ['driver_available', 'driver_busy']]
                
                metrics['demand_signal'] = len(demand_events)
                metrics['supply_signal'] = len(supply_events)
                metrics['demand_supply_ratio'] = (
                    len(demand_events) / len(supply_events) if supply_events else float('inf')
                )
                
                yield metrics
                
        except Exception as e:
            logger.error(f"Error calculating real-time metrics: {e}")

class TriggerPricingUpdate(beam.DoFn):
    """Trigger pricing updates based on real-time metrics"""
    
    def __init__(self):
        self.bigquery_client = None
    
    def setup(self):
        """Initialize BigQuery client"""
        self.bigquery_client = BigQueryClient()
    
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        try:
            location_id = element['location_id']
            demand_supply_ratio = element.get('demand_supply_ratio', 1.0)
            
            # Determine if pricing update is needed
            update_threshold = 1.5  # Trigger update if demand/supply ratio > 1.5
            
            if demand_supply_ratio > update_threshold:
                # Call BigQuery pricing function
                pricing_query = f"""
                SELECT * FROM `{settings.project_id}.{settings.bigquery_dataset}.calculate_optimal_price`(
                    '{location_id}',
                    CURRENT_TIMESTAMP()
                )
                """
                
                pricing_results = list(self.bigquery_client.query(pricing_query))
                if pricing_results:
                    pricing_data = dict(pricing_results[0])
                    
                    # Create pricing update message
                    pricing_update = {
                        'location_id': location_id,
                        'trigger_reason': 'demand_supply_imbalance',
                        'demand_supply_ratio': demand_supply_ratio,
                        'new_pricing': pricing_data,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    yield pricing_update
            
        except Exception as e:
            logger.error(f"Error triggering pricing update: {e}")

def create_pipeline_options() -> PipelineOptions:
    """Create pipeline options for Dataflow"""
    options = {
        'project': settings.project_id,
        'region': settings.region,
        'runner': 'DataflowRunner',
        'temp_location': f'gs://{settings.project_id}-data-processing/temp',
        'staging_location': f'gs://{settings.project_id}-data-processing/staging',
        'job_name': f'dynamic-pricing-stream-processor-{int(datetime.now().timestamp())}',
        'streaming': True,
        'autoscaling_algorithm': 'THROUGHPUT_BASED',
        'max_num_workers': 10,
        'disk_size_gb': 30,
        'machine_type': 'n1-standard-2',
        'use_public_ips': False,
        'network': f'projects/{settings.project_id}/global/networks/{settings.project_id}-vpc',
        'subnetwork': f'projects/{settings.project_id}/regions/{settings.region}/subnetworks/{settings.project_id}-services-subnet'
    }
    
    return PipelineOptions(flags=[], **options)

def run_pipeline():
    """Main pipeline execution"""
    logger.info("Starting Dynamic Pricing Stream Processor")
    
    # Pipeline options
    pipeline_options = create_pipeline_options()
    
    # BigQuery table schemas
    metrics_schema = {
        'fields': [
            {'name': 'location_id', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'window_start', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'window_end', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'event_count', 'type': 'INTEGER', 'mode': 'REQUIRED'},
            {'name': 'event_types', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'event_type_counts', 'type': 'JSON', 'mode': 'NULLABLE'},
            {'name': 'demand_signal', 'type': 'INTEGER', 'mode': 'REQUIRED'},
            {'name': 'supply_signal', 'type': 'INTEGER', 'mode': 'REQUIRED'},
            {'name': 'demand_supply_ratio', 'type': 'FLOAT', 'mode': 'REQUIRED'},
            {'name': 'processing_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'}
        ]
    }
    
    pricing_updates_schema = {
        'fields': [
            {'name': 'location_id', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'trigger_reason', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'demand_supply_ratio', 'type': 'FLOAT', 'mode': 'REQUIRED'},
            {'name': 'new_pricing', 'type': 'JSON', 'mode': 'REQUIRED'},
            {'name': 'timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'}
        ]
    }
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        
        # Read from Pub/Sub
        raw_messages = (
            pipeline
            | 'Read from Pub/Sub' >> ReadFromPubSub(
                subscription=f'projects/{settings.project_id}/subscriptions/realtime-events-subscription'
            )
        )
        
        # Parse and validate messages
        parsed_messages = (
            raw_messages
            | 'Parse Messages' >> beam.ParDo(ParsePubSubMessage())
        )
        
        # Enrich with location data
        enriched_messages = (
            parsed_messages
            | 'Enrich Location Data' >> beam.ParDo(EnrichLocationData())
        )
        
        # Window the data (5-minute sliding windows)
        windowed_messages = (
            enriched_messages
            | 'Apply Windowing' >> beam.WindowInto(
                window.SlidingWindows(size=300, period=60)  # 5-min windows, 1-min slide
            )
        )
        
        # Calculate real-time metrics
        metrics = (
            windowed_messages
            | 'Group by Location' >> beam.GroupBy(lambda x: x['location_id'])
            | 'Calculate Metrics' >> beam.ParDo(CalculateRealTimeMetrics())
        )
        
        # Write metrics to BigQuery
        _ = (
            metrics
            | 'Write Metrics to BigQuery' >> WriteToBigQuery(
                table=f'{settings.project_id}:{settings.bigquery_dataset}.realtime_metrics',
                schema=metrics_schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )
        
        # Trigger pricing updates
        pricing_updates = (
            metrics
            | 'Trigger Pricing Updates' >> beam.ParDo(TriggerPricingUpdate())
        )
        
        # Write pricing updates to BigQuery
        _ = (
            pricing_updates
            | 'Write Pricing Updates to BigQuery' >> WriteToBigQuery(
                table=f'{settings.project_id}:{settings.bigquery_dataset}.pricing_updates',
                schema=pricing_updates_schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )
    
    logger.info("Pipeline execution completed")

if __name__ == '__main__':
    run_pipeline()
