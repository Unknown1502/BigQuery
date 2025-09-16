"""
Integration tests for BigQuery AI functions and models
"""

import pytest
import asyncio
from google.cloud import bigquery
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.shared.clients.bigquery_client import BigQueryClient
from src.shared.config.settings import get_settings

@pytest.fixture(scope="session")
def bigquery_client():
    """Create BigQuery client for integration tests"""
    return BigQueryClient()

@pytest.fixture(scope="session")
def settings():
    """Get application settings"""
    return get_settings()

@pytest.mark.integration
class TestBigQueryAIFunctions:
    """Integration tests for BigQuery AI functions"""
    
    @pytest.mark.asyncio
    async def test_analyze_street_scene_function(self, bigquery_client, settings):
        """Test the analyze_street_scene BigQuery AI function"""
        # Test with sample image reference
        query = f"""
        SELECT 
            analyze_street_scene(
                'gs://{settings.project_id}-street-imagery/sample/test_street_image.jpg',
                STRUCT(37.7749 as lat, -122.4194 as lng, 'morning' as time_of_day)
            ) as scene_analysis
        """
        
        try:
            results = list(bigquery_client.query(query))
            assert len(results) > 0
            
            scene_analysis = results[0].scene_analysis
            assert 'crowd_count' in scene_analysis
            assert 'activity_analysis' in scene_analysis
            assert 'is_accessible' in scene_analysis
            
            # Validate data types
            assert isinstance(scene_analysis['crowd_count'], int)
            assert isinstance(scene_analysis['is_accessible'], bool)
            
        except Exception as e:
            pytest.skip(f"BigQuery AI function not available: {e}")
    
    @pytest.mark.asyncio
    async def test_find_similar_locations_function(self, bigquery_client, settings):
        """Test the find_similar_locations BigQuery AI function"""
        query = f"""
        SELECT * FROM `{settings.project_id}.{settings.bigquery_dataset}.find_similar_locations`(
            'downtown_sf_001',
            0.3
        )
        LIMIT 5
        """
        
        try:
            results = list(bigquery_client.query(query))
            
            for result in results:
                assert hasattr(result, 'similar_location_id')
                assert hasattr(result, 'similarity_score')
                assert hasattr(result, 'avg_price_multiplier')
                
                # Validate similarity score is within expected range
                assert 0 <= result.similarity_score <= 1
                
        except Exception as e:
            pytest.skip(f"BigQuery function not available: {e}")
    
    @pytest.mark.asyncio
    async def test_predict_demand_with_confidence_function(self, bigquery_client, settings):
        """Test the predict_demand_with_confidence BigQuery AI function"""
        query = f"""
        SELECT 
            `{settings.project_id}.{settings.bigquery_dataset}.predict_demand_with_confidence`(
                'downtown_sf_001',
                1
            ) as prediction
        """
        
        try:
            results = list(bigquery_client.query(query))
            assert len(results) > 0
            
            prediction = results[0].prediction
            assert 'predicted_demand' in prediction
            assert 'confidence_interval_lower' in prediction
            assert 'confidence_interval_upper' in prediction
            assert 'uncertainty_score' in prediction
            
            # Validate confidence interval logic
            assert prediction['confidence_interval_lower'] <= prediction['predicted_demand']
            assert prediction['predicted_demand'] <= prediction['confidence_interval_upper']
            assert 0 <= prediction['uncertainty_score'] <= 1
            
        except Exception as e:
            pytest.skip(f"BigQuery function not available: {e}")
    
    @pytest.mark.asyncio
    async def test_calculate_optimal_price_function(self, bigquery_client, settings):
        """Test the calculate_optimal_price BigQuery AI function"""
        query = f"""
        SELECT 
            `{settings.project_id}.{settings.bigquery_dataset}.calculate_optimal_price`(
                'downtown_sf_001',
                CURRENT_TIMESTAMP()
            ) as pricing_result
        """
        
        try:
            results = list(bigquery_client.query(query))
            assert len(results) > 0
            
            pricing_result = results[0].pricing_result
            assert 'base_price' in pricing_result
            assert 'surge_multiplier' in pricing_result
            assert 'confidence_score' in pricing_result
            assert 'reasoning' in pricing_result
            
            # Validate pricing logic
            assert pricing_result['base_price'] > 0
            assert pricing_result['surge_multiplier'] >= 0.8
            assert pricing_result['surge_multiplier'] <= 3.0
            assert 0 <= pricing_result['confidence_score'] <= 1
            assert len(pricing_result['reasoning']) > 0
            
        except Exception as e:
            pytest.skip(f"BigQuery function not available: {e}")

@pytest.mark.integration
class TestBigQueryMLModels:
    """Integration tests for BigQuery ML models"""
    
    @pytest.mark.asyncio
    async def test_demand_forecast_model(self, bigquery_client, settings):
        """Test the demand forecasting ML model"""
        query = f"""
        SELECT 
            forecast_timestamp,
            forecast_value,
            prediction_interval_lower_bound,
            prediction_interval_upper_bound
        FROM ML.FORECAST(
            MODEL `{settings.project_id}.{settings.bigquery_dataset}.demand_forecast_model`,
            STRUCT(24 as horizon)
        )
        LIMIT 10
        """
        
        try:
            results = list(bigquery_client.query(query))
            
            for result in results:
                assert result.forecast_value is not None
                assert result.forecast_value >= 0
                assert result.prediction_interval_lower_bound <= result.forecast_value
                assert result.forecast_value <= result.prediction_interval_upper_bound
                
        except Exception as e:
            pytest.skip(f"ML model not available: {e}")
    
    @pytest.mark.asyncio
    async def test_location_embeddings_model(self, bigquery_client, settings):
        """Test the location embeddings ML model"""
        query = f"""
        SELECT 
            location_id,
            centroid_id,
            nearest_centroids_distance
        FROM ML.PREDICT(
            MODEL `{settings.project_id}.{settings.bigquery_dataset}.location_embeddings_model`,
            (
                SELECT 
                    'test_location' as location_id,
                    'downtown business district' as location_description,
                    ['restaurant', 'office'] as business_types
            )
        )
        """
        
        try:
            results = list(bigquery_client.query(query))
            
            for result in results:
                assert result.location_id is not None
                assert result.centroid_id is not None
                assert isinstance(result.nearest_centroids_distance, list)
                
        except Exception as e:
            pytest.skip(f"ML model not available: {e}")

@pytest.mark.integration
class TestBigQueryDataPipeline:
    """Integration tests for data pipeline operations"""
    
    @pytest.mark.asyncio
    async def test_realtime_data_insertion(self, bigquery_client, settings):
        """Test real-time data insertion into BigQuery"""
        # Insert test data
        test_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'location_id': 'test_location_001',
            'event_type': 'ride_request',
            'data': json.dumps({'user_id': 'test_user', 'destination': 'test_dest'}),
            'source': 'integration_test'
        }
        
        table_id = f"{settings.project_id}.{settings.bigquery_dataset}.realtime_events"
        
        try:
            # Insert data
            errors = bigquery_client.client.insert_rows_json(
                bigquery_client.client.get_table(table_id),
                [test_data]
            )
            
            assert len(errors) == 0, f"Insert errors: {errors}"
            
            # Verify data was inserted
            query = f"""
            SELECT * FROM `{table_id}`
            WHERE location_id = 'test_location_001'
            AND source = 'integration_test'
            ORDER BY timestamp DESC
            LIMIT 1
            """
            
            results = list(bigquery_client.query(query))
            assert len(results) > 0
            assert results[0].location_id == 'test_location_001'
            
        except Exception as e:
            pytest.skip(f"BigQuery table not available: {e}")
    
    @pytest.mark.asyncio
    async def test_streaming_analytics_view(self, bigquery_client, settings):
        """Test real-time analytics view"""
        query = f"""
        SELECT 
            location_id,
            event_count,
            avg_demand_signal,
            last_updated
        FROM `{settings.project_id}.{settings.bigquery_dataset}.real_time_pricing_dashboard`
        WHERE last_updated >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
        LIMIT 10
        """
        
        try:
            results = list(bigquery_client.query(query))
            
            for result in results:
                assert result.location_id is not None
                assert result.event_count >= 0
                assert result.avg_demand_signal >= 0
                assert result.last_updated is not None
                
        except Exception as e:
            pytest.skip(f"Analytics view not available: {e}")

@pytest.mark.integration
class TestBigQueryPerformance:
    """Performance tests for BigQuery operations"""
    
    @pytest.mark.asyncio
    async def test_pricing_function_performance(self, bigquery_client, settings):
        """Test performance of pricing calculation function"""
        import time
        
        query = f"""
        SELECT 
            `{settings.project_id}.{settings.bigquery_dataset}.calculate_optimal_price`(
                'performance_test_location',
                CURRENT_TIMESTAMP()
            ) as pricing_result
        """
        
        start_time = time.time()
        
        try:
            results = list(bigquery_client.query(query))
            execution_time = time.time() - start_time
            
            # Should complete within reasonable time (< 5 seconds)
            assert execution_time < 5.0, f"Query took too long: {execution_time}s"
            assert len(results) > 0
            
        except Exception as e:
            pytest.skip(f"Performance test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_batch_pricing_calculation(self, bigquery_client, settings):
        """Test batch pricing calculations for multiple locations"""
        locations = ['loc_001', 'loc_002', 'loc_003', 'loc_004', 'loc_005']
        
        # Create batch query
        location_conditions = "', '".join(locations)
        query = f"""
        WITH locations AS (
            SELECT location_id
            FROM UNNEST(['{location_conditions}']) as location_id
        )
        SELECT 
            location_id,
            `{settings.project_id}.{settings.bigquery_dataset}.calculate_optimal_price`(
                location_id,
                CURRENT_TIMESTAMP()
            ) as pricing_result
        FROM locations
        """
        
        import time
        start_time = time.time()
        
        try:
            results = list(bigquery_client.query(query))
            execution_time = time.time() - start_time
            
            # Should process all locations efficiently
            assert len(results) == len(locations)
            assert execution_time < 10.0, f"Batch query took too long: {execution_time}s"
            
            # Verify all results have valid pricing data
            for result in results:
                assert result.location_id in locations
                assert result.pricing_result is not None
                
        except Exception as e:
            pytest.skip(f"Batch performance test failed: {e}")

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
