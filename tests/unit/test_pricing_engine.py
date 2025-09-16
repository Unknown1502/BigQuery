"""
Unit tests for the Pricing Engine Service
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.services.pricing_engine.main import PricingEngineService
from src.shared.config.settings import get_settings

@pytest.fixture
def pricing_service():
    """Create a pricing service instance for testing"""
    return PricingEngineService()

@pytest.fixture
def sample_location_data():
    """Sample location data for testing"""
    return {
        'location_id': 'loc_123',
        'latitude': 37.7749,
        'longitude': -122.4194,
        'business_district': 'downtown',
        'demographic_profile': 'business',
        'base_demand': 50
    }

@pytest.fixture
def sample_visual_data():
    """Sample visual analysis data"""
    return {
        'crowd_count': 25,
        'activities': [
            {'type': 'business_meeting', 'confidence': 0.85},
            {'type': 'shopping', 'confidence': 0.65}
        ],
        'accessibility_score': 0.8,
        'traffic_level': 'moderate'
    }

class TestPricingEngineService:
    """Test cases for PricingEngineService"""
    
    @pytest.mark.asyncio
    async def test_calculate_base_price(self, pricing_service, sample_location_data):
        """Test base price calculation"""
        base_price = await pricing_service.calculate_base_price(sample_location_data)
        
        assert isinstance(base_price, float)
        assert base_price > 0
        assert base_price >= 2.0  # Minimum base price
        assert base_price <= 10.0  # Maximum base price
    
    @pytest.mark.asyncio
    async def test_calculate_surge_multiplier(self, pricing_service, sample_visual_data):
        """Test surge multiplier calculation"""
        demand_ratio = 1.5
        weather_factor = 1.2
        
        surge_multiplier = await pricing_service.calculate_surge_multiplier(
            demand_ratio=demand_ratio,
            weather_factor=weather_factor,
            visual_data=sample_visual_data
        )
        
        assert isinstance(surge_multiplier, float)
        assert surge_multiplier >= 0.8  # Minimum surge multiplier
        assert surge_multiplier <= 3.0  # Maximum surge multiplier
    
    @pytest.mark.asyncio
    async def test_visual_factor_calculation(self, pricing_service, sample_visual_data):
        """Test visual factor calculation from crowd and activity data"""
        visual_factor = await pricing_service.calculate_visual_factor(sample_visual_data)
        
        assert isinstance(visual_factor, float)
        assert visual_factor > 0
        
        # High crowd count should increase factor
        high_crowd_data = sample_visual_data.copy()
        high_crowd_data['crowd_count'] = 100
        high_crowd_factor = await pricing_service.calculate_visual_factor(high_crowd_data)
        
        assert high_crowd_factor > visual_factor
    
    @pytest.mark.asyncio
    @patch('src.shared.clients.bigquery_client.BigQueryClient')
    async def test_get_demand_prediction(self, mock_bigquery, pricing_service, sample_location_data):
        """Test demand prediction retrieval"""
        # Mock BigQuery response
        mock_result = Mock()
        mock_result.predicted_demand = 75.5
        mock_result.confidence_interval_lower = 65.0
        mock_result.confidence_interval_upper = 85.0
        mock_result.uncertainty_score = 0.15
        
        mock_bigquery.return_value.query.return_value = [mock_result]
        
        prediction = await pricing_service.get_demand_prediction(
            sample_location_data['location_id']
        )
        
        assert prediction['predicted_demand'] == 75.5
        assert prediction['confidence_interval_lower'] == 65.0
        assert prediction['confidence_interval_upper'] == 85.0
        assert prediction['uncertainty_score'] == 0.15
    
    @pytest.mark.asyncio
    @patch('src.shared.clients.bigquery_client.BigQueryClient')
    async def test_calculate_optimal_price_integration(self, mock_bigquery, pricing_service, sample_location_data):
        """Test complete optimal price calculation"""
        # Mock BigQuery responses
        mock_demand_result = Mock()
        mock_demand_result.predicted_demand = 75.5
        mock_demand_result.confidence_interval_lower = 65.0
        mock_demand_result.confidence_interval_upper = 85.0
        mock_demand_result.uncertainty_score = 0.15
        
        mock_visual_result = Mock()
        mock_visual_result.crowd_count = 25
        mock_visual_result.activities = [{'type': 'business_meeting', 'confidence': 0.85}]
        mock_visual_result.accessibility_score = 0.8
        
        mock_bigquery.return_value.query.side_effect = [
            [mock_demand_result],  # Demand prediction query
            [mock_visual_result]   # Visual analysis query
        ]
        
        result = await pricing_service.calculate_optimal_price(
            location_id=sample_location_data['location_id'],
            current_timestamp=datetime.utcnow()
        )
        
        assert 'base_price' in result
        assert 'surge_multiplier' in result
        assert 'final_price' in result
        assert 'confidence_score' in result
        assert 'reasoning' in result
        
        assert isinstance(result['base_price'], float)
        assert isinstance(result['surge_multiplier'], float)
        assert isinstance(result['final_price'], float)
        assert isinstance(result['confidence_score'], float)
        assert isinstance(result['reasoning'], str)
        
        # Final price should be base price * surge multiplier
        expected_final_price = result['base_price'] * result['surge_multiplier']
        assert abs(result['final_price'] - expected_final_price) < 0.01
    
    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, pricing_service):
        """Test confidence score calculation"""
        # High uncertainty should result in low confidence
        high_uncertainty = 0.8
        low_confidence = await pricing_service.calculate_confidence_score(high_uncertainty)
        
        # Low uncertainty should result in high confidence
        low_uncertainty = 0.1
        high_confidence = await pricing_service.calculate_confidence_score(low_uncertainty)
        
        assert isinstance(low_confidence, float)
        assert isinstance(high_confidence, float)
        assert 0 <= low_confidence <= 1
        assert 0 <= high_confidence <= 1
        assert high_confidence > low_confidence
    
    @pytest.mark.asyncio
    async def test_business_rules_validation(self, pricing_service):
        """Test business rules validation"""
        # Test maximum price increase validation
        current_price = 5.0
        new_price = 15.0  # 200% increase
        
        validated_price = await pricing_service.validate_business_rules(
            current_price=current_price,
            new_price=new_price,
            location_id='loc_123'
        )
        
        # Should be capped by business rules
        assert validated_price < new_price
        assert validated_price <= current_price * 2.0  # Max 100% increase
    
    @pytest.mark.asyncio
    async def test_competitor_analysis_integration(self, pricing_service):
        """Test competitor analysis integration"""
        with patch.object(pricing_service, 'get_competitor_prices') as mock_competitor:
            mock_competitor.return_value = {
                'competitor_a': 4.50,
                'competitor_b': 5.20,
                'competitor_c': 4.80
            }
            
            competitor_analysis = await pricing_service.analyze_competitors('loc_123')
            
            assert 'avg_competitor_price' in competitor_analysis
            assert 'our_position' in competitor_analysis
            assert 'price_gap' in competitor_analysis
            
            # Average should be calculated correctly
            expected_avg = (4.50 + 5.20 + 4.80) / 3
            assert abs(competitor_analysis['avg_competitor_price'] - expected_avg) < 0.01

class TestPricingValidation:
    """Test pricing validation and edge cases"""
    
    @pytest.mark.asyncio
    async def test_invalid_location_handling(self, pricing_service):
        """Test handling of invalid location IDs"""
        with patch('src.shared.clients.bigquery_client.BigQueryClient') as mock_bigquery:
            mock_bigquery.return_value.query.return_value = []  # No results
            
            result = await pricing_service.calculate_optimal_price(
                location_id='invalid_location',
                current_timestamp=datetime.utcnow()
            )
            
            # Should return default pricing
            assert result['base_price'] == 3.50  # Default base price
            assert result['surge_multiplier'] == 1.0  # Default multiplier
    
    @pytest.mark.asyncio
    async def test_extreme_demand_scenarios(self, pricing_service):
        """Test pricing under extreme demand scenarios"""
        # Very high demand
        high_demand_factor = await pricing_service.calculate_surge_multiplier(
            demand_ratio=10.0,  # Extremely high demand
            weather_factor=1.0,
            visual_data={'crowd_count': 200, 'accessibility_score': 0.3}
        )
        
        # Should be capped at maximum
        assert high_demand_factor <= 3.0
        
        # Very low demand
        low_demand_factor = await pricing_service.calculate_surge_multiplier(
            demand_ratio=0.1,  # Very low demand
            weather_factor=1.0,
            visual_data={'crowd_count': 2, 'accessibility_score': 0.9}
        )
        
        # Should be at minimum
        assert low_demand_factor >= 0.8
    
    @pytest.mark.asyncio
    async def test_pricing_consistency(self, pricing_service):
        """Test that pricing calculations are consistent"""
        location_id = 'loc_test'
        timestamp = datetime.utcnow()
        
        with patch('src.shared.clients.bigquery_client.BigQueryClient') as mock_bigquery:
            # Mock consistent responses
            mock_result = Mock()
            mock_result.predicted_demand = 50.0
            mock_result.confidence_interval_lower = 45.0
            mock_result.confidence_interval_upper = 55.0
            mock_result.uncertainty_score = 0.1
            
            mock_bigquery.return_value.query.return_value = [mock_result]
            
            # Calculate price multiple times
            result1 = await pricing_service.calculate_optimal_price(location_id, timestamp)
            result2 = await pricing_service.calculate_optimal_price(location_id, timestamp)
            
            # Results should be identical for same inputs
            assert result1['base_price'] == result2['base_price']
            assert result1['surge_multiplier'] == result2['surge_multiplier']
            assert result1['final_price'] == result2['final_price']

@pytest.mark.integration
class TestPricingEngineIntegration:
    """Integration tests for pricing engine"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pricing_flow(self, pricing_service):
        """Test complete end-to-end pricing calculation flow"""
        # This would require actual BigQuery connection in integration environment
        # For now, we'll test the flow with mocked dependencies
        
        location_id = 'integration_test_location'
        timestamp = datetime.utcnow()
        
        with patch('src.shared.clients.bigquery_client.BigQueryClient') as mock_bigquery:
            # Setup realistic mock responses
            demand_result = Mock()
            demand_result.predicted_demand = 65.0
            demand_result.confidence_interval_lower = 55.0
            demand_result.confidence_interval_upper = 75.0
            demand_result.uncertainty_score = 0.2
            
            visual_result = Mock()
            visual_result.crowd_count = 30
            visual_result.activities = [
                {'type': 'commuting', 'confidence': 0.9},
                {'type': 'shopping', 'confidence': 0.4}
            ]
            visual_result.accessibility_score = 0.7
            
            mock_bigquery.return_value.query.side_effect = [
                [demand_result],
                [visual_result]
            ]
            
            result = await pricing_service.calculate_optimal_price(location_id, timestamp)
            
            # Validate complete result structure
            required_fields = [
                'base_price', 'surge_multiplier', 'final_price',
                'confidence_score', 'reasoning', 'visual_factors',
                'demand_prediction', 'competitor_analysis'
            ]
            
            for field in required_fields:
                assert field in result, f"Missing required field: {field}"
            
            # Validate business logic
            assert result['final_price'] > 0
            assert result['confidence_score'] >= 0 and result['confidence_score'] <= 1
            assert len(result['reasoning']) > 0
            assert isinstance(result['visual_factors'], list)

if __name__ == '__main__':
    pytest.main([__file__])
