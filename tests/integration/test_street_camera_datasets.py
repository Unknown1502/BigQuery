"""
Integration tests for Street Camera Image Dataset System
Tests the complete workflow from camera discovery to dataset export
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
import json
import cv2
import numpy as np

from src.shared.config.gcp_config import GCPConfig
import importlib.util

# Import modules using importlib to handle hyphenated directory
_project_root = Path(__file__).parent.parent.parent
_connector_path = _project_root / "src" / "services" / "data-ingestion" / "connectors" / "street_camera_connector.py"
_collector_path = _project_root / "src" / "services" / "data-ingestion" / "collectors" / "street_image_dataset_collector.py"

# Import street_camera_connector
_connector_spec = importlib.util.spec_from_file_location("street_camera_connector", _connector_path)
_street_camera_connector = importlib.util.module_from_spec(_connector_spec)
_connector_spec.loader.exec_module(_street_camera_connector)

# Import street_image_dataset_collector
_collector_spec = importlib.util.spec_from_file_location("street_image_dataset_collector", _collector_path)
_street_image_dataset_collector = importlib.util.module_from_spec(_collector_spec)
_collector_spec.loader.exec_module(_street_image_dataset_collector)

# Import the classes we need
StreetCameraConnector = _street_camera_connector.StreetCameraConnector
CameraFeed = _street_camera_connector.CameraFeed
ImageCapture = _street_camera_connector.ImageCapture
CameraSource = _street_camera_connector.CameraSource
StreetImageDatasetCollector = _street_image_dataset_collector.StreetImageDatasetCollector
DatasetImage = _street_image_dataset_collector.DatasetImage
DatasetCollection = _street_image_dataset_collector.DatasetCollection

class TestStreetCameraDatasets:
    """Integration tests for street camera dataset system"""
    
    @pytest.fixture
    async def config(self):
        """Test configuration"""
        config = GCPConfig()
        # Override with test values
        config.project_id = "test-project"
        config.street_imagery_bucket = "test-bucket"
        return config
    
    @pytest.fixture
    async def temp_dir(self):
        """Temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def camera_connector(self, config):
        """Camera connector instance"""
        connector = StreetCameraConnector(config)
        await connector.initialize()
        yield connector
        await connector.cleanup()
    
    @pytest.fixture
    async def dataset_collector(self, config, temp_dir):
        """Dataset collector instance"""
        collector = StreetImageDatasetCollector(config)
        # Override dataset directory for testing
        collector.dataset_dir = temp_dir / "datasets"
        collector.dataset_dir.mkdir(parents=True, exist_ok=True)
        await collector.initialize()
        yield collector
        await collector.cleanup()
    
    @pytest.fixture
    def sample_locations(self):
        """Sample test locations"""
        return [
            {
                'location_id': 'test_location_1',
                'lat': 40.7589,
                'lng': -73.9851,
                'name': 'Test Location 1',
                'type': 'test'
            },
            {
                'location_id': 'test_location_2',
                'lat': 40.7829,
                'lng': -73.9654,
                'name': 'Test Location 2',
                'type': 'test'
            }
        ]
    
    @pytest.fixture
    def sample_camera_feeds(self):
        """Sample camera feeds for testing"""
        return [
            CameraFeed(
                camera_id="test_cam_001",
                location_id="test_location_1",
                feed_url="http://test-camera-1.com/feed",
                camera_type="traffic",
                resolution="1920x1080",
                fps=2,
                auth_required=False
            ),
            CameraFeed(
                camera_id="test_cam_002",
                location_id="test_location_2",
                feed_url="http://test-camera-2.com/feed",
                camera_type="security",
                resolution="1280x720",
                fps=1,
                auth_required=False
            )
        ]
    
    @pytest.fixture
    def sample_image_data(self):
        """Generate sample image data"""
        # Create a simple test image
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (540, 540), (255, 255, 255), -1)
        cv2.putText(image, "TEST", (250, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Encode as JPEG
        _, encoded = cv2.imencode('.jpg', image)
        return encoded.tobytes()
    
    @pytest.mark.asyncio
    async def test_camera_feed_discovery(self, camera_connector):
        """Test camera feed discovery functionality"""
        
        # Mock the discovery methods
        with patch.object(camera_connector, '_discover_nyc_dot_cameras') as mock_nyc, \
             patch.object(camera_connector, '_discover_chicago_dot_cameras') as mock_chicago:
            
            # Setup mock returns
            mock_nyc.return_value = [
                CameraSource(
                    source_id="nyc_001",
                    source_name="NYC DOT Camera 001",
                    data_source="nyc_dot",
                    feed_url="http://nyc-cam-001.com/feed",
                    location_coordinates=(40.7589, -73.9851),
                    camera_type="traffic",
                    feed_format="mjpeg",
                    is_active=True
                )
            ]
            
            mock_chicago.return_value = [
                CameraSource(
                    source_id="chi_001",
                    source_name="Chicago DOT Camera 001",
                    data_source="chicago_dot",
                    feed_url="http://chi-cam-001.com/feed",
                    location_coordinates=(41.8781, -87.6298),
                    camera_type="traffic",
                    feed_format="mjpeg",
                    is_active=True
                )
            ]
            
            # Test discovery
            discovered_feeds = await camera_connector.discover_camera_feeds()
            
            assert len(discovered_feeds) >= 2
            assert any(feed.data_source == "nyc_dot" for feed in discovered_feeds)
            assert any(feed.data_source == "chicago_dot" for feed in discovered_feeds)
    
    @pytest.mark.asyncio
    async def test_camera_feed_registration(self, camera_connector, sample_camera_feeds):
        """Test camera feed registration and management"""
        
        # Register feeds
        for feed in sample_camera_feeds:
            await camera_connector.add_camera_feed(feed)
        
        # Check registration
        stats = camera_connector.get_feed_statistics()
        assert stats['total_feeds'] == len(sample_camera_feeds)
        assert stats['active_feeds'] == len(sample_camera_feeds)
        
        # Test feed status update
        await camera_connector.update_feed_status("test_cam_001", False)
        stats = camera_connector.get_feed_statistics()
        assert stats['active_feeds'] == len(sample_camera_feeds) - 1
    
    @pytest.mark.asyncio
    async def test_image_capture_simulation(self, camera_connector, sample_camera_feeds, sample_image_data):
        """Test image capture from camera feeds"""
        
        # Register test feeds
        for feed in sample_camera_feeds:
            await camera_connector.add_camera_feed(feed)
        
        # Mock the image fetching
        with patch.object(camera_connector, '_fetch_image_from_feed') as mock_fetch:
            mock_fetch.return_value = sample_image_data
            
            # Test capture stream (limited)
            captured_images = []
            capture_count = 0
            max_captures = 2
            
            async for capture in camera_connector.start_capture_streams():
                if capture_count >= max_captures:
                    break
                
                captured_images.append(capture)
                capture_count += 1
            
            assert len(captured_images) == max_captures
            
            for capture in captured_images:
                assert isinstance(capture, ImageCapture)
                assert capture.image_data == sample_image_data
                assert capture.camera_id in ["test_cam_001", "test_cam_002"]
                assert capture.location_id in ["test_location_1", "test_location_2"]
    
    @pytest.mark.asyncio
    async def test_dataset_collection_creation(self, dataset_collector, sample_locations):
        """Test comprehensive dataset collection"""
        
        # Mock external API calls
        with patch.object(dataset_collector, '_collect_google_street_view') as mock_gsv, \
             patch.object(dataset_collector, '_collect_mapillary_images') as mock_mapillary, \
             patch.object(dataset_collector, '_collect_camera_feed_images') as mock_camera:
            
            # Setup mock returns
            mock_gsv.return_value = [
                DatasetImage(
                    image_id="gsv_test_1",
                    source="google_street_view",
                    location_id="test_location_1",
                    image_path="/test/path/gsv_1.jpg",
                    coordinates={'lat': 40.7589, 'lng': -73.9851},
                    timestamp=datetime.now(timezone.utc),
                    quality_score=0.8
                )
            ]
            
            mock_mapillary.return_value = [
                DatasetImage(
                    image_id="mapillary_test_1",
                    source="mapillary",
                    location_id="test_location_2",
                    image_path="/test/path/mapillary_1.jpg",
                    coordinates={'lat': 40.7829, 'lng': -73.9654},
                    timestamp=datetime.now(timezone.utc),
                    quality_score=0.7
                )
            ]
            
            mock_camera.return_value = [
                DatasetImage(
                    image_id="camera_test_1",
                    source="camera_feed",
                    location_id="test_location_1",
                    image_path="/test/path/camera_1.jpg",
                    timestamp=datetime.now(timezone.utc),
                    quality_score=0.9
                )
            ]
            
            # Create dataset
            collection = await dataset_collector.create_comprehensive_dataset(
                locations=sample_locations,
                collection_name="Test Dataset",
                sources=['google_street_view', 'mapillary', 'camera_feeds']
            )
            
            assert isinstance(collection, DatasetCollection)
            assert collection.name == "Test Dataset"
            assert len(collection.sources) == 3
            assert collection.total_images > 0
    
    @pytest.mark.asyncio
    async def test_image_quality_analysis(self, dataset_collector, temp_dir, sample_image_data):
        """Test image quality analysis and filtering"""
        
        # Create test image file
        test_image_path = temp_dir / "test_image.jpg"
        with open(test_image_path, 'wb') as f:
            f.write(sample_image_data)
        
        # Create test dataset image
        dataset_image = DatasetImage(
            image_id="quality_test_1",
            source="test",
            location_id="test_location",
            image_path=str(test_image_path),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Process image
        processed_images = await dataset_collector._process_dataset_images([dataset_image])
        
        assert len(processed_images) == 1
        processed_image = processed_images[0]
        
        assert processed_image.quality_score > 0
        assert processed_image.resolution is not None
        assert processed_image.time_of_day is not None
        assert len(processed_image.labels) > 0
    
    @pytest.mark.asyncio
    async def test_dataset_validation(self, dataset_collector, temp_dir, sample_image_data):
        """Test dataset quality validation"""
        
        # Create test collection with images
        collection_id = "test_validation_collection"
        
        # Create test images
        test_images = []
        for i in range(3):
            image_path = temp_dir / f"test_image_{i}.jpg"
            with open(image_path, 'wb') as f:
                f.write(sample_image_data)
            
            dataset_image = DatasetImage(
                image_id=f"validation_test_{i}",
                source="test",
                location_id=f"test_location_{i}",
                image_path=str(image_path),
                quality_score=0.8 - (i * 0.1),  # Varying quality scores
                timestamp=datetime.now(timezone.utc)
            )
            test_images.append(dataset_image)
        
        # Create collection
        collection = DatasetCollection(
            collection_id=collection_id,
            name="Validation Test Collection",
            description="Test collection for validation",
            created_at=datetime.now(timezone.utc),
            total_images=len(test_images),
            sources=["test"],
            locations=["test_location_0", "test_location_1", "test_location_2"],
            size_bytes=len(sample_image_data) * len(test_images),
            images=test_images
        )
        
        dataset_collector.collections[collection_id] = collection
        
        # Validate dataset
        validation_results = await dataset_collector.validate_dataset_quality(collection_id)
        
        assert validation_results['collection_id'] == collection_id
        assert validation_results['total_images'] == len(test_images)
        assert 'quality_metrics' in validation_results
        assert 'summary' in validation_results
        assert validation_results['summary']['overall_health'] in ['good', 'needs_attention']
    
    @pytest.mark.asyncio
    async def test_dataset_export_formats(self, dataset_collector, temp_dir, sample_image_data):
        """Test dataset export in different formats"""
        
        # Create test collection
        collection_id = "test_export_collection"
        
        # Create test image
        image_path = temp_dir / "export_test.jpg"
        with open(image_path, 'wb') as f:
            f.write(sample_image_data)
        
        dataset_image = DatasetImage(
            image_id="export_test_1",
            source="test",
            location_id="test_location",
            image_path=str(image_path),
            coordinates={'lat': 40.7589, 'lng': -73.9851},
            timestamp=datetime.now(timezone.utc),
            resolution=(640, 640),
            quality_score=0.8,
            labels=["test", "export"]
        )
        
        collection = DatasetCollection(
            collection_id=collection_id,
            name="Export Test Collection",
            description="Test collection for export",
            created_at=datetime.now(timezone.utc),
            total_images=1,
            sources=["test"],
            locations=["test_location"],
            size_bytes=len(sample_image_data),
            images=[dataset_image]
        )
        
        dataset_collector.collections[collection_id] = collection
        
        # Test different export formats
        export_formats = ['coco', 'yolo', 'csv']
        
        for format_name in export_formats:
            output_path = await dataset_collector.export_dataset(
                collection_id,
                export_format=format_name,
                output_path=str(temp_dir / f"test_export_{format_name}")
            )
            
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
    
    @pytest.mark.asyncio
    async def test_collection_management(self, dataset_collector):
        """Test collection management operations"""
        
        # Create test collection
        collection = DatasetCollection(
            collection_id="management_test",
            name="Management Test Collection",
            description="Test collection for management operations",
            created_at=datetime.now(timezone.utc),
            total_images=5,
            sources=["test"],
            locations=["test_location"],
            size_bytes=1024 * 1024  # 1MB
        )
        
        dataset_collector.collections["management_test"] = collection
        
        # Test list collections
        collections = dataset_collector.list_collections()
        assert len(collections) >= 1
        assert any(c['collection_id'] == "management_test" for c in collections)
        
        # Test get collection summary
        summary = dataset_collector.get_collection_summary("management_test")
        assert summary['collection_id'] == "management_test"
        assert summary['name'] == "Management Test Collection"
        assert summary['total_images'] == 5
        assert summary['size_mb'] == 1.0
    
    @pytest.mark.asyncio
    async def test_feed_health_monitoring(self, camera_connector, sample_camera_feeds):
        """Test camera feed health monitoring"""
        
        # Register feeds
        for feed in sample_camera_feeds:
            await camera_connector.add_camera_feed(feed)
        
        # Mock feed testing
        with patch.object(camera_connector, 'test_feed_connection') as mock_test:
            mock_test.return_value = {
                'status': 'success',
                'response_time_ms': 150.0,
                'test_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Monitor health
            health_report = await camera_connector.monitor_feed_health()
            
            assert 'total_feeds' in health_report
            assert 'healthy_feeds' in health_report
            assert 'unhealthy_feeds' in health_report
            assert health_report['total_feeds'] == len(sample_camera_feeds)
    
    @pytest.mark.asyncio
    async def test_enhanced_statistics(self, camera_connector, sample_camera_feeds):
        """Test enhanced statistics generation"""
        
        # Register feeds with different sources
        feeds_with_sources = []
        for i, feed in enumerate(sample_camera_feeds):
            feed.data_source = f"test_source_{i % 2}"  # Alternate sources
            feed.feed_format = "mjpeg" if i % 2 == 0 else "rtsp"
            feeds_with_sources.append(feed)
            await camera_connector.add_camera_feed(feed)
        
        # Get enhanced statistics
        stats = camera_connector.get_enhanced_statistics()
        
        assert 'total_feeds' in stats
        assert 'active_feeds' in stats
        assert 'data_sources' in stats
        assert 'feed_formats' in stats
        assert 'camera_types' in stats
        
        assert stats['total_feeds'] == len(sample_camera_feeds)
        assert len(stats['data_sources']) == 2  # Two different sources
        assert len(stats['feed_formats']) == 2  # Two different formats
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, camera_connector, dataset_collector):
        """Test error handling and recovery mechanisms"""
        
        # Test camera connector error handling
        with patch.object(camera_connector, '_fetch_image_from_feed') as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            
            # Should handle errors gracefully
            result = await camera_connector._fetch_image_from_feed(
                CameraFeed(
                    camera_id="error_test",
                    location_id="test_location",
                    feed_url="http://invalid-url.com",
                    camera_type="test",
                    resolution="640x480",
                    fps=1,
                    auth_required=False
                )
            )
            
            assert result is None  # Should return None on error
        
        # Test dataset collector error handling
        with patch.object(dataset_collector, '_collect_google_street_view') as mock_collect:
            mock_collect.side_effect = Exception("API error")
            
            # Should continue with other sources
            collection = await dataset_collector.create_comprehensive_dataset(
                locations=[{'location_id': 'test', 'lat': 40.0, 'lng': -74.0}],
                collection_name="Error Test",
                sources=['google_street_view', 'camera_feeds']
            )
            
            # Should still create collection even if one source fails
            assert isinstance(collection, DatasetCollection)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, camera_connector, sample_camera_feeds):
        """Test concurrent operations and thread safety"""
        
        # Register feeds
        for feed in sample_camera_feeds:
            await camera_connector.add_camera_feed(feed)
        
        # Test concurrent feed testing
        async def test_feed(feed_id):
            return await camera_connector.test_feed_connection(
                camera_connector.active_feeds[feed_id]
            )
        
        # Run concurrent tests
        tasks = [test_feed(feed.camera_id) for feed in sample_camera_feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle concurrent operations
        assert len(results) == len(sample_camera_feeds)
        for result in results:
            assert not isinstance(result, Exception)

@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    
    config = GCPConfig()
    config.project_id = "test-project"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize components
        camera_connector = StreetCameraConnector(config)
        await camera_connector.initialize()
        
        dataset_collector = StreetImageDatasetCollector(config)
        dataset_collector.dataset_dir = temp_path / "datasets"
        dataset_collector.dataset_dir.mkdir(parents=True, exist_ok=True)
        await dataset_collector.initialize()
        
        try:
            # 1. Discover and register camera feeds
            test_feed = CameraFeed(
                camera_id="e2e_test_cam",
                location_id="e2e_test_location",
                feed_url="http://test-camera.com/feed",
                camera_type="traffic",
                resolution="1920x1080",
                fps=2,
                auth_required=False
            )
            
            await camera_connector.add_camera_feed(test_feed)
            
            # 2. Create dataset collection
            locations = [
                {
                    'location_id': 'e2e_test_location',
                    'lat': 40.7589,
                    'lng': -73.9851,
                    'name': 'E2E Test Location'
                }
            ]
            
            # Mock the collection methods for E2E test
            with patch.object(dataset_collector, '_collect_camera_feed_images') as mock_camera:
                # Create test image
                test_image = np.zeros((640, 640, 3), dtype=np.uint8)
                _, encoded = cv2.imencode('.jpg', test_image)
                image_data = encoded.tobytes()
                
                # Create test image file
                image_path = temp_path / "e2e_test.jpg"
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                
                mock_camera.return_value = [
                    DatasetImage(
                        image_id="e2e_test_image",
                        source="camera_feed",
                        location_id="e2e_test_location",
                        image_path=str(image_path),
                        timestamp=datetime.now(timezone.utc),
                        quality_score=0.8,
                        file_size=len(image_data)
                    )
                ]
                
                # Create dataset
                collection = await dataset_collector.create_comprehensive_dataset(
                    locations=locations,
                    collection_name="E2E Test Dataset",
                    sources=['camera_feeds']
                )
                
                # 3. Validate dataset
                validation_results = await dataset_collector.validate_dataset_quality(
                    collection.collection_id
                )
                
                # 4. Export dataset
                export_path = await dataset_collector.export_dataset(
                    collection.collection_id,
                    export_format='csv',
                    output_path=str(temp_path / "e2e_export.csv")
                )
                
                # Verify end-to-end workflow
                assert collection.total_images > 0
                assert validation_results['total_images'] > 0
                assert Path(export_path).exists()
                
        finally:
            # Cleanup
            await camera_connector.cleanup()
            await dataset_collector.cleanup()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
