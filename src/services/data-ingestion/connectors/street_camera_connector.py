"""
Enhanced Street Camera Connector
Connects to various street camera feeds and processes real-time imagery
Supports multiple feed types, real camera sources, and automated dataset collection
"""

import asyncio
import aiohttp
import cv2
import numpy as np
from typing import Dict, List, Optional, AsyncGenerator, Any, Union
from datetime import datetime, timezone, timedelta
import json
import base64
from dataclasses import dataclass, field
import logging
import hashlib
from pathlib import Path
import requests
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

from src.shared.utils.logging_utils import get_logger
from src.shared.clients.cloud_storage_client import CloudStorageClient
from src.shared.config.gcp_config import GCPConfig
from src.shared.utils.error_handling import RetryableError

logger = get_logger(__name__)

@dataclass
class CameraFeed:
    """Enhanced camera feed configuration"""
    camera_id: str
    location_id: str
    feed_url: str
    camera_type: str  # 'traffic', 'security', 'public', 'transit', 'commercial'
    resolution: str
    fps: int
    auth_required: bool
    auth_token: Optional[str] = None
    is_active: bool = True
    # Enhanced fields
    feed_format: str = 'mjpeg'  # 'mjpeg', 'rtsp', 'hls', 'static_image'
    location_coordinates: Optional[Dict[str, float]] = None
    camera_angle: Optional[int] = None  # 0-360 degrees
    coverage_area: Optional[str] = None  # 'intersection', 'street', 'plaza', 'building'
    data_source: str = 'unknown'  # 'dot', 'city', 'private', 'api'
    quality_score: float = 1.0  # 0.0-1.0 based on image quality
    last_successful_capture: Optional[datetime] = None
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ImageCapture:
    """Enhanced captured image data"""
    camera_id: str
    location_id: str
    capture_timestamp: datetime
    image_data: bytes
    image_format: str
    metadata: Dict[str, Any]
    # Enhanced fields
    image_hash: Optional[str] = None
    image_size: Optional[tuple] = None  # (width, height)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    weather_conditions: Optional[Dict[str, Any]] = None
    processing_status: str = 'pending'  # 'pending', 'processed', 'failed'

@dataclass
class CameraSource:
    """Camera source configuration for discovery"""
    source_name: str
    source_type: str  # 'dot_api', 'city_portal', 'rtsp_stream', 'web_scraper'
    base_url: str
    api_key: Optional[str] = None
    discovery_endpoint: Optional[str] = None
    auth_method: str = 'none'  # 'none', 'api_key', 'basic_auth', 'oauth'
    rate_limit: int = 60  # requests per minute
    supported_formats: List[str] = field(default_factory=lambda: ['mjpeg'])

class StreetCameraConnector:
    """
    Enhanced connector for various street camera feeds with real-world integration
    """
    
    def __init__(self, config: GCPConfig):
        self.config = config
        self.storage_client = CloudStorageClient(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_feeds: Dict[str, CameraFeed] = {}
        self.camera_sources: Dict[str, CameraSource] = {}
        self.capture_interval = 30  # seconds
        self.quality_threshold = 0.7  # Minimum quality score for images
        self.max_failure_count = 5  # Max failures before deactivating feed
        
        # Initialize real camera sources
        self._initialize_camera_sources()
        
    async def initialize(self):
        """Initialize the connector"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        
        # Load camera feed configurations
        await self._load_camera_feeds()
        
        logger.info(f"Initialized street camera connector with {len(self.active_feeds)} feeds")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def _load_camera_feeds(self):
        """Load camera feed configurations from BigQuery"""
        try:
            # This would typically load from a configuration table
            # For demo purposes, we'll use hardcoded feeds
            demo_feeds = [
                CameraFeed(
                    camera_id="cam_downtown_001",
                    location_id="downtown_business_district",
                    feed_url="https://example-traffic-cam.com/feed/001",
                    camera_type="traffic",
                    resolution="1920x1080",
                    fps=2,
                    auth_required=False
                ),
                CameraFeed(
                    camera_id="cam_airport_002",
                    location_id="airport_terminal_pickup",
                    feed_url="https://example-security-cam.com/feed/002",
                    camera_type="security",
                    resolution="1280x720",
                    fps=1,
                    auth_required=True,
                    auth_token="demo_token_123"
                ),
                CameraFeed(
                    camera_id="cam_entertainment_003",
                    location_id="entertainment_district",
                    feed_url="https://example-public-cam.com/feed/003",
                    camera_type="public",
                    resolution="1920x1080",
                    fps=1,
                    auth_required=False
                )
            ]
            
            for feed in demo_feeds:
                if feed.is_active:
                    self.active_feeds[feed.camera_id] = feed
                    
        except Exception as e:
            logger.error(f"Error loading camera feeds: {e}")
            raise
    
    async def start_capture_streams(self) -> AsyncGenerator[ImageCapture, None]:
        """Start capturing from all active camera feeds"""
        try:
            # Create tasks for all active feeds
            tasks = [
                self._capture_from_feed(feed)
                for feed in self.active_feeds.values()
            ]
            
            # Process captures as they come in
            async for capture in self._merge_capture_streams(tasks):
                yield capture
                
        except Exception as e:
            logger.error(f"Error in capture streams: {e}")
            raise
    
    async def _merge_capture_streams(self, tasks: List[asyncio.Task]) -> AsyncGenerator[ImageCapture, None]:
        """Merge multiple capture streams into a single stream"""
        queue = asyncio.Queue()
        
        async def task_wrapper(task):
            async for capture in task:
                await queue.put(capture)
        
        # Start all tasks
        wrapped_tasks = [asyncio.create_task(task_wrapper(task)) for task in tasks]
        
        try:
            while True:
                # Get next capture with timeout
                try:
                    capture = await asyncio.wait_for(queue.get(), timeout=60.0)
                    yield capture
                except asyncio.TimeoutError:
                    logger.warning("No captures received in 60 seconds")
                    continue
                    
        finally:
            # Cancel all tasks
            for task in wrapped_tasks:
                task.cancel()
    
    async def _capture_from_feed(self, feed: CameraFeed) -> AsyncGenerator[ImageCapture, None]:
        """Capture images from a specific camera feed"""
        logger.info(f"Starting capture from feed {feed.camera_id}")
        
        while True:
            try:
                # Capture image from feed
                image_data = await self._fetch_image_from_feed(feed)
                
                if image_data:
                    # Create capture object
                    capture = ImageCapture(
                        camera_id=feed.camera_id,
                        location_id=feed.location_id,
                        capture_timestamp=datetime.now(timezone.utc),
                        image_data=image_data,
                        image_format="jpeg",
                        metadata={
                            "camera_type": feed.camera_type,
                            "resolution": feed.resolution,
                            "fps": feed.fps,
                            "feed_url": feed.feed_url
                        }
                    )
                    
                    yield capture
                
                # Wait for next capture
                await asyncio.sleep(self.capture_interval)
                
            except Exception as e:
                logger.error(f"Error capturing from feed {feed.camera_id}: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _fetch_image_from_feed(self, feed: CameraFeed) -> Optional[bytes]:
        """Fetch a single image from a camera feed"""
        try:
            headers = {}
            if feed.auth_required and feed.auth_token:
                headers['Authorization'] = f"Bearer {feed.auth_token}"
            
            async with self.session.get(feed.feed_url, headers=headers) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    if 'image' in content_type:
                        return await response.read()
                    elif 'application/json' in content_type:
                        # Handle JSON response with base64 image
                        data = await response.json()
                        if 'image_data' in data:
                            return base64.b64decode(data['image_data'])
                    else:
                        logger.warning(f"Unexpected content type from {feed.camera_id}: {content_type}")
                        return None
                else:
                    logger.warning(f"HTTP {response.status} from feed {feed.camera_id}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching from feed {feed.camera_id}")
            return None
        except Exception as e:
            logger.error(f"Error fetching from feed {feed.camera_id}: {e}")
            return None
    
    async def store_image_capture(self, capture: ImageCapture) -> str:
        """Store captured image in Cloud Storage"""
        try:
            # Generate storage path
            timestamp_str = capture.capture_timestamp.strftime("%Y/%m/%d/%H")
            filename = f"{capture.camera_id}_{int(capture.capture_timestamp.timestamp())}.{capture.image_format}"
            storage_path = f"street-imagery/{timestamp_str}/{filename}"
            
            # Upload to Cloud Storage
            blob_uri = await self.storage_client.upload_bytes(
                bucket_name=self.config.street_imagery_bucket,
                blob_name=storage_path,
                data=capture.image_data,
                content_type=f"image/{capture.image_format}"
            )
            
            logger.debug(f"Stored image capture: {blob_uri}")
            return blob_uri
            
        except Exception as e:
            logger.error(f"Error storing image capture: {e}")
            raise
    
    async def get_recent_captures(
        self, 
        location_id: str, 
        hours_back: int = 1
    ) -> List[Dict[str, Any]]:
        """Get recent image captures for a location"""
        try:
            # This would query BigQuery for recent captures
            # For demo purposes, return mock data
            captures = [
                {
                    "camera_id": f"cam_{location_id}_001",
                    "location_id": location_id,
                    "capture_timestamp": datetime.now(timezone.utc).isoformat(),
                    "storage_uri": f"gs://{self.config.street_imagery_bucket}/street-imagery/2024/01/15/12/cam_{location_id}_001_1705320000.jpeg",
                    "analysis_status": "pending"
                }
            ]
            
            return captures
            
        except Exception as e:
            logger.error(f"Error getting recent captures for {location_id}: {e}")
            return []
    
    async def update_feed_status(self, camera_id: str, is_active: bool):
        """Update the status of a camera feed"""
        try:
            if camera_id in self.active_feeds:
                self.active_feeds[camera_id].is_active = is_active
                logger.info(f"Updated feed {camera_id} status to {'active' if is_active else 'inactive'}")
            else:
                logger.warning(f"Camera feed {camera_id} not found")
                
        except Exception as e:
            logger.error(f"Error updating feed status: {e}")
    
    async def add_camera_feed(self, feed: CameraFeed):
        """Add a new camera feed"""
        try:
            self.active_feeds[feed.camera_id] = feed
            logger.info(f"Added new camera feed: {feed.camera_id}")
            
        except Exception as e:
            logger.error(f"Error adding camera feed: {e}")
    
    async def remove_camera_feed(self, camera_id: str):
        """Remove a camera feed"""
        try:
            if camera_id in self.active_feeds:
                del self.active_feeds[camera_id]
                logger.info(f"Removed camera feed: {camera_id}")
            else:
                logger.warning(f"Camera feed {camera_id} not found")
                
        except Exception as e:
            logger.error(f"Error removing camera feed: {e}")
    
    async def test_feed_connection(self, feed: CameraFeed) -> Dict[str, Any]:
        """Test connection to a camera feed"""
        try:
            start_time = datetime.now(timezone.utc)
            
            # Try to fetch an image
            image_data = await self._fetch_image_from_feed(feed)
            
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000
            
            result = {
                "camera_id": feed.camera_id,
                "status": "success" if image_data else "failed",
                "response_time_ms": response_time,
                "image_size_bytes": len(image_data) if image_data else 0,
                "test_timestamp": start_time.isoformat()
            }
            
            if image_data:
                logger.info(f"Feed test successful for {feed.camera_id}: {len(image_data)} bytes in {response_time:.1f}ms")
            else:
                logger.warning(f"Feed test failed for {feed.camera_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing feed {feed.camera_id}: {e}")
            return {
                "camera_id": feed.camera_id,
                "status": "error",
                "error": str(e),
                "test_timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_feed_statistics(self) -> Dict[str, Any]:
        """Get statistics about camera feeds"""
        active_count = sum(1 for feed in self.active_feeds.values() if feed.is_active)
        
        feed_types = {}
        for feed in self.active_feeds.values():
            feed_types[feed.camera_type] = feed_types.get(feed.camera_type, 0) + 1
        
        return {
            "total_feeds": len(self.active_feeds),
            "active_feeds": active_count,
            "inactive_feeds": len(self.active_feeds) - active_count,
            "feed_types": feed_types,
            "capture_interval_seconds": self.capture_interval
        }
    
    def _initialize_camera_sources(self):
        """Initialize real camera sources for discovery"""
        self.camera_sources = {
            # NYC DOT Traffic Cameras
            "nyc_dot": CameraSource(
                source_name="NYC DOT Traffic Cameras",
                source_type="dot_api",
                base_url="https://webcams.nyctmc.org",
                discovery_endpoint="/api/cameras",
                auth_method="none",
                rate_limit=60,
                supported_formats=["mjpeg", "static_image"]
            ),
            
            # Chicago Traffic Cameras
            "chicago_dot": CameraSource(
                source_name="Chicago DOT Cameras",
                source_type="city_portal",
                base_url="https://www.chicago.gov/city/en/depts/cdot",
                discovery_endpoint="/traffic_cameras.json",
                auth_method="none",
                rate_limit=30,
                supported_formats=["static_image"]
            ),
            
            # Washington State DOT
            "wsdot": CameraSource(
                source_name="Washington State DOT",
                source_type="dot_api",
                base_url="https://www.wsdot.wa.gov/traffic/api",
                discovery_endpoint="/cameras",
                auth_method="none",
                rate_limit=120,
                supported_formats=["mjpeg", "static_image"]
            ),
            
            # California DOT (Caltrans)
            "caltrans": CameraSource(
                source_name="California DOT",
                source_type="web_scraper",
                base_url="https://cwwp2.dot.ca.gov/vm",
                discovery_endpoint="/noauth/cameras.json",
                auth_method="none",
                rate_limit=60,
                supported_formats=["static_image"]
            )
        }
    
    async def discover_camera_feeds(self, source_name: Optional[str] = None) -> List[CameraFeed]:
        """Discover available camera feeds from configured sources"""
        discovered_feeds = []
        
        sources_to_check = [self.camera_sources[source_name]] if source_name else self.camera_sources.values()
        
        for source in sources_to_check:
            try:
                logger.info(f"Discovering cameras from {source.source_name}")
                feeds = await self._discover_from_source(source)
                discovered_feeds.extend(feeds)
                
            except Exception as e:
                logger.error(f"Error discovering from {source.source_name}: {e}")
                continue
        
        logger.info(f"Discovered {len(discovered_feeds)} camera feeds")
        return discovered_feeds
    
    async def _discover_from_source(self, source: CameraSource) -> List[CameraFeed]:
        """Discover camera feeds from a specific source"""
        feeds = []
        
        try:
            if source.source_type == "dot_api":
                feeds = await self._discover_dot_api(source)
            elif source.source_type == "city_portal":
                feeds = await self._discover_city_portal(source)
            elif source.source_type == "web_scraper":
                feeds = await self._discover_web_scraper(source)
            
        except Exception as e:
            logger.error(f"Error discovering from {source.source_name}: {e}")
        
        return feeds
    
    async def _discover_dot_api(self, source: CameraSource) -> List[CameraFeed]:
        """Discover cameras from DOT API"""
        feeds = []
        
        try:
            url = f"{source.base_url}{source.discovery_endpoint}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse camera data (format varies by DOT)
                    cameras = data.get('cameras', data.get('features', []))
                    
                    for camera in cameras:
                        try:
                            camera_id = camera.get('id', camera.get('properties', {}).get('id'))
                            if not camera_id:
                                continue
                            
                            # Extract camera information
                            properties = camera.get('properties', camera)
                            geometry = camera.get('geometry', {})
                            coordinates = geometry.get('coordinates', [None, None])
                            
                            feed = CameraFeed(
                                camera_id=f"{source.source_name.lower().replace(' ', '_')}_{camera_id}",
                                location_id=f"location_{camera_id}",
                                feed_url=properties.get('url', properties.get('image_url', '')),
                                camera_type="traffic",
                                resolution=properties.get('resolution', '640x480'),
                                fps=1,  # Most DOT cameras are static images
                                auth_required=False,
                                feed_format='static_image',
                                location_coordinates={
                                    'lat': coordinates[1] if len(coordinates) > 1 else None,
                                    'lng': coordinates[0] if len(coordinates) > 0 else None
                                },
                                coverage_area=properties.get('location', 'intersection'),
                                data_source=source.source_name.lower().replace(' ', '_'),
                                metadata={
                                    'description': properties.get('description', ''),
                                    'location_name': properties.get('name', ''),
                                    'direction': properties.get('direction', ''),
                                    'source': source.source_name
                                }
                            )
                            
                            if feed.feed_url:
                                feeds.append(feed)
                                
                        except Exception as e:
                            logger.warning(f"Error parsing camera data: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error fetching from DOT API {source.base_url}: {e}")
        
        return feeds
    
    async def _discover_city_portal(self, source: CameraSource) -> List[CameraFeed]:
        """Discover cameras from city portal"""
        feeds = []
        
        try:
            url = f"{source.base_url}{source.discovery_endpoint}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for camera in data.get('cameras', []):
                        try:
                            feed = CameraFeed(
                                camera_id=f"city_{camera.get('id')}",
                                location_id=f"city_location_{camera.get('id')}",
                                feed_url=camera.get('image_url', ''),
                                camera_type="public",
                                resolution=camera.get('resolution', '640x480'),
                                fps=1,
                                auth_required=False,
                                feed_format='static_image',
                                location_coordinates={
                                    'lat': camera.get('latitude'),
                                    'lng': camera.get('longitude')
                                },
                                coverage_area=camera.get('area_type', 'street'),
                                data_source=source.source_name.lower().replace(' ', '_'),
                                metadata={
                                    'description': camera.get('description', ''),
                                    'address': camera.get('address', ''),
                                    'source': source.source_name
                                }
                            )
                            
                            if feed.feed_url:
                                feeds.append(feed)
                                
                        except Exception as e:
                            logger.warning(f"Error parsing city camera data: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error fetching from city portal {source.base_url}: {e}")
        
        return feeds
    
    async def _discover_web_scraper(self, source: CameraSource) -> List[CameraFeed]:
        """Discover cameras using web scraping"""
        feeds = []
        
        try:
            # This would implement web scraping logic
            # For now, return empty list as web scraping requires specific implementation
            logger.info(f"Web scraping not implemented for {source.source_name}")
            
        except Exception as e:
            logger.error(f"Error in web scraping for {source.source_name}: {e}")
        
        return feeds
    
    async def enhance_image_capture(self, capture: ImageCapture) -> ImageCapture:
        """Enhance captured image with additional metadata and quality metrics"""
        try:
            # Calculate image hash for deduplication
            capture.image_hash = hashlib.md5(capture.image_data).hexdigest()
            
            # Decode image to get dimensions and quality metrics
            image_array = np.frombuffer(capture.image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is not None:
                capture.image_size = (image.shape[1], image.shape[0])  # (width, height)
                
                # Calculate quality metrics
                capture.quality_metrics = self._calculate_image_quality(image)
                
                # Add weather conditions if available
                capture.weather_conditions = await self._get_weather_conditions(
                    capture.location_id, capture.capture_timestamp
                )
            
            return capture
            
        except Exception as e:
            logger.error(f"Error enhancing image capture: {e}")
            return capture
    
    def _calculate_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate image quality metrics"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = gray.std()
            
            # Normalize metrics to 0-1 scale
            quality_metrics = {
                'sharpness': min(1.0, sharpness / 1000.0),  # Normalize sharpness
                'brightness': brightness / 255.0,
                'contrast': min(1.0, contrast / 128.0),
                'overall_quality': 0.0
            }
            
            # Calculate overall quality score
            quality_metrics['overall_quality'] = (
                quality_metrics['sharpness'] * 0.4 +
                min(1.0, abs(quality_metrics['brightness'] - 0.5) * 2) * 0.3 +
                quality_metrics['contrast'] * 0.3
            )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error calculating image quality: {e}")
            return {'overall_quality': 0.5}  # Default quality
    
    async def _get_weather_conditions(self, location_id: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get weather conditions for the capture location and time"""
        try:
            # This would integrate with weather APIs
            # For now, return None as it requires weather service integration
            return None
            
        except Exception as e:
            logger.error(f"Error getting weather conditions: {e}")
            return None
    
    async def batch_process_captures(self, captures: List[ImageCapture]) -> List[ImageCapture]:
        """Process multiple captures in batch for efficiency"""
        processed_captures = []
        
        for capture in captures:
            try:
                # Enhance capture with metadata
                enhanced_capture = await self.enhance_image_capture(capture)
                
                # Filter by quality threshold
                overall_quality = enhanced_capture.quality_metrics.get('overall_quality', 0.0)
                if overall_quality >= self.quality_threshold:
                    enhanced_capture.processing_status = 'processed'
                    processed_captures.append(enhanced_capture)
                else:
                    logger.debug(f"Filtered low quality image: {enhanced_capture.camera_id}")
                    enhanced_capture.processing_status = 'filtered'
                
            except Exception as e:
                logger.error(f"Error processing capture {capture.camera_id}: {e}")
                capture.processing_status = 'failed'
                processed_captures.append(capture)
        
        return processed_captures
    
    async def monitor_feed_health(self):
        """Monitor health of all active camera feeds"""
        health_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_feeds': len(self.active_feeds),
            'healthy_feeds': 0,
            'unhealthy_feeds': 0,
            'feed_details': []
        }
        
        for camera_id, feed in self.active_feeds.items():
            try:
                # Test feed connection
                test_result = await self.test_feed_connection(feed)
                
                is_healthy = test_result['status'] == 'success'
                
                if is_healthy:
                    feed.failure_count = 0
                    feed.last_successful_capture = datetime.now(timezone.utc)
                    health_report['healthy_feeds'] += 1
                else:
                    feed.failure_count += 1
                    health_report['unhealthy_feeds'] += 1
                    
                    # Deactivate feed if too many failures
                    if feed.failure_count >= self.max_failure_count:
                        feed.is_active = False
                        logger.warning(f"Deactivated feed {camera_id} due to {feed.failure_count} failures")
                
                health_report['feed_details'].append({
                    'camera_id': camera_id,
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'failure_count': feed.failure_count,
                    'last_successful_capture': feed.last_successful_capture.isoformat() if feed.last_successful_capture else None,
                    'response_time_ms': test_result.get('response_time_ms', 0)
                })
                
            except Exception as e:
                logger.error(f"Error monitoring feed {camera_id}: {e}")
                health_report['unhealthy_feeds'] += 1
        
        logger.info(f"Feed health check: {health_report['healthy_feeds']} healthy, {health_report['unhealthy_feeds']} unhealthy")
        return health_report
    
    async def auto_discover_and_register_feeds(self) -> int:
        """Automatically discover and register new camera feeds"""
        try:
            # Discover feeds from all sources
            discovered_feeds = await self.discover_camera_feeds()
            
            new_feeds_count = 0
            
            for feed in discovered_feeds:
                # Check if feed already exists
                if feed.camera_id not in self.active_feeds:
                    # Test feed before adding
                    test_result = await self.test_feed_connection(feed)
                    
                    if test_result['status'] == 'success':
                        await self.add_camera_feed(feed)
                        new_feeds_count += 1
                        logger.info(f"Auto-registered new feed: {feed.camera_id}")
                    else:
                        logger.warning(f"Failed to register feed {feed.camera_id}: test failed")
            
            logger.info(f"Auto-discovery completed: {new_feeds_count} new feeds registered")
            return new_feeds_count
            
        except Exception as e:
            logger.error(f"Error in auto-discovery: {e}")
            return 0
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics about camera feeds and performance"""
        stats = self.get_feed_statistics()
        
        # Add enhanced metrics
        quality_scores = [feed.quality_score for feed in self.active_feeds.values()]
        failure_counts = [feed.failure_count for feed in self.active_feeds.values()]
        
        stats.update({
            'average_quality_score': np.mean(quality_scores) if quality_scores else 0.0,
            'total_failures': sum(failure_counts),
            'feeds_with_failures': sum(1 for count in failure_counts if count > 0),
            'data_sources': list(set(feed.data_source for feed in self.active_feeds.values())),
            'coverage_areas': list(set(feed.coverage_area for feed in self.active_feeds.values() if feed.coverage_area)),
            'feed_formats': list(set(feed.feed_format for feed in self.active_feeds.values())),
            'quality_threshold': self.quality_threshold,
            'max_failure_count': self.max_failure_count
        })
        
        return stats
