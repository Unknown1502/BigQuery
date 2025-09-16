"""
Street Image Dataset Collector
Comprehensive collector for building street camera image datasets from multiple sources
Integrates with Google Street View, Mapillary, Open Images, and real camera feeds
"""

import asyncio
import aiohttp
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import json
import os
import hashlib
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, field
import logging
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import tarfile

from src.shared.utils.logging_utils import get_logger
from src.shared.clients.cloud_storage_client import CloudStorageClient
from src.shared.config.gcp_config import GCPConfig
from src.shared.utils.error_handling import RetryableError
import sys
import importlib.util
from pathlib import Path

# Import street_camera_connector using importlib to handle hyphenated directory
_connector_path = Path(__file__).parent.parent / "connectors" / "street_camera_connector.py"
_spec = importlib.util.spec_from_file_location("street_camera_connector", _connector_path)
_street_camera_connector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_street_camera_connector)

# Import the classes we need
StreetCameraConnector = _street_camera_connector.StreetCameraConnector
ImageCapture = _street_camera_connector.ImageCapture

logger = get_logger(__name__)

@dataclass
class DatasetImage:
    """Dataset image with comprehensive metadata"""
    image_id: str
    source: str  # 'street_view', 'mapillary', 'open_images', 'camera_feed'
    location_id: str
    image_path: str
    image_url: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    timestamp: Optional[datetime] = None
    heading: Optional[int] = None
    pitch: Optional[int] = None
    fov: Optional[int] = None
    resolution: Optional[Tuple[int, int]] = None
    file_size: Optional[int] = None
    image_hash: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    weather_conditions: Optional[Dict[str, Any]] = None
    time_of_day: Optional[str] = None
    crowd_level: Optional[str] = None
    traffic_level: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetCollection:
    """Collection of dataset images with metadata"""
    collection_id: str
    name: str
    description: str
    created_at: datetime
    total_images: int
    sources: List[str]
    locations: List[str]
    size_bytes: int
    images: List[DatasetImage] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

class StreetImageDatasetCollector:
    """
    Comprehensive collector for street camera image datasets
    """
    
    def __init__(self, config: GCPConfig):
        self.config = config
        self.storage_client = CloudStorageClient(config)
        self.camera_connector = StreetCameraConnector(config)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Dataset configuration
        self.dataset_dir = Path("data/street_image_datasets")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # API configurations
        self.google_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.mapillary_token = os.getenv('MAPILLARY_ACCESS_TOKEN')
        
        # Collection settings
        self.max_images_per_location = 50
        self.image_quality_threshold = 0.6
        self.max_concurrent_downloads = 10
        
        # Initialize collections
        self.collections: Dict[str, DatasetCollection] = {}
        
    async def initialize(self):
        """Initialize the collector"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(limit=50)
        )
        
        await self.camera_connector.initialize()
        
        logger.info("Street image dataset collector initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        
        await self.camera_connector.cleanup()
    
    async def create_comprehensive_dataset(
        self, 
        locations: List[Dict[str, Any]], 
        collection_name: str,
        sources: List[str] = None
    ) -> DatasetCollection:
        """Create a comprehensive street image dataset from multiple sources"""
        
        if sources is None:
            sources = ['google_street_view', 'mapillary', 'camera_feeds']
        
        collection_id = f"dataset_{int(datetime.now().timestamp())}"
        
        logger.info(f"Creating comprehensive dataset '{collection_name}' with {len(locations)} locations")
        
        # Initialize collection
        collection = DatasetCollection(
            collection_id=collection_id,
            name=collection_name,
            description=f"Comprehensive street image dataset with {len(locations)} locations",
            created_at=datetime.now(timezone.utc),
            total_images=0,
            sources=sources,
            locations=[loc.get('location_id', f"loc_{i}") for i, loc in enumerate(locations)],
            size_bytes=0
        )
        
        all_images = []
        
        # Collect from each source
        for source in sources:
            try:
                logger.info(f"Collecting images from {source}")
                
                if source == 'google_street_view':
                    images = await self._collect_google_street_view(locations, collection_id)
                elif source == 'mapillary':
                    images = await self._collect_mapillary_images(locations, collection_id)
                elif source == 'open_images':
                    images = await self._collect_open_images(locations, collection_id)
                elif source == 'camera_feeds':
                    images = await self._collect_camera_feed_images(locations, collection_id)
                else:
                    logger.warning(f"Unknown source: {source}")
                    continue
                
                all_images.extend(images)
                logger.info(f"Collected {len(images)} images from {source}")
                
            except Exception as e:
                logger.error(f"Error collecting from {source}: {e}")
                continue
        
        # Process and enhance images
        processed_images = await self._process_dataset_images(all_images)
        
        # Update collection
        collection.images = processed_images
        collection.total_images = len(processed_images)
        collection.size_bytes = sum(img.file_size or 0 for img in processed_images)
        collection.statistics = self._calculate_dataset_statistics(processed_images)
        
        # Save collection
        self.collections[collection_id] = collection
        await self._save_collection_metadata(collection)
        
        logger.info(f"Created dataset '{collection_name}' with {collection.total_images} images ({collection.size_bytes / 1024 / 1024:.1f} MB)")
        
        return collection
    
    async def _collect_google_street_view(
        self, 
        locations: List[Dict[str, Any]], 
        collection_id: str
    ) -> List[DatasetImage]:
        """Collect Google Street View images"""
        
        if not self.google_api_key:
            logger.warning("Google Maps API key not available")
            return []
        
        images = []
        images_dir = self.dataset_dir / collection_id / "google_street_view"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for location in locations:
            lat, lng = location['lat'], location['lng']
            location_id = location.get('location_id', f"loc_{lat}_{lng}")
            
            # Multiple angles and times for comprehensive coverage
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
            pitches = [-10, 0, 10]
            
            for heading in angles:
                for pitch in pitches:
                    try:
                        # Generate image
                        image_data = await self._fetch_street_view_image(
                            lat, lng, heading, pitch
                        )
                        
                        if image_data:
                            # Save image
                            filename = f"{location_id}_gsv_h{heading}_p{pitch}.jpg"
                            image_path = images_dir / filename
                            
                            with open(image_path, 'wb') as f:
                                f.write(image_data)
                            
                            # Create dataset image
                            dataset_image = DatasetImage(
                                image_id=f"gsv_{location_id}_{heading}_{pitch}",
                                source="google_street_view",
                                location_id=location_id,
                                image_path=str(image_path),
                                coordinates={'lat': lat, 'lng': lng},
                                timestamp=datetime.now(timezone.utc),
                                heading=heading,
                                pitch=pitch,
                                fov=90,
                                file_size=len(image_data),
                                image_hash=hashlib.md5(image_data).hexdigest(),
                                metadata={
                                    'source_api': 'google_street_view',
                                    'collection_method': 'api_download'
                                }
                            )
                            
                            images.append(dataset_image)
                        
                        # Rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.warning(f"Error collecting Street View image for {location_id}: {e}")
                        continue
        
        return images
    
    async def _fetch_street_view_image(
        self, 
        lat: float, 
        lng: float, 
        heading: int, 
        pitch: int
    ) -> Optional[bytes]:
        """Fetch a single Street View image"""
        
        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            'size': '640x640',
            'location': f"{lat},{lng}",
            'heading': heading,
            'pitch': pitch,
            'fov': 90,
            'key': self.google_api_key
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.warning(f"Street View API returned {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching Street View image: {e}")
            return None
    
    async def _collect_mapillary_images(
        self, 
        locations: List[Dict[str, Any]], 
        collection_id: str
    ) -> List[DatasetImage]:
        """Collect Mapillary street-level images"""
        
        if not self.mapillary_token:
            logger.warning("Mapillary access token not available")
            return []
        
        images = []
        images_dir = self.dataset_dir / collection_id / "mapillary"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for location in locations:
            lat, lng = location['lat'], location['lng']
            location_id = location.get('location_id', f"loc_{lat}_{lng}")
            
            try:
                # Search for images near location
                url = "https://graph.mapillary.com/images"
                params = {
                    'access_token': self.mapillary_token,
                    'fields': 'id,thumb_2048_url,computed_geometry,compass_angle,captured_at',
                    'bbox': f"{lng-0.01},{lat-0.01},{lng+0.01},{lat+0.01}",
                    'limit': min(self.max_images_per_location, 20)
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for img_data in data.get('data', []):
                            try:
                                # Download image
                                img_url = img_data['thumb_2048_url']
                                
                                async with self.session.get(img_url) as img_response:
                                    if img_response.status == 200:
                                        image_data = await img_response.read()
                                        
                                        # Save image
                                        filename = f"{location_id}_mapillary_{img_data['id']}.jpg"
                                        image_path = images_dir / filename
                                        
                                        with open(image_path, 'wb') as f:
                                            f.write(image_data)
                                        
                                        # Create dataset image
                                        coords = img_data['computed_geometry']['coordinates']
                                        dataset_image = DatasetImage(
                                            image_id=f"mapillary_{img_data['id']}",
                                            source="mapillary",
                                            location_id=location_id,
                                            image_path=str(image_path),
                                            image_url=img_url,
                                            coordinates={'lat': coords[1], 'lng': coords[0]},
                                            timestamp=datetime.fromisoformat(img_data.get('captured_at', '').replace('Z', '+00:00')) if img_data.get('captured_at') else None,
                                            heading=img_data.get('compass_angle'),
                                            file_size=len(image_data),
                                            image_hash=hashlib.md5(image_data).hexdigest(),
                                            metadata={
                                                'mapillary_id': img_data['id'],
                                                'source_api': 'mapillary',
                                                'collection_method': 'api_download'
                                            }
                                        )
                                        
                                        images.append(dataset_image)
                                        
                            except Exception as e:
                                logger.warning(f"Error downloading Mapillary image: {e}")
                                continue
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error collecting Mapillary images for {location_id}: {e}")
                continue
        
        return images
    
    async def _collect_open_images(
        self, 
        locations: List[Dict[str, Any]], 
        collection_id: str
    ) -> List[DatasetImage]:
        """Collect relevant images from Open Images dataset"""
        
        images = []
        images_dir = self.dataset_dir / collection_id / "open_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download Open Images metadata
            metadata_url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
            
            # For demo purposes, we'll simulate this
            # In production, you would download and filter the actual Open Images dataset
            logger.info("Open Images integration would download relevant street scene images")
            
            # This would involve:
            # 1. Downloading Open Images metadata
            # 2. Filtering for street scene categories (car, person, building, etc.)
            # 3. Downloading relevant images
            # 4. Processing and organizing them
            
        except Exception as e:
            logger.error(f"Error collecting Open Images: {e}")
        
        return images
    
    async def _collect_camera_feed_images(
        self, 
        locations: List[Dict[str, Any]], 
        collection_id: str
    ) -> List[DatasetImage]:
        """Collect images from real camera feeds"""
        
        images = []
        images_dir = self.dataset_dir / collection_id / "camera_feeds"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Discover and register camera feeds for locations
            await self.camera_connector.auto_discover_and_register_feeds()
            
            # Collect images from active feeds
            capture_count = 0
            max_captures = len(locations) * 5  # 5 images per location
            
            async for capture in self.camera_connector.start_capture_streams():
                if capture_count >= max_captures:
                    break
                
                try:
                    # Save captured image
                    filename = f"{capture.camera_id}_{int(capture.capture_timestamp.timestamp())}.jpg"
                    image_path = images_dir / filename
                    
                    with open(image_path, 'wb') as f:
                        f.write(capture.image_data)
                    
                    # Create dataset image
                    dataset_image = DatasetImage(
                        image_id=f"camera_{capture.camera_id}_{int(capture.capture_timestamp.timestamp())}",
                        source="camera_feed",
                        location_id=capture.location_id,
                        image_path=str(image_path),
                        timestamp=capture.capture_timestamp,
                        file_size=len(capture.image_data),
                        image_hash=capture.image_hash,
                        metadata={
                            'camera_id': capture.camera_id,
                            'camera_metadata': capture.metadata,
                            'collection_method': 'real_time_capture'
                        }
                    )
                    
                    images.append(dataset_image)
                    capture_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing camera capture: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error collecting camera feed images: {e}")
        
        return images
    
    async def _process_dataset_images(self, images: List[DatasetImage]) -> List[DatasetImage]:
        """Process and enhance dataset images"""
        
        processed_images = []
        
        for image in images:
            try:
                # Load and analyze image
                if os.path.exists(image.image_path):
                    img_array = cv2.imread(image.image_path)
                    
                    if img_array is not None:
                        # Update resolution
                        image.resolution = (img_array.shape[1], img_array.shape[0])
                        
                        # Calculate quality score
                        image.quality_score = self._calculate_image_quality_score(img_array)
                        
                        # Extract time of day
                        image.time_of_day = self._extract_time_of_day(image.timestamp)
                        
                        # Add labels based on analysis
                        image.labels = self._generate_image_labels(img_array, image)
                        
                        # Filter by quality
                        if image.quality_score >= self.image_quality_threshold:
                            processed_images.append(image)
                        else:
                            logger.debug(f"Filtered low quality image: {image.image_id}")
                    
            except Exception as e:
                logger.warning(f"Error processing image {image.image_id}: {e}")
                continue
        
        return processed_images
    
    def _calculate_image_quality_score(self, image: np.ndarray) -> float:
        """Calculate image quality score"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = gray.std()
            
            # Normalize and combine
            sharpness_score = min(1.0, sharpness / 1000.0)
            brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
            contrast_score = min(1.0, contrast / 64.0)
            
            quality_score = (sharpness_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    def _extract_time_of_day(self, timestamp: Optional[datetime]) -> Optional[str]:
        """Extract time of day category"""
        if not timestamp:
            return None
        
        hour = timestamp.hour
        
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _generate_image_labels(self, image: np.ndarray, dataset_image: DatasetImage) -> List[str]:
        """Generate labels for the image"""
        labels = []
        
        # Basic labels based on source
        if dataset_image.source == "google_street_view":
            labels.extend(["street_view", "urban", "outdoor"])
        elif dataset_image.source == "mapillary":
            labels.extend(["street_level", "urban", "outdoor"])
        elif dataset_image.source == "camera_feed":
            labels.extend(["real_time", "traffic_camera", "outdoor"])
        
        # Time-based labels
        if dataset_image.time_of_day:
            labels.append(dataset_image.time_of_day)
        
        # Quality-based labels
        if dataset_image.quality_score > 0.8:
            labels.append("high_quality")
        elif dataset_image.quality_score > 0.6:
            labels.append("medium_quality")
        else:
            labels.append("low_quality")
        
        return labels
    
    def _calculate_dataset_statistics(self, images: List[DatasetImage]) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics"""
        
        if not images:
            return {}
        
        # Basic statistics
        total_images = len(images)
        total_size = sum(img.file_size or 0 for img in images)
        
        # Source distribution
        sources = {}
        for img in images:
            sources[img.source] = sources.get(img.source, 0) + 1
        
        # Quality distribution
        quality_scores = [img.quality_score for img in images if img.quality_score > 0]
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Time distribution
        time_distribution = {}
        for img in images:
            if img.time_of_day:
                time_distribution[img.time_of_day] = time_distribution.get(img.time_of_day, 0) + 1
        
        # Resolution distribution
        resolutions = {}
        for img in images:
            if img.resolution:
                res_key = f"{img.resolution[0]}x{img.resolution[1]}"
                resolutions[res_key] = resolutions.get(res_key, 0) + 1
        
        # Location distribution
        locations = {}
        for img in images:
            locations[img.location_id] = locations.get(img.location_id, 0) + 1
        
        return {
            'total_images': total_images,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / 1024 / 1024,
            'average_quality_score': avg_quality,
            'source_distribution': sources,
            'time_distribution': time_distribution,
            'resolution_distribution': resolutions,
            'location_distribution': locations,
            'unique_locations': len(locations),
            'date_range': {
                'earliest': min((img.timestamp for img in images if img.timestamp), default=None),
                'latest': max((img.timestamp for img in images if img.timestamp), default=None)
            }
        }
    
    async def _save_collection_metadata(self, collection: DatasetCollection):
        """Save collection metadata to file"""
        
        metadata_path = self.dataset_dir / collection.collection_id / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        metadata = {
            'collection_id': collection.collection_id,
            'name': collection.name,
            'description': collection.description,
            'created_at': collection.created_at.isoformat(),
            'total_images': collection.total_images,
            'sources': collection.sources,
            'locations': collection.locations,
            'size_bytes': collection.size_bytes,
            'statistics': collection.statistics,
            'images': [
                {
                    'image_id': img.image_id,
                    'source': img.source,
                    'location_id': img.location_id,
                    'image_path': img.image_path,
                    'coordinates': img.coordinates,
                    'timestamp': img.timestamp.isoformat() if img.timestamp else None,
                    'heading': img.heading,
                    'pitch': img.pitch,
                    'resolution': img.resolution,
                    'file_size': img.file_size,
                    'image_hash': img.image_hash,
                    'labels': img.labels,
                    'quality_score': img.quality_score,
                    'time_of_day': img.time_of_day,
                    'metadata': img.metadata
                }
                for img in collection.images
            ]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved collection metadata: {metadata_path}")
    
    async def export_dataset(
        self, 
        collection_id: str, 
        export_format: str = 'coco',
        output_path: Optional[str] = None
    ) -> str:
        """Export dataset in specified format"""
        
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} not found")
        
        collection = self.collections[collection_id]
        
        if output_path is None:
            output_path = str(self.dataset_dir / f"{collection.name}_{export_format}.zip")
        
        if export_format == 'coco':
            return await self._export_coco_format(collection, output_path)
        elif export_format == 'yolo':
            return await self._export_yolo_format(collection, output_path)
        elif export_format == 'csv':
            return await self._export_csv_format(collection, output_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    async def _export_coco_format(self, collection: DatasetCollection, output_path: str) -> str:
        """Export dataset in COCO format"""
        
        # Create COCO annotation structure
        coco_data = {
            'info': {
                'description': collection.description,
                'version': '1.0',
                'year': collection.created_at.year,
                'contributor': 'Street Camera Dataset Collector',
                'date_created': collection.created_at.isoformat()
            },
            'licenses': [
                {
                    'id': 1,
                    'name': 'Custom License',
                    'url': ''
                }
            ],
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'person', 'supercategory': 'person'},
                {'id': 2, 'name': 'vehicle', 'supercategory': 'vehicle'},
                {'id': 3, 'name': 'building', 'supercategory': 'object'},
                {'id': 4, 'name': 'street', 'supercategory': 'scene'}
            ]
        }
        
        # Add images
        for i, img in enumerate(collection.images):
            coco_data['images'].append({
                'id': i + 1,
                'width': img.resolution[0] if img.resolution else 640,
                'height': img.resolution[1] if img.resolution else 640,
                'file_name': os.path.basename(img.image_path),
                'license': 1,
                'flickr_url': img.image_url or '',
                'coco_url': '',
                'date_captured': img.timestamp.isoformat() if img.timestamp else ''
            })
        
        # Create zip file with images and annotations
        with zipfile.ZipFile(output_path, 'w') as zipf:
            # Add annotation file
            zipf.writestr('annotations.json', json.dumps(coco_data, indent=2))
            
            # Add images
            for img in collection.images:
                if os.path.exists(img.image_path):
                    zipf.write(img.image_path, f"images/{os.path.basename(img.image_path)}")
        
        logger.info(f"Exported COCO dataset: {output_path}")
        return output_path
    
    async def _export_yolo_format(self, collection: DatasetCollection, output_path: str) -> str:
        """Export dataset in YOLO format"""
        
        with zipfile.ZipFile(output_path, 'w') as zipf:
            # Add images and create label files
            for img in collection.images:
                if os.path.exists(img.image_path):
                    # Add image
                    zipf.write(img.image_path, f"images/{os.path.basename(img.image_path)}")
                    
                    # Create empty label file (would contain bounding box annotations)
                    label_filename = os.path.basename(img.image_path).replace('.jpg', '.txt')
                    zipf.writestr(f"labels/{label_filename}", "")
            
            # Add dataset configuration
            config = {
                'train': 'images/',
                'val': 'images/',
                'nc': 4,
                'names': ['person', 'vehicle', 'building', 'street']
            }
            zipf.writestr('dataset.yaml', json.dumps(config, indent=2))
        
        logger.info(f"Exported YOLO dataset: {output_path}")
        return output_path
    
    async def _export_csv_format(self, collection: DatasetCollection, output_path: str) -> str:
        """Export dataset metadata in CSV format"""
        
        # Create DataFrame from images
        data = []
        for img in collection.images:
            data.append({
                'image_id': img.image_id,
                'source': img.source,
                'location_id': img.location_id,
                'image_path': img.image_path,
                'latitude': img.coordinates.get('lat') if img.coordinates else None,
                'longitude': img.coordinates.get('lng') if img.coordinates else None,
                'timestamp': img.timestamp.isoformat() if img.timestamp else None,
                'heading': img.heading,
                'pitch': img.pitch,
                'width': img.resolution[0] if img.resolution else None,
                'height': img.resolution[1] if img.resolution else None,
                'file_size': img.file_size,
                'quality_score': img.quality_score,
                'time_of_day': img.time_of_day,
                'labels': ','.join(img.labels)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported CSV dataset: {output_path}")
        return output_path
    
    def get_collection_summary(self, collection_id: str) -> Dict[str, Any]:
        """Get summary information about a collection"""
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} not found")
        
        collection = self.collections[collection_id]
        
        return {
            'collection_id': collection.collection_id,
            'name': collection.name,
            'description': collection.description,
            'created_at': collection.created_at.isoformat(),
            'total_images': collection.total_images,
            'sources': collection.sources,
            'locations': collection.locations,
            'size_bytes': collection.size_bytes,
            'size_mb': collection.size_bytes / 1024 / 1024,
            'statistics': collection.statistics
        }
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all available collections"""
        return [
            {
                'collection_id': collection.collection_id,
                'name': collection.name,
                'total_images': collection.total_images,
                'size_mb': collection.size_bytes / 1024 / 1024,
                'created_at': collection.created_at.isoformat(),
                'sources': collection.sources
            }
            for collection in self.collections.values()
        ]
    
    async def validate_dataset_quality(self, collection_id: str) -> Dict[str, Any]:
        """Validate the quality of a dataset collection"""
        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} not found")
        
        collection = self.collections[collection_id]
        
        validation_results = {
            'collection_id': collection_id,
            'total_images': len(collection.images),
            'validation_timestamp': datetime.now(timezone.utc).isoformat(),
            'issues': [],
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Check for missing files
        missing_files = 0
        for img in collection.images:
            if not os.path.exists(img.image_path):
                missing_files += 1
                validation_results['issues'].append(f"Missing file: {img.image_path}")
        
        # Check quality distribution
        quality_scores = [img.quality_score for img in collection.images if img.quality_score > 0]
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            min_quality = min(quality_scores)
            
            validation_results['quality_metrics'] = {
                'average_quality': avg_quality,
                'minimum_quality': min_quality,
                'low_quality_count': sum(1 for q in quality_scores if q < 0.5),
                'high_quality_count': sum(1 for q in quality_scores if q > 0.8)
            }
            
            if avg_quality < 0.6:
                validation_results['recommendations'].append("Consider filtering out low quality images")
            
            if min_quality < 0.3:
                validation_results['recommendations'].append("Some images have very low quality scores")
        
        # Check source diversity
        sources = set(img.source for img in collection.images)
        if len(sources) < 2:
            validation_results['recommendations'].append("Consider adding images from multiple sources for diversity")
        
        # Check location coverage
        locations = set(img.location_id for img in collection.images)
        images_per_location = len(collection.images) / len(locations) if locations else 0
        
        if images_per_location < 5:
            validation_results['recommendations'].append("Consider collecting more images per location")
        
        validation_results['summary'] = {
            'missing_files': missing_files,
            'source_diversity': len(sources),
            'location_coverage': len(locations),
            'images_per_location': images_per_location,
            'overall_health': 'good' if missing_files == 0 and avg_quality > 0.6 else 'needs_attention'
        }
        
        return validation_results
