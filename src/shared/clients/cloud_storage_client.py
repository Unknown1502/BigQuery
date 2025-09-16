"""
Cloud Storage Client - Advanced file operations and data management
Handles image storage, model artifacts, and data pipeline operations
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, BinaryIO
from datetime import datetime, timedelta, timezone
import io
import base64
from google.cloud import storage
from google.cloud.storage import Blob, Bucket
from google.api_core import exceptions

from ..config.settings import settings
from ..utils.logging_utils import get_logger
from ..utils.error_handling import PricingIntelligenceError, ErrorCategory, ErrorSeverity, handle_exceptions

# Setup logging
logger = get_logger(__name__)

class CloudStorageClient:
    """
    Advanced Cloud Storage client for data and model management
    Handles street imagery, ML models, and processing artifacts
    """
    
    def __init__(self):
        """Initialize Cloud Storage client with project configuration"""
        self.project_id = settings.database.project_id
        self.client = storage.Client(project=self.project_id)
        
        # Bucket configurations
        self.buckets = {
            'street_imagery': settings.cloud_storage.street_imagery_bucket,
            'ml_models': settings.cloud_storage.model_artifacts_bucket,
            'data_processing': settings.cloud_storage.data_processing_bucket,
            'backups': settings.cloud_storage.backup_bucket,
            'analytics': settings.cloud_storage.analytics_bucket
        }
        
        # Initialize buckets
        self._initialize_buckets()
        
        logger.info(f"Cloud Storage client initialized for project {self.project_id}")
    
    def _initialize_buckets(self):
        """Initialize and configure storage buckets"""
        for bucket_name, bucket_id in self.buckets.items():
            try:
                bucket = self.client.bucket(bucket_id)
                if not bucket.exists():
                    logger.info(f"Creating bucket: {bucket_id}")
                    bucket = self.client.create_bucket(bucket_id, location=settings.database.location)
                    
                    # Set lifecycle policies
                    self._set_lifecycle_policy(bucket, bucket_name)
                    
                logger.debug(f"Bucket {bucket_id} is ready")
                
            except Exception as e:
                logger.warning(f"Could not initialize bucket {bucket_id}: {e}")
    
    def _set_lifecycle_policy(self, bucket: Bucket, bucket_type: str):
        """Set appropriate lifecycle policies for different bucket types"""
        try:
            if bucket_type == 'street_imagery':
                # Keep raw images for 90 days, then move to coldline
                lifecycle_rule = {
                    'action': {'type': 'SetStorageClass', 'storageClass': 'COLDLINE'},
                    'condition': {'age': 90}
                }
                bucket.lifecycle_rules = [lifecycle_rule]
                
            elif bucket_type == 'data_processing':
                # Delete temporary processing files after 7 days
                lifecycle_rule = {
                    'action': {'type': 'Delete'},
                    'condition': {'age': 7}
                }
                bucket.lifecycle_rules = [lifecycle_rule]
                
            elif bucket_type == 'backups':
                # Move to archive after 30 days, delete after 1 year
                lifecycle_rules = [
                    {
                        'action': {'type': 'SetStorageClass', 'storageClass': 'ARCHIVE'},
                        'condition': {'age': 30}
                    },
                    {
                        'action': {'type': 'Delete'},
                        'condition': {'age': 365}
                    }
                ]
                bucket.lifecycle_rules = lifecycle_rules
            
            bucket.patch()
            logger.debug(f"Lifecycle policy set for bucket {bucket.name}")
            
        except Exception as e:
            logger.warning(f"Could not set lifecycle policy for {bucket.name}: {e}")
    
    @handle_exceptions
    async def upload_image(self, image_data: Union[bytes, BinaryIO], 
                          location_id: str, timestamp: datetime = None,
                          metadata: Dict[str, Any] = None) -> str:
        """
        Upload street imagery with metadata and processing optimization
        
        Args:
            image_data: Image bytes or file-like object
            location_id: Location identifier
            timestamp: Image timestamp (defaults to current time)
            metadata: Additional metadata
            
        Returns:
            GCS URI of uploaded image
        """
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            # Generate structured path
            date_path = timestamp.strftime("%Y/%m/%d")
            filename = f"{location_id}_{timestamp.strftime('%H%M%S')}_{timestamp.microsecond}.jpg"
            blob_name = f"raw/{date_path}/{filename}"
            
            # Get bucket and create blob
            bucket = self.client.bucket(self.buckets['street_imagery'])
            blob = bucket.blob(blob_name)
            
            # Set metadata
            blob.metadata = {
                'location_id': location_id,
                'timestamp': timestamp.isoformat(),
                'upload_time': datetime.utcnow().isoformat(),
                'processing_status': 'pending',
                **(metadata or {})
            }
            
            # Set content type
            blob.content_type = 'image/jpeg'
            
            # Upload image
            if isinstance(image_data, bytes):
                await asyncio.get_event_loop().run_in_executor(
                    None, blob.upload_from_string, image_data
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, blob.upload_from_file, image_data
                )
            
            gcs_uri = f"gs://{bucket.name}/{blob_name}"
            logger.info(f"Image uploaded successfully: {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Error uploading image: {e}")
            raise PricingIntelligenceError(f"Image upload failed: {e}")
    
    @handle_exceptions
    async def get_image(self, gcs_uri: str) -> bytes:
        """
        Retrieve image data from Cloud Storage
        
        Args:
            gcs_uri: GCS URI of the image
            
        Returns:
            Image bytes
        """
        try:
            # Parse GCS URI
            if not gcs_uri.startswith('gs://'):
                raise ValueError(f"Invalid GCS URI: {gcs_uri}")
            
            parts = gcs_uri[5:].split('/', 1)
            bucket_name, blob_name = parts[0], parts[1]
            
            # Get blob and download
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            image_data = await asyncio.get_event_loop().run_in_executor(
                None, blob.download_as_bytes
            )
            
            return image_data
            
        except Exception as e:
            logger.error(f"Error retrieving image {gcs_uri}: {e}")
            raise PricingIntelligenceError(f"Image retrieval failed: {e}")
    
    @handle_exceptions
    async def get_image_base64(self, gcs_uri: str) -> str:
        """
        Retrieve image as base64 encoded string
        
        Args:
            gcs_uri: GCS URI of the image
            
        Returns:
            Base64 encoded image string
        """
        try:
            image_data = await self.get_image(gcs_uri)
            return base64.b64encode(image_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise PricingIntelligenceError(f"Base64 conversion failed: {e}")
    
    @handle_exceptions
    async def list_images(self, location_id: str = None, 
                         start_date: datetime = None, 
                         end_date: datetime = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        List images with optional filtering
        
        Args:
            location_id: Filter by location ID
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results
            
        Returns:
            List of image metadata
        """
        try:
            bucket = self.client.bucket(self.buckets['street_imagery'])
            
            # Build prefix for filtering
            prefix = "raw/"
            if start_date:
                date_prefix = start_date.strftime("%Y/%m/%d")
                prefix += date_prefix
            
            # List blobs
            blobs = await asyncio.get_event_loop().run_in_executor(
                None, lambda: list(bucket.list_blobs(prefix=prefix, max_results=limit))
            )
            
            images = []
            for blob in blobs:
                # Filter by location_id if specified
                if location_id and blob.metadata:
                    if blob.metadata.get('location_id') != location_id:
                        continue
                
                # Filter by date range
                if end_date and blob.metadata:
                    blob_timestamp = datetime.fromisoformat(blob.metadata.get('timestamp', ''))
                    if blob_timestamp > end_date:
                        continue
                
                images.append({
                    'gcs_uri': f"gs://{bucket.name}/{blob.name}",
                    'name': blob.name,
                    'size': blob.size,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'updated': blob.updated.isoformat() if blob.updated else None,
                    'metadata': blob.metadata or {},
                    'content_type': blob.content_type
                })
            
            return images
            
        except Exception as e:
            logger.error(f"Error listing images: {e}")
            raise PricingIntelligenceError(f"Image listing failed: {e}")
    
    @handle_exceptions
    async def update_image_metadata(self, gcs_uri: str, metadata: Dict[str, Any]) -> bool:
        """
        Update image metadata (e.g., processing status, analysis results)
        
        Args:
            gcs_uri: GCS URI of the image
            metadata: Metadata to update
            
        Returns:
            Success status
        """
        try:
            # Parse GCS URI
            parts = gcs_uri[5:].split('/', 1)
            bucket_name, blob_name = parts[0], parts[1]
            
            # Get blob and update metadata
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Merge with existing metadata
            existing_metadata = blob.metadata or {}
            existing_metadata.update(metadata)
            existing_metadata['last_updated'] = datetime.utcnow().isoformat()
            
            blob.metadata = existing_metadata
            
            await asyncio.get_event_loop().run_in_executor(
                None, blob.patch
            )
            
            logger.debug(f"Updated metadata for {gcs_uri}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metadata for {gcs_uri}: {e}")
            return False
    
    @handle_exceptions
    async def upload_model_artifact(self, model_data: Union[bytes, BinaryIO],
                                   model_name: str, version: str,
                                   metadata: Dict[str, Any] = None) -> str:
        """
        Upload ML model artifacts
        
        Args:
            model_data: Model file data
            model_name: Name of the model
            version: Model version
            metadata: Model metadata
            
        Returns:
            GCS URI of uploaded model
        """
        try:
            # Generate model path
            blob_name = f"models/{model_name}/{version}/model.pkl"
            
            # Get bucket and create blob
            bucket = self.client.bucket(self.buckets['ml_models'])
            blob = bucket.blob(blob_name)
            
            # Set metadata
            blob.metadata = {
                'model_name': model_name,
                'version': version,
                'upload_time': datetime.utcnow().isoformat(),
                'model_type': 'pricing_intelligence',
                **(metadata or {})
            }
            
            # Upload model
            if isinstance(model_data, bytes):
                await asyncio.get_event_loop().run_in_executor(
                    None, blob.upload_from_string, model_data
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, blob.upload_from_file, model_data
                )
            
            gcs_uri = f"gs://{bucket.name}/{blob_name}"
            logger.info(f"Model uploaded successfully: {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            raise PricingIntelligenceError(f"Model upload failed: {e}")
    
    @handle_exceptions
    async def backup_data(self, source_data: Dict[str, Any], 
                         backup_type: str, identifier: str) -> str:
        """
        Create data backup with versioning
        
        Args:
            source_data: Data to backup
            backup_type: Type of backup (e.g., 'pricing_config', 'model_state')
            identifier: Unique identifier for the backup
            
        Returns:
            GCS URI of backup
        """
        try:
            # Generate backup path
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            blob_name = f"{backup_type}/{identifier}_{timestamp}.json"
            
            # Get bucket and create blob
            bucket = self.client.bucket(self.buckets['backups'])
            blob = bucket.blob(blob_name)
            
            # Set metadata
            blob.metadata = {
                'backup_type': backup_type,
                'identifier': identifier,
                'timestamp': datetime.utcnow().isoformat(),
                'data_size': len(json.dumps(source_data))
            }
            
            # Upload backup
            backup_json = json.dumps(source_data, indent=2)
            await asyncio.get_event_loop().run_in_executor(
                None, blob.upload_from_string, backup_json
            )
            
            gcs_uri = f"gs://{bucket.name}/{blob_name}"
            logger.info(f"Backup created successfully: {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise PricingIntelligenceError(f"Backup creation failed: {e}")
    
    @handle_exceptions
    async def cleanup_old_files(self, bucket_type: str, days_old: int = 30) -> int:
        """
        Clean up old files based on age
        
        Args:
            bucket_type: Type of bucket to clean
            days_old: Age threshold in days
            
        Returns:
            Number of files deleted
        """
        try:
            if bucket_type not in self.buckets:
                raise ValueError(f"Unknown bucket type: {bucket_type}")
            
            bucket = self.client.bucket(self.buckets[bucket_type])
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # List old blobs
            blobs = await asyncio.get_event_loop().run_in_executor(
                None, lambda: list(bucket.list_blobs())
            )
            
            deleted_count = 0
            for blob in blobs:
                if blob.time_created and blob.time_created.replace(tzinfo=None) < cutoff_date:
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, blob.delete
                        )
                        deleted_count += 1
                        logger.debug(f"Deleted old file: {blob.name}")
                    except Exception as e:
                        logger.warning(f"Could not delete {blob.name}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old files from {bucket_type}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise PricingIntelligenceError(f"Cleanup failed: {e}")
    
    async def get_storage_metrics(self) -> Dict[str, Any]:
        """
        Get storage usage metrics for all buckets
        
        Returns:
            Dict containing storage metrics
        """
        metrics = {
            'buckets': {},
            'total_size': 0,
            'total_objects': 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for bucket_type, bucket_name in self.buckets.items():
            try:
                bucket = self.client.bucket(bucket_name)
                
                # Get bucket info
                blobs = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: list(bucket.list_blobs())
                )
                
                bucket_size = sum(blob.size or 0 for blob in blobs)
                object_count = len(blobs)
                
                metrics['buckets'][bucket_type] = {
                    'name': bucket_name,
                    'size_bytes': bucket_size,
                    'object_count': object_count,
                    'size_gb': round(bucket_size / (1024**3), 2)
                }
                
                metrics['total_size'] += bucket_size
                metrics['total_objects'] += object_count
                
            except Exception as e:
                logger.warning(f"Could not get metrics for bucket {bucket_name}: {e}")
                metrics['buckets'][bucket_type] = {
                    'name': bucket_name,
                    'error': str(e)
                }
        
        metrics['total_size_gb'] = round(metrics['total_size'] / (1024**3), 2)
        
        return metrics
    
    @handle_exceptions
    async def upload_file(self, file_data: Union[bytes, BinaryIO], 
                         bucket_name: str, blob_name: str,
                         content_type: str = None, 
                         metadata: Dict[str, Any] = None) -> str:
        """
        Upload file to specified bucket
        
        Args:
            file_data: File bytes or file-like object
            bucket_name: Target bucket name
            blob_name: Target blob name
            content_type: Content type of the file
            metadata: Additional metadata
            
        Returns:
            GCS URI of uploaded file
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Set metadata
            if metadata:
                blob.metadata = metadata
            
            # Set content type
            if content_type:
                blob.content_type = content_type
            
            # Upload file
            if isinstance(file_data, bytes):
                await asyncio.get_event_loop().run_in_executor(
                    None, blob.upload_from_string, file_data
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, blob.upload_from_file, file_data
                )
            
            gcs_uri = f"gs://{bucket_name}/{blob_name}"
            logger.info(f"File uploaded successfully: {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise PricingIntelligenceError(f"File upload failed: {e}")
    
    @handle_exceptions
    async def upload_bytes(self, data: bytes, bucket_name: str, blob_name: str,
                          content_type: str = None, 
                          metadata: Dict[str, Any] = None) -> str:
        """
        Upload bytes data to specified bucket
        
        Args:
            data: Bytes data to upload
            bucket_name: Target bucket name
            blob_name: Target blob name
            content_type: Content type of the data
            metadata: Additional metadata
            
        Returns:
            GCS URI of uploaded data
        """
        return await self.upload_file(data, bucket_name, blob_name, content_type, metadata)
    
    @handle_exceptions
    async def download_blob_as_bytes(self, bucket_name: str, blob_name: str) -> bytes:
        """
        Download blob as bytes
        
        Args:
            bucket_name: Source bucket name
            blob_name: Source blob name
            
        Returns:
            Blob data as bytes
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            data = await asyncio.get_event_loop().run_in_executor(
                None, blob.download_as_bytes
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading blob {bucket_name}/{blob_name}: {e}")
            raise PricingIntelligenceError(f"Blob download failed: {e}")
    
    @handle_exceptions
    async def cleanup_old_blobs(self, bucket_name: str, days_old: int = 30) -> int:
        """
        Clean up old blobs in specified bucket
        
        Args:
            bucket_name: Bucket name to clean
            days_old: Age threshold in days
            
        Returns:
            Number of blobs deleted
        """
        try:
            bucket = self.client.bucket(bucket_name)
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # List old blobs
            blobs = await asyncio.get_event_loop().run_in_executor(
                None, lambda: list(bucket.list_blobs())
            )
            
            deleted_count = 0
            for blob in blobs:
                if blob.time_created and blob.time_created.replace(tzinfo=None) < cutoff_date:
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, blob.delete
                        )
                        deleted_count += 1
                        logger.debug(f"Deleted old blob: {blob.name}")
                    except Exception as e:
                        logger.warning(f"Could not delete {blob.name}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old blobs from {bucket_name}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during blob cleanup: {e}")
            raise PricingIntelligenceError(
                f"Blob cleanup failed: {e}",
                category=ErrorCategory.EXTERNAL_SERVICE,
                severity=ErrorSeverity.MEDIUM
            )

# Global client instance
_cloud_storage_client = None

def get_cloud_storage_client() -> CloudStorageClient:
    """Get singleton Cloud Storage client instance"""
    global _cloud_storage_client
    if _cloud_storage_client is None:
        _cloud_storage_client = CloudStorageClient()
    return _cloud_storage_client
