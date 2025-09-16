"""
Upload to Google Cloud Storage
Uploads downloaded datasets to GCS buckets for the geospatial pricing system
Handles batch uploads with progress tracking and error handling
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json
import hashlib

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.shared.clients.cloud_storage_client import get_cloud_storage_client
from src.shared.utils.logging_utils import get_logger

logger = get_logger(__name__)

class GCSUploader:
    """
    Uploads local datasets to Google Cloud Storage buckets
    """
    
    def __init__(self, local_data_dir: str = "data/real_datasets"):
        """Initialize uploader"""
        self.local_data_dir = Path(local_data_dir)
        self.storage_client = get_cloud_storage_client()
        self.upload_results = {}
        
        logger.info(f"GCS uploader initialized for: {self.local_data_dir}")
    
    def calculate_file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of file for integrity checking"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {filepath}: {e}")
            return ""
    
    async def upload_file_to_bucket(self, local_path: Path, bucket_type: str, 
                                   remote_path: str) -> Dict[str, Any]:
        """Upload a single file to GCS bucket"""
        
        result = {
            'local_path': str(local_path),
            'bucket_type': bucket_type,
            'remote_path': remote_path,
            'success': False,
            'size_mb': 0,
            'upload_time': None,
            'error': None,
            'file_hash': None
        }
        
        try:
            if not local_path.exists():
                result['error'] = "Local file does not exist"
                return result
            
            # Calculate file info
            file_size = local_path.stat().st_size
            result['size_mb'] = round(file_size / 1024 / 1024, 2)
            result['file_hash'] = self.calculate_file_hash(local_path)
            
            logger.info(f"Uploading {local_path.name} ({result['size_mb']} MB) to {bucket_type}/{remote_path}")
            
            start_time = datetime.now()
            
            # Upload to GCS
            success = await self.storage_client.upload_file(
                local_path=str(local_path),
                bucket_type=bucket_type,
                remote_path=remote_path
            )
            
            end_time = datetime.now()
            upload_duration = (end_time - start_time).total_seconds()
            
            result['success'] = success
            result['upload_time'] = upload_duration
            
            if success:
                logger.info(f"✓ Uploaded {local_path.name} in {upload_duration:.1f}s")
            else:
                result['error'] = "Upload failed (unknown reason)"
                logger.error(f"✗ Failed to upload {local_path.name}")
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error uploading {local_path}: {e}")
        
        return result
    
    async def upload_transportation_data(self) -> Dict[str, Any]:
        """Upload transportation datasets"""
        logger.info("Uploading transportation datasets...")
        
        transport_dir = self.local_data_dir / "transportation"
        results = {
            'category': 'transportation',
            'files': {},
            'summary': {
                'total_files': 0,
                'successful_uploads': 0,
                'failed_uploads': 0,
                'total_size_mb': 0
            }
        }
        
        if not transport_dir.exists():
            logger.warning(f"Transportation directory not found: {transport_dir}")
            return results
        
        # Upload all parquet files in transportation directory
        for parquet_file in transport_dir.glob("*.parquet"):
            remote_path = f"transportation/{parquet_file.name}"
            
            upload_result = await self.upload_file_to_bucket(
                parquet_file, 'data_processing', remote_path
            )
            
            results['files'][parquet_file.name] = upload_result
            results['summary']['total_files'] += 1
            results['summary']['total_size_mb'] += upload_result['size_mb']
            
            if upload_result['success']:
                results['summary']['successful_uploads'] += 1
            else:
                results['summary']['failed_uploads'] += 1
        
        return results
    
    async def upload_demographics_data(self) -> Dict[str, Any]:
        """Upload demographics datasets"""
        logger.info("Uploading demographics datasets...")
        
        demo_dir = self.local_data_dir / "demographics"
        results = {
            'category': 'demographics',
            'files': {},
            'summary': {
                'total_files': 0,
                'successful_uploads': 0,
                'failed_uploads': 0,
                'total_size_mb': 0
            }
        }
        
        if not demo_dir.exists():
            logger.warning(f"Demographics directory not found: {demo_dir}")
            return results
        
        # Upload all parquet files in demographics directory
        for parquet_file in demo_dir.glob("*.parquet"):
            remote_path = f"demographics/{parquet_file.name}"
            
            upload_result = await self.upload_file_to_bucket(
                parquet_file, 'analytics', remote_path
            )
            
            results['files'][parquet_file.name] = upload_result
            results['summary']['total_files'] += 1
            results['summary']['total_size_mb'] += upload_result['size_mb']
            
            if upload_result['success']:
                results['summary']['successful_uploads'] += 1
            else:
                results['summary']['failed_uploads'] += 1
        
        return results
    
    async def upload_weather_data(self) -> Dict[str, Any]:
        """Upload weather datasets"""
        logger.info("Uploading weather datasets...")
        
        weather_dir = self.local_data_dir / "weather"
        results = {
            'category': 'weather',
            'files': {},
            'summary': {
                'total_files': 0,
                'successful_uploads': 0,
                'failed_uploads': 0,
                'total_size_mb': 0
            }
        }
        
        if not weather_dir.exists():
            logger.warning(f"Weather directory not found: {weather_dir}")
            return results
        
        # Upload all parquet files in weather directory
        for parquet_file in weather_dir.glob("*.parquet"):
            remote_path = f"weather/{parquet_file.name}"
            
            upload_result = await self.upload_file_to_bucket(
                parquet_file, 'analytics', remote_path
            )
            
            results['files'][parquet_file.name] = upload_result
            results['summary']['total_files'] += 1
            results['summary']['total_size_mb'] += upload_result['size_mb']
            
            if upload_result['success']:
                results['summary']['successful_uploads'] += 1
            else:
                results['summary']['failed_uploads'] += 1
        
        return results
    
    async def upload_poi_data(self) -> Dict[str, Any]:
        """Upload POI datasets"""
        logger.info("Uploading POI datasets...")
        
        poi_dir = self.local_data_dir / "poi"
        results = {
            'category': 'poi',
            'files': {},
            'summary': {
                'total_files': 0,
                'successful_uploads': 0,
                'failed_uploads': 0,
                'total_size_mb': 0
            }
        }
        
        if not poi_dir.exists():
            logger.warning(f"POI directory not found: {poi_dir}")
            return results
        
        # Upload all parquet files in POI directory
        for parquet_file in poi_dir.glob("*.parquet"):
            remote_path = f"poi/{parquet_file.name}"
            
            upload_result = await self.upload_file_to_bucket(
                parquet_file, 'analytics', remote_path
            )
            
            results['files'][parquet_file.name] = upload_result
            results['summary']['total_files'] += 1
            results['summary']['total_size_mb'] += upload_result['size_mb']
            
            if upload_result['success']:
                results['summary']['successful_uploads'] += 1
            else:
                results['summary']['failed_uploads'] += 1
        
        return results
    
    async def upload_images_data(self) -> Dict[str, Any]:
        """Upload image datasets"""
        logger.info("Uploading image datasets...")
        
        images_dir = self.local_data_dir / "images"
        results = {
            'category': 'images',
            'files': {},
            'summary': {
                'total_files': 0,
                'successful_uploads': 0,
                'failed_uploads': 0,
                'total_size_mb': 0
            }
        }
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return results
        
        # Upload all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.webp']
        
        for image_file in images_dir.rglob("*"):
            if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                # Preserve directory structure in remote path
                relative_path = image_file.relative_to(images_dir)
                remote_path = f"images/{relative_path}"
                
                upload_result = await self.upload_file_to_bucket(
                    image_file, 'street_imagery', str(remote_path)
                )
                
                results['files'][str(relative_path)] = upload_result
                results['summary']['total_files'] += 1
                results['summary']['total_size_mb'] += upload_result['size_mb']
                
                if upload_result['success']:
                    results['summary']['successful_uploads'] += 1
                else:
                    results['summary']['failed_uploads'] += 1
        
        return results
    
    async def upload_all_datasets(self) -> Dict[str, Any]:
        """Upload all datasets to GCS"""
        logger.info("Starting upload of all datasets to GCS...")
        
        upload_results = {
            'upload_timestamp': datetime.now().isoformat(),
            'local_directory': str(self.local_data_dir),
            'categories': {},
            'overall_summary': {
                'total_categories': 0,
                'successful_categories': 0,
                'total_files': 0,
                'successful_uploads': 0,
                'failed_uploads': 0,
                'total_size_mb': 0,
                'total_upload_time': 0,
                'issues': []
            }
        }
        
        # Upload each category
        upload_functions = [
            ('transportation', self.upload_transportation_data),
            ('demographics', self.upload_demographics_data),
            ('weather', self.upload_weather_data),
            ('poi', self.upload_poi_data),
            ('images', self.upload_images_data)
        ]
        
        start_time = datetime.now()
        
        for category_name, upload_func in upload_functions:
            try:
                logger.info(f"Processing {category_name} category...")
                category_result = await upload_func()
                upload_results['categories'][category_name] = category_result
                
                # Update overall summary
                summary = category_result['summary']
                upload_results['overall_summary']['total_categories'] += 1
                upload_results['overall_summary']['total_files'] += summary['total_files']
                upload_results['overall_summary']['successful_uploads'] += summary['successful_uploads']
                upload_results['overall_summary']['failed_uploads'] += summary['failed_uploads']
                upload_results['overall_summary']['total_size_mb'] += summary['total_size_mb']
                
                if summary['successful_uploads'] == summary['total_files'] and summary['total_files'] > 0:
                    upload_results['overall_summary']['successful_categories'] += 1
                
                # Collect issues
                for filename, file_result in category_result['files'].items():
                    if not file_result['success'] and file_result['error']:
                        upload_results['overall_summary']['issues'].append(
                            f"{category_name}/{filename}: {file_result['error']}"
                        )
                
            except Exception as e:
                logger.error(f"Error uploading {category_name}: {e}")
                upload_results['overall_summary']['issues'].append(
                    f"{category_name}: Upload failed - {str(e)}"
                )
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        upload_results['overall_summary']['total_upload_time'] = round(total_time, 1)
        
        self.upload_results = upload_results
        return upload_results
    
    def save_upload_report(self, output_file: str = "gcs_upload_report.json"):
        """Save upload results to file"""
        if not self.upload_results:
            logger.warning("No upload results to save")
            return
        
        output_path = Path(output_file)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.upload_results, f, indent=2)
            
            logger.info(f"Upload report saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving upload report: {e}")
    
    def print_upload_summary(self):
        """Print human-readable upload summary"""
        if not self.upload_results:
            print("No upload results available")
            return
        
        overall = self.upload_results['overall_summary']
        
        print("\n" + "=" * 70)
        print("☁️ GOOGLE CLOUD STORAGE UPLOAD REPORT")
        print("=" * 70)
        print(f"📅 Upload Date: {self.upload_results['upload_timestamp']}")
        print(f"📁 Local Directory: {self.upload_results['local_directory']}")
        print(f"⏱️ Total Time: {overall['total_upload_time']} seconds")
        
        print(f"\n📋 Summary:")
        print(f"  • Categories: {overall['successful_categories']}/{overall['total_categories']}")
        print(f"  • Files: {overall['successful_uploads']}/{overall['total_files']}")
        print(f"  • Size: {overall['total_size_mb']:.1f} MB")
        print(f"  • Failed: {overall['failed_uploads']}")
        
        # Category details
        print(f"\n📂 Category Details:")
        for category_name, category_data in self.upload_results['categories'].items():
            summary = category_data['summary']
            status = "✅" if summary['failed_uploads'] == 0 else "⚠️"
            print(f"  {status} {category_name.title()}: {summary['successful_uploads']}/{summary['total_files']} files ({summary['total_size_mb']:.1f} MB)")
        
        # Issues
        if overall['issues']:
            print(f"\n⚠️ Issues:")
            for issue in overall['issues'][:10]:  # Show first 10 issues
                print(f"  • {issue}")
            if len(overall['issues']) > 10:
                print(f"  ... and {len(overall['issues']) - 10} more issues")
        
        # Overall status
        success_rate = (overall['successful_uploads'] / overall['total_files'] * 100) if overall['total_files'] > 0 else 0
        
        print(f"\n🎯 Overall Status:")
        if success_rate >= 95:
            print("  ✅ EXCELLENT - All files uploaded successfully")
        elif success_rate >= 80:
            print("  ⚠️ GOOD - Most files uploaded successfully")
        elif success_rate >= 50:
            print("  ⚠️ PARTIAL - Some files failed to upload")
        else:
            print("  ❌ POOR - Many files failed to upload")
        
        print(f"  📊 Success Rate: {success_rate:.1f}%")
        print("=" * 70)

async def main():
    """Main upload function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload datasets to Google Cloud Storage")
    parser.add_argument("--data-dir", "-d", default="data/real_datasets", 
                       help="Local data directory to upload")
    parser.add_argument("--output", "-o", default="gcs_upload_report.json",
                       help="Output file for upload report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Check if data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Run the dataset downloader first: python scripts/data/download_datasets.py")
        return
    
    # Run upload
    uploader = GCSUploader(args.data_dir)
    
    print("☁️ Starting upload to Google Cloud Storage...")
    print("This will upload your datasets to GCS buckets for the pricing system.")
    
    try:
        results = await uploader.upload_all_datasets()
        
        # Print summary
        uploader.print_upload_summary()
        
        # Save report
        uploader.save_upload_report(args.output)
        
        print(f"\n📄 Detailed report saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Upload process failed: {e}")
        print(f"\n❌ Upload failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
