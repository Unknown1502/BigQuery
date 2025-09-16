"""
Dataset Download Runner
Simple script to download free datasets for the geospatial pricing system
Run this script to populate your storage with real data at zero cost
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.data.free_dataset_downloader import FreeDatasetDownloader
from src.shared.utils.logging_utils import get_logger

logger = get_logger(__name__)

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dataset_download.log')
        ]
    )

async def download_all_datasets(output_dir: str = "data/real_datasets", verbose: bool = False):
    """Download all free datasets"""
    
    print("🚀 Starting Free Dataset Download Process")
    print("=" * 60)
    print("This will download real datasets at ZERO cost:")
    print("• NYC Taxi Data (100M+ records)")
    print("• Chicago Transportation Data")
    print("• US Census Demographics")
    print("• Weather Data (NOAA)")
    print("• Points of Interest (OpenStreetMap)")
    print("=" * 60)
    
    # Sample locations for weather data
    locations = [
        {'lat': 40.7589, 'lng': -73.9851, 'location_id': 'loc_downtown_001'},
        {'lat': 40.6413, 'lng': -73.7781, 'location_id': 'loc_airport_001'},
        {'lat': 41.8781, 'lng': -87.6298, 'location_id': 'loc_chicago_001'},
        {'lat': 37.7749, 'lng': -122.4194, 'location_id': 'loc_sf_001'},
        {'lat': 34.0522, 'lng': -118.2437, 'location_id': 'loc_la_001'},
    ]
    
    try:
        async with FreeDatasetDownloader(output_dir) as downloader:
            
            print("\n📊 Phase 1: Transportation Data")
            print("-" * 30)
            
            # NYC Taxi Data (largest dataset)
            print("Downloading NYC Taxi Data (3 months, 3 datasets)...")
            nyc_records = await downloader.download_nyc_taxi_data(2024, [1, 2, 3])
            print(f"✓ NYC Taxi: {nyc_records} files downloaded")
            
            # Chicago Taxi Data
            print("\nDownloading Chicago Transportation Data...")
            chicago_records = await downloader.download_chicago_taxi_data(50000)
            print(f"✓ Chicago Taxi: {chicago_records:,} records downloaded")
            
            print("\n🏘️ Phase 2: Demographics & Location Data")
            print("-" * 40)
            
            # US Census Data
            print("Downloading US Census Demographics...")
            census_records = await downloader.download_census_data()
            print(f"✓ Census Data: {census_records:,} records downloaded")
            
            # OpenStreetMap POI Data
            print("\nDownloading Points of Interest (OSM)...")
            poi_records = await downloader.download_osm_poi_data(["New York", "Chicago", "San Francisco"])
            print(f"✓ POI Data: {poi_records:,} records downloaded")
            
            print("\n🌤️ Phase 3: Weather Data")
            print("-" * 25)
            
            # Weather Data
            print("Downloading Weather Data (NOAA)...")
            weather_records = await downloader.download_noaa_weather_data(locations, 30)
            print(f"✓ Weather Data: {weather_records:,} records downloaded")
            
            # Get final summary
            summary = downloader.get_download_summary()
            
            print("\n" + "=" * 60)
            print("🎉 DOWNLOAD COMPLETE!")
            print("=" * 60)
            print(f"📈 Total Datasets: {summary['total_datasets']}")
            print(f"✅ Completed: {summary['completed']}")
            print(f"❌ Failed: {summary['failed']}")
            print(f"📊 Total Records: {summary['total_records']:,}")
            print(f"💰 Total Cost: $0.00 (All free datasets!)")
            
            print("\n📁 Dataset Details:")
            for name, details in summary['datasets'].items():
                status_icon = "✅" if details['status'] == 'completed' else "❌"
                print(f"  {status_icon} {name.replace('_', ' ').title()}: {details['records']:,} records")
                if details['error']:
                    print(f"    ⚠️ Error: {details['error']}")
            
            print(f"\n📂 Files saved to: {Path(output_dir).absolute()}")
            print("\n🚀 Next Steps:")
            print("1. Deploy storage buckets: scripts/deploy_storage_buckets.bat")
            print("2. Upload data to GCS: python scripts/data/upload_to_gcs.py")
            print("3. Validate data quality: python scripts/validation/validate_datasets.py")
            
            return summary
            
    except Exception as e:
        logger.error(f"Download process failed: {e}")
        print(f"\n❌ Download failed: {e}")
        print("Check dataset_download.log for details")
        return None

async def download_specific_dataset(dataset_name: str, output_dir: str = "data/real_datasets"):
    """Download a specific dataset only"""
    
    locations = [
        {'lat': 40.7589, 'lng': -73.9851, 'location_id': 'loc_downtown_001'},
        {'lat': 41.8781, 'lng': -87.6298, 'location_id': 'loc_chicago_001'},
    ]
    
    async with FreeDatasetDownloader(output_dir) as downloader:
        
        if dataset_name == "nyc_taxi":
            print("Downloading NYC Taxi Data...")
            result = await downloader.download_nyc_taxi_data(2024, [1, 2, 3])
            print(f"Downloaded {result} files")
            
        elif dataset_name == "chicago_taxi":
            print("Downloading Chicago Taxi Data...")
            result = await downloader.download_chicago_taxi_data(50000)
            print(f"Downloaded {result:,} records")
            
        elif dataset_name == "census":
            print("Downloading Census Data...")
            result = await downloader.download_census_data()
            print(f"Downloaded {result:,} records")
            
        elif dataset_name == "weather":
            print("Downloading Weather Data...")
            result = await downloader.download_noaa_weather_data(locations, 30)
            print(f"Downloaded {result:,} records")
            
        elif dataset_name == "poi":
            print("Downloading POI Data...")
            result = await downloader.download_osm_poi_data(["New York", "Chicago"])
            print(f"Downloaded {result:,} records")
            
        else:
            print(f"Unknown dataset: {dataset_name}")
            print("Available datasets: nyc_taxi, chicago_taxi, census, weather, poi")

def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description="Download free datasets for geospatial pricing system")
    parser.add_argument("--dataset", "-d", help="Download specific dataset only", 
                       choices=["nyc_taxi", "chicago_taxi", "census", "weather", "poi"])
    parser.add_argument("--output", "-o", default="data/real_datasets", 
                       help="Output directory (default: data/real_datasets)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.dataset:
            # Download specific dataset
            asyncio.run(download_specific_dataset(args.dataset, args.output))
        else:
            # Download all datasets
            asyncio.run(download_all_datasets(args.output, args.verbose))
            
    except KeyboardInterrupt:
        print("\n⏹️ Download cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n💥 Unexpected error: {e}")
        print("Check dataset_download.log for details")

if __name__ == "__main__":
    main()
