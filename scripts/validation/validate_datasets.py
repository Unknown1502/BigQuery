"""
Dataset Validation Script
Validates downloaded datasets for quality, completeness, and format
Ensures data is ready for the geospatial pricing system
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import hashlib

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.shared.utils.logging_utils import get_logger

logger = get_logger(__name__)

class DatasetValidator:
    """
    Validates downloaded datasets for quality and completeness
    """
    
    def __init__(self, data_dir: str = "data/real_datasets"):
        """Initialize validator"""
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        
        logger.info(f"Dataset validator initialized for: {self.data_dir}")
    
    def validate_file_exists(self, filepath: Path) -> bool:
        """Check if file exists and is readable"""
        try:
            return filepath.exists() and filepath.is_file() and filepath.stat().st_size > 0
        except Exception:
            return False
    
    def validate_parquet_file(self, filepath: Path) -> Dict[str, Any]:
        """Validate a parquet file"""
        result = {
            'file': str(filepath),
            'exists': False,
            'readable': False,
            'records': 0,
            'columns': 0,
            'size_mb': 0,
            'issues': [],
            'quality_score': 0.0
        }
        
        try:
            if not self.validate_file_exists(filepath):
                result['issues'].append("File does not exist or is empty")
                return result
            
            result['exists'] = True
            result['size_mb'] = round(filepath.stat().st_size / 1024 / 1024, 2)
            
            # Try to read the parquet file
            df = pd.read_parquet(filepath)
            result['readable'] = True
            result['records'] = len(df)
            result['columns'] = len(df.columns)
            
            # Check for basic data quality issues
            if len(df) == 0:
                result['issues'].append("Dataset is empty")
            
            # Check for missing values
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct > 50:
                result['issues'].append(f"High missing values: {missing_pct:.1f}%")
            elif missing_pct > 20:
                result['issues'].append(f"Moderate missing values: {missing_pct:.1f}%")
            
            # Check for duplicate records
            duplicate_pct = (df.duplicated().sum() / len(df)) * 100
            if duplicate_pct > 10:
                result['issues'].append(f"High duplicate records: {duplicate_pct:.1f}%")
            
            # Calculate quality score
            quality_score = 100
            quality_score -= min(missing_pct, 50)  # Penalize missing values
            quality_score -= min(duplicate_pct, 30)  # Penalize duplicates
            quality_score = max(0, quality_score) / 100
            
            result['quality_score'] = round(quality_score, 2)
            result['sample_data'] = df.head(3).to_dict('records') if len(df) > 0 else []
            
        except Exception as e:
            result['issues'].append(f"Error reading file: {str(e)}")
            logger.error(f"Error validating {filepath}: {e}")
        
        return result
    
    def validate_transportation_data(self) -> Dict[str, Any]:
        """Validate transportation datasets"""
        logger.info("Validating transportation datasets...")
        
        transport_dir = self.data_dir / "transportation"
        results = {
            'category': 'transportation',
            'files': {},
            'summary': {
                'total_files': 0,
                'valid_files': 0,
                'total_records': 0,
                'total_size_mb': 0,
                'issues': []
            }
        }
        
        # Expected NYC taxi files
        expected_files = [
            "yellow_tripdata_2024-01.parquet",
            "yellow_tripdata_2024-02.parquet", 
            "yellow_tripdata_2024-03.parquet",
            "green_tripdata_2024-01.parquet",
            "green_tripdata_2024-02.parquet",
            "green_tripdata_2024-03.parquet",
            "fhv_tripdata_2024-01.parquet",
            "fhv_tripdata_2024-02.parquet",
            "fhv_tripdata_2024-03.parquet",
            "chicago_taxi_2024.parquet"
        ]
        
        for filename in expected_files:
            filepath = transport_dir / filename
            file_result = self.validate_parquet_file(filepath)
            results['files'][filename] = file_result
            
            results['summary']['total_files'] += 1
            if file_result['readable']:
                results['summary']['valid_files'] += 1
                results['summary']['total_records'] += file_result['records']
                results['summary']['total_size_mb'] += file_result['size_mb']
        
        # Check for minimum data requirements
        if results['summary']['total_records'] < 1000000:  # 1M records minimum
            results['summary']['issues'].append("Insufficient transportation data (< 1M records)")
        
        if results['summary']['valid_files'] < 5:
            results['summary']['issues'].append("Missing critical transportation files")
        
        return results
    
    def validate_demographics_data(self) -> Dict[str, Any]:
        """Validate demographics datasets"""
        logger.info("Validating demographics datasets...")
        
        demo_dir = self.data_dir / "demographics"
        results = {
            'category': 'demographics',
            'files': {},
            'summary': {
                'total_files': 0,
                'valid_files': 0,
                'total_records': 0,
                'total_size_mb': 0,
                'issues': []
            }
        }
        
        # Expected demographics files
        expected_files = [
            "census_demographics.parquet"
        ]
        
        for filename in expected_files:
            filepath = demo_dir / filename
            file_result = self.validate_parquet_file(filepath)
            results['files'][filename] = file_result
            
            results['summary']['total_files'] += 1
            if file_result['readable']:
                results['summary']['valid_files'] += 1
                results['summary']['total_records'] += file_result['records']
                results['summary']['total_size_mb'] += file_result['size_mb']
        
        # Check for minimum data requirements
        if results['summary']['total_records'] < 1000:  # 1K records minimum
            results['summary']['issues'].append("Insufficient demographics data")
        
        return results
    
    def validate_weather_data(self) -> Dict[str, Any]:
        """Validate weather datasets"""
        logger.info("Validating weather datasets...")
        
        weather_dir = self.data_dir / "weather"
        results = {
            'category': 'weather',
            'files': {},
            'summary': {
                'total_files': 0,
                'valid_files': 0,
                'total_records': 0,
                'total_size_mb': 0,
                'issues': []
            }
        }
        
        # Expected weather files
        expected_files = [
            "noaa_weather_data.parquet"
        ]
        
        for filename in expected_files:
            filepath = weather_dir / filename
            file_result = self.validate_parquet_file(filepath)
            results['files'][filename] = file_result
            
            results['summary']['total_files'] += 1
            if file_result['readable']:
                results['summary']['valid_files'] += 1
                results['summary']['total_records'] += file_result['records']
                results['summary']['total_size_mb'] += file_result['size_mb']
        
        return results
    
    def validate_poi_data(self) -> Dict[str, Any]:
        """Validate POI datasets"""
        logger.info("Validating POI datasets...")
        
        poi_dir = self.data_dir / "poi"
        results = {
            'category': 'poi',
            'files': {},
            'summary': {
                'total_files': 0,
                'valid_files': 0,
                'total_records': 0,
                'total_size_mb': 0,
                'issues': []
            }
        }
        
        # Expected POI files
        expected_files = [
            "osm_poi_data.parquet"
        ]
        
        for filename in expected_files:
            filepath = poi_dir / filename
            file_result = self.validate_parquet_file(filepath)
            results['files'][filename] = file_result
            
            results['summary']['total_files'] += 1
            if file_result['readable']:
                results['summary']['valid_files'] += 1
                results['summary']['total_records'] += file_result['records']
                results['summary']['total_size_mb'] += file_result['size_mb']
        
        return results
    
    def validate_all_datasets(self) -> Dict[str, Any]:
        """Validate all datasets"""
        logger.info("Starting comprehensive dataset validation...")
        
        # Validate each category
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'data_directory': str(self.data_dir),
            'categories': {},
            'overall_summary': {
                'total_categories': 0,
                'valid_categories': 0,
                'total_files': 0,
                'valid_files': 0,
                'total_records': 0,
                'total_size_mb': 0,
                'overall_quality_score': 0.0,
                'critical_issues': [],
                'recommendations': []
            }
        }
        
        # Validate each category
        categories = [
            ('transportation', self.validate_transportation_data),
            ('demographics', self.validate_demographics_data),
            ('weather', self.validate_weather_data),
            ('poi', self.validate_poi_data)
        ]
        
        category_scores = []
        
        for category_name, validator_func in categories:
            try:
                category_result = validator_func()
                validation_results['categories'][category_name] = category_result
                
                # Update overall summary
                summary = category_result['summary']
                validation_results['overall_summary']['total_categories'] += 1
                validation_results['overall_summary']['total_files'] += summary['total_files']
                validation_results['overall_summary']['valid_files'] += summary['valid_files']
                validation_results['overall_summary']['total_records'] += summary['total_records']
                validation_results['overall_summary']['total_size_mb'] += summary['total_size_mb']
                
                # Calculate category quality score
                if summary['total_files'] > 0:
                    file_success_rate = summary['valid_files'] / summary['total_files']
                    category_score = file_success_rate * 100
                    category_scores.append(category_score)
                    
                    if file_success_rate >= 0.8:
                        validation_results['overall_summary']['valid_categories'] += 1
                
                # Collect critical issues
                if summary['issues']:
                    validation_results['overall_summary']['critical_issues'].extend(
                        [f"{category_name}: {issue}" for issue in summary['issues']]
                    )
                
            except Exception as e:
                logger.error(f"Error validating {category_name}: {e}")
                validation_results['overall_summary']['critical_issues'].append(
                    f"{category_name}: Validation failed - {str(e)}"
                )
        
        # Calculate overall quality score
        if category_scores:
            validation_results['overall_summary']['overall_quality_score'] = round(
                sum(category_scores) / len(category_scores), 1
            )
        
        # Generate recommendations
        overall = validation_results['overall_summary']
        
        if overall['total_records'] < 1000000:
            overall['recommendations'].append("Consider downloading more transportation data for better ML training")
        
        if overall['valid_files'] < overall['total_files']:
            overall['recommendations'].append("Some files failed validation - check download integrity")
        
        if overall['overall_quality_score'] < 70:
            overall['recommendations'].append("Data quality is below recommended threshold - review data sources")
        
        if overall['total_size_mb'] < 100:
            overall['recommendations'].append("Dataset size is small - consider adding more data sources")
        
        self.validation_results = validation_results
        return validation_results
    
    def save_validation_report(self, output_file: str = "dataset_validation_report.json"):
        """Save validation results to file"""
        if not self.validation_results:
            logger.warning("No validation results to save")
            return
        
        output_path = Path(output_file)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            
            logger.info(f"Validation report saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")
    
    def print_validation_summary(self):
        """Print a human-readable validation summary"""
        if not self.validation_results:
            print("No validation results available")
            return
        
        overall = self.validation_results['overall_summary']
        
        print("\n" + "=" * 70)
        print("📊 DATASET VALIDATION REPORT")
        print("=" * 70)
        print(f"📅 Validation Date: {self.validation_results['validation_timestamp']}")
        print(f"📁 Data Directory: {self.validation_results['data_directory']}")
        print(f"📈 Overall Quality Score: {overall['overall_quality_score']}/100")
        
        print(f"\n📋 Summary:")
        print(f"  • Categories: {overall['valid_categories']}/{overall['total_categories']}")
        print(f"  • Files: {overall['valid_files']}/{overall['total_files']}")
        print(f"  • Records: {overall['total_records']:,}")
        print(f"  • Size: {overall['total_size_mb']:.1f} MB")
        
        # Category details
        print(f"\n📂 Category Details:")
        for category_name, category_data in self.validation_results['categories'].items():
            summary = category_data['summary']
            status = "✅" if summary['valid_files'] == summary['total_files'] else "⚠️"
            print(f"  {status} {category_name.title()}: {summary['valid_files']}/{summary['total_files']} files, {summary['total_records']:,} records")
        
        # Issues
        if overall['critical_issues']:
            print(f"\n⚠️ Critical Issues:")
            for issue in overall['critical_issues']:
                print(f"  • {issue}")
        
        # Recommendations
        if overall['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in overall['recommendations']:
                print(f"  • {rec}")
        
        # Overall status
        print(f"\n🎯 Overall Status:")
        if overall['overall_quality_score'] >= 80:
            print("  ✅ EXCELLENT - Data is ready for production use")
        elif overall['overall_quality_score'] >= 60:
            print("  ⚠️ GOOD - Data is usable with minor issues")
        elif overall['overall_quality_score'] >= 40:
            print("  ⚠️ FAIR - Data needs improvement before production use")
        else:
            print("  ❌ POOR - Significant data quality issues detected")
        
        print("=" * 70)

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate downloaded datasets")
    parser.add_argument("--data-dir", "-d", default="data/real_datasets", 
                       help="Data directory to validate")
    parser.add_argument("--output", "-o", default="dataset_validation_report.json",
                       help="Output file for validation report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run validation
    validator = DatasetValidator(args.data_dir)
    
    print("🔍 Starting dataset validation...")
    results = validator.validate_all_datasets()
    
    # Print summary
    validator.print_validation_summary()
    
    # Save report
    validator.save_validation_report(args.output)
    
    print(f"\n📄 Detailed report saved to: {args.output}")

if __name__ == "__main__":
    main()
