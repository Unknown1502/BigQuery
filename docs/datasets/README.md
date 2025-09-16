# Real Datasets Implementation for Geospatial Pricing System

This directory contains comprehensive guides and tools for integrating real-world datasets into your geospatial pricing intelligence system.

##  Overview

Your geospatial pricing system requires multiple types of real-world data for training robust ML models and delivering accurate pricing intelligence. This implementation provides production-ready integration for:

- **Transportation Data**: NYC Taxi, Chicago Taxi, Uber Movement data
- **Location Intelligence**: POI data, demographics, business districts  
- **Visual Data**: Street-level imagery for crowd detection and scene analysis
- **Weather Data**: Historical and real-time weather conditions
- **Economic Data**: Market indicators and competitive intelligence

##  Files in this Directory

### Documentation
- **`PROFESSIONAL_DATASETS_GUIDE.md`** - Complete implementation guide with dataset sources and code examples
- **`REAL_DATASETS_GUIDE.md`** - Alternative guide with visual indicators (contains emojis)
- **`README.md`** - This file

### Implementation Scripts
- **`../scripts/data/production_dataset_integrator.py`** - Main integration class with all dataset collectors
- **`../scripts/data/run_dataset_integration.py`** - Simple execution script to run the integration
- **`../scripts/data/integrate_real_datasets.py`** - Original integration script (basic version)

##  Quick Start

### 1. Set Up API Keys

Set the following environment variables for the APIs you want to use:

```bash
# Required for most functionality
export GOOGLE_MAPS_API_KEY="your_google_maps_key"
export OPENWEATHER_API_KEY="your_openweather_key"

# Optional but recommended
export FOURSQUARE_API_KEY="your_foursquare_key"
export MAPILLARY_ACCESS_TOKEN="your_mapillary_token"
export NOAA_API_TOKEN="your_noaa_token"
export FRED_API_KEY="your_fred_key"
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install pandas requests sodapy googlemaps cenpy fredapi

# Optional for enhanced functionality
pip install geopandas overpy
```

### 3. Run the Integration

```bash
# Simple execution
python scripts/data/run_dataset_integration.py

# Or use the main integrator directly
python scripts/data/production_dataset_integrator.py
```

##  Dataset Sources and Links

### Transportation Data

#### NYC Taxi & Limousine Commission (TLC)
- **Volume**: 100M+ records annually
- **Direct Links**:
  ```
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
  https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet
  https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_2024-01.parquet
  https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-01.parquet
  ```

#### Chicago Transportation
- **API**: https://data.cityofchicago.org/resource/wrvz-psew.json
- **Volume**: 100M+ records

#### Uber Movement
- **Source**: https://movement.uber.com/
- **Coverage**: 60+ cities globally

### Location Intelligence

#### US Census Bureau
- **API**: https://api.census.gov/data/2022/acs/acs5
- **Coverage**: Block group level demographics

#### OpenStreetMap
- **Global Extract**: https://download.geofabrik.de/
- **Overpass API**: https://overpass-api.de/api/interpreter

#### Google Places API
- **Endpoint**: https://maps.googleapis.com/maps/api/place/nearbysearch/json
- **Rate Limits**: 100,000 requests/day (free tier)

#### Foursquare Places API
- **Endpoint**: https://api.foursquare.com/v3/places/search
- **Volume**: 100M+ venues globally

### Visual Data

#### Google Street View
- **Endpoint**: https://maps.googleapis.com/maps/api/streetview
- **Usage**: 25,000 requests/day (free tier)

#### Mapillary
- **API**: https://graph.mapillary.com/
- **Coverage**: 2B+ street-level images globally

#### Open Images Dataset V7
- **Source**: https://storage.googleapis.com/openimages/web/index.html
- **Volume**: 9M+ images with bounding boxes

### Weather Data

#### OpenWeatherMap
- **Historical**: https://api.openweathermap.org/data/2.5/onecall/timemachine
- **Free Tier**: 1,000 calls/day

#### NOAA
- **API**: https://www.ncdc.noaa.gov/cdo-web/api/v2/
- **Free Access**: No rate limits

### Economic Data

#### Federal Reserve (FRED)
- **API**: https://api.stlouisfed.org/fred/
- **Coverage**: GDP, unemployment, inflation by region

#### Bureau of Labor Statistics
- **API**: https://api.bls.gov/publicAPI/v2/timeseries/data/
- **Coverage**: Employment, price indices

##  Expected Dataset Sizes

| Dataset Category | Records | Size | Update Frequency |
|-----------------|---------|------|------------------|
| NYC Taxi Data | 100M+ | 30GB | Monthly |
| Chicago Taxi | 50M+ | 15GB | Daily |
| Street Images | 50K+ | 20GB | Weekly |
| Weather Data | 2M+ | 5GB | Daily |
| POI Data | 500K+ | 2GB | Monthly |
| Demographics | 10K+ | 500MB | Annually |
| **Total** | **162M+** | **72GB+** | **Continuous** |

## 🔧 Configuration

### API Key Requirements

| Service | Required For | Free Tier Limits |
|---------|-------------|------------------|
| Google Maps | Street View, Places | 25K requests/day |
| OpenWeatherMap | Weather data | 1K calls/day |
| Foursquare | POI data | 100K requests/month |
| Mapillary | Street images | Academic use free |
| NOAA | Weather data | No limits |
| FRED | Economic data | No limits |

### Storage Requirements

- **Development**: ~10GB (reduced datasets)
- **Production**: ~80GB (full datasets)
- **Recommended**: SSD storage for fast I/O

### Processing Requirements

- **Memory**: 16GB+ recommended
- **CPU**: Multi-core for parallel processing
- **Network**: High bandwidth for initial downloads

##  Implementation Phases

### Phase 1: Transportation Data (Week 1-2)
- NYC TLC data integration
- Chicago taxi data via Socrata API
- Uber Movement data
- Data cleaning and standardization

### Phase 2: Location Intelligence (Week 2-3)
- Google Places API integration
- Foursquare Places data
- US Census demographics
- OpenStreetMap POI extraction

### Phase 3: Visual Data (Week 3-4)
- Google Street View collection
- Mapillary API integration
- Open Images dataset subset
- Image preprocessing pipeline

### Phase 4: Weather & Economic (Week 4)
- OpenWeatherMap historical API
- NOAA weather stations
- FRED economic indicators
- BLS employment data

### Phase 5: Integration & Testing (Week 5)
- Unified data pipeline
- Quality checks and validation
- Automated refresh setup
- Performance testing

##  Data Quality Features

### Automated Cleaning
- Coordinate validation (geographic bounds)
- Temporal filtering (reasonable trip durations)
- Outlier removal (extreme fares, distances)
- Missing value handling

### Standardization
- Unified column naming across datasets
- Consistent datetime formats
- Standardized geographic projections
- Normalized categorical values

### Validation
- Data completeness checks
- Cross-dataset consistency validation
- Statistical anomaly detection
- Schema compliance verification

##  Usage Examples

### Basic Integration
```python
from scripts.data.production_dataset_integrator import ProductionDatasetIntegrator

config = {
    'google_maps_api_key': 'your_key',
    'openweather_api_key': 'your_key'
}

integrator = ProductionDatasetIntegrator(config)

# Integrate NYC taxi data for 2024
integrator.integrate_nyc_taxi_data(2024, [1, 2, 3])

# Collect weather data for locations
locations = [
    {'lat': 40.7589, 'lng': -73.9851, 'location_id': 'loc_downtown_001'}
]
integrator.integrate_comprehensive_weather(locations, days_back=30)
```

### Custom Data Processing
```python
# Access cleaned data
import pandas as pd

# Load processed taxi data
taxi_data = pd.read_parquet('data/real_datasets/transportation/nyc_taxi_combined_2024.parquet')

# Load weather data
weather_data = pd.read_parquet('data/real_datasets/weather/comprehensive_weather_data.parquet')

# Merge for analysis
merged_data = taxi_data.merge(weather_data, on=['location_id', 'date'])
```

##  Important Notes

### Rate Limiting
- All API integrations include proper rate limiting
- Automatic retry logic for failed requests
- Respectful usage of free tier limits

### Error Handling
- Comprehensive logging for debugging
- Graceful degradation when APIs are unavailable
- Data validation at each processing step

### Privacy & Compliance
- No personal data collection
- Aggregated data only
- Compliance with API terms of service

##  Support

### Troubleshooting
1. Check API key configuration
2. Verify network connectivity
3. Review logs in `dataset_integration.log`
4. Ensure sufficient disk space

### Common Issues
- **API Rate Limits**: Reduce batch sizes or add delays
- **Network Timeouts**: Increase timeout values
- **Storage Space**: Clean old datasets or increase storage
- **Memory Issues**: Process data in smaller chunks

##  Maintenance

### Regular Updates
- Monthly refresh of transportation data
- Weekly update of weather data
- Quarterly refresh of POI and demographic data
- Annual update of economic indicators

### Monitoring
- Data freshness checks
- Quality metric tracking
- Storage usage monitoring
- API usage tracking

---

This implementation provides a robust foundation for integrating real-world datasets into your geospatial pricing system. The modular design allows you to start with basic datasets and gradually expand to include more comprehensive data sources as your system scales.
