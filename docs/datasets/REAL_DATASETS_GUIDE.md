# Real Datasets Implementation Guide
## Comprehensive Dataset Sources for Geospatial Pricing Intelligence

---

##  Overview

This guide provides comprehensive real-world dataset sources and implementation instructions for your geospatial pricing system. All datasets are production-ready with 100K-10M+ records suitable for training robust ML models.

---

##  Dataset Categories & Sources

### 1.  Transportation & Ride Data

#### **NYC Taxi & Limousine Commission (TLC) Data**
- **Source**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Volume**: 100M+ records annually
- **Format**: Parquet files (monthly)
- **Coverage**: 2009-present
- **Direct Links**:
  ```
  # Yellow Taxi Data (2024)
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-03.parquet
  
  # Green Taxi Data (2024)
  https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet
  https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-02.parquet
  
  # For-Hire Vehicle Data (2024)
  https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_2024-01.parquet
  https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_2024-02.parquet
  
  # High Volume For-Hire Vehicle Data (Uber/Lyft)
  https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-01.parquet
  https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-02.parquet
  ```

#### **Chicago Transportation Data**
- **Source**: https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew
- **API Endpoint**: https://data.cityofchicago.org/resource/wrvz-psew.json
- **Volume**: 100M+ records
- **Format**: JSON/CSV via Socrata API

#### **Uber Movement Data**
- **Source**: https://movement.uber.com/
- **Coverage**: 60+ cities globally
- **Data Types**: Travel times, speed data
- **Format**: CSV downloads

#### **Lyft Bay Wheels (Bike Share)**
- **Source**: https://www.lyft.com/bikes/bay-wheels/system-data
- **Volume**: 2M+ trips annually
- **Format**: CSV files

### 2.  Location Intelligence & Demographics

#### **US Census Bureau Data**
- **American Community Survey (ACS)**:
  - API: https://api.census.gov/data/2022/acs/acs5
  - Documentation: https://www.census.gov/data/developers/data-sets/acs-5year.html
  - Coverage: Block group level demographics

#### **OpenStreetMap (OSM) Data**
- **Global POI Data**: https://download.geofabrik.de/
- **US Extract**: https://download.geofabrik.de/north-america/us-latest.osm.pbf
- **NYC Extract**: https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf
- **Overpass API**: https://overpass-api.de/api/interpreter

#### **Foursquare Places API**
- **Endpoint**: https://api.foursquare.com/v3/places/search
- **Coverage**: Global POI data with categories, ratings, popularity
- **Volume**: 100M+ venues globally

#### **Google Places API**
- **Endpoint**: https://maps.googleapis.com/maps/api/place/nearbysearch/json
- **Coverage**: Global POI with ratings, reviews, business hours
- **Rate Limits**: 100,000 requests/day (free tier)

### 3.  Visual & Street-Level Data

#### **Google Street View Static API**
- **Endpoint**: https://maps.googleapis.com/maps/api/streetview
- **Parameters**: 
  ```
  size=640x640
  location=lat,lng
  heading=0,90,180,270
  pitch=0
  fov=90
  key=YOUR_API_KEY
  ```
- **Usage**: 25,000 requests/day (free tier)

#### **Mapillary Street-Level Imagery**
- **API**: https://graph.mapillary.com/
- **Coverage**: 2B+ street-level images globally
- **Free Access**: Academic/research use
- **Documentation**: https://www.mapillary.com/developer/api-documentation

#### **Microsoft Bing Maps Streetside**
- **API**: https://dev.virtualearth.net/REST/v1/Imagery/Map/Streetside
- **Coverage**: Major cities worldwide
- **Format**: 360-degree imagery

#### **Open Images Dataset V7**
- **Source**: https://storage.googleapis.com/openimages/web/index.html
- **Volume**: 9M+ images with bounding boxes
- **Categories**: 600+ object classes including vehicles, people, buildings
- **Direct Download**: 
  ```
  # Training Images
  https://storage.googleapis.com/openimages/v5/train-images-boxable-with-rotation.csv
  
  # Annotations
  https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv
  ```

### 4.  Weather Data

#### **OpenWeatherMap API**
- **Current Weather**: https://api.openweathermap.org/data/2.5/weather
- **Historical Weather**: https://api.openweathermap.org/data/2.5/onecall/timemachine
- **5-Day Forecast**: https://api.openweathermap.org/data/2.5/forecast
- **Free Tier**: 1,000 calls/day

#### **NOAA Weather Data**
- **Climate Data Online**: https://www.ncdc.noaa.gov/cdo-web/api/v2/
- **Historical Data**: https://www.ncei.noaa.gov/data/
- **NYC Weather Station**: USW00094728 (Central Park)
- **Free Access**: No rate limits

#### **Weather Underground API**
- **Historical Weather**: https://api.weather.com/v1/location/point
- **Hourly Data**: Available for major cities
- **IBM Weather API**: https://weather.com/swagger-docs/ui/sun/v1/sunV1.json

### 5.  Economic & Competitive Data

#### **Bureau of Labor Statistics (BLS)**
- **Consumer Price Index**: https://api.bls.gov/publicAPI/v2/timeseries/data/
- **Employment Data**: https://api.bls.gov/publicAPI/v1/timeseries/data/
- **Regional Price Parities**: Available by metro area

#### **Federal Reserve Economic Data (FRED)**
- **API**: https://api.stlouisfed.org/fred/
- **Economic Indicators**: GDP, unemployment, inflation by region
- **Real Estate Data**: Housing price indices

#### **Zillow Research Data**
- **Home Values**: https://www.zillow.com/research/data/
- **Rental Data**: https://files.zillowstatic.com/research/public_csvs/
- **Format**: CSV files updated monthly

---

##  Implementation Plan

### Phase 1: Core Transportation Data (Week 1-2)

#### **Step 1: NYC TLC Data Integration**
```python
# Enhanced integration script
import pandas as pd
import requests
from pathlib import Path

def download_nyc_taxi_data(year=2024, months=range(1, 13)):
    """Download NYC taxi data for specified months"""
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    
    datasets = {
        'yellow': f"{base_url}/yellow_tripdata_{year}-{month:02d}.parquet",
        'green': f"{base_url}/green_tripdata_{year}-{month:02d}.parquet",
        'fhv': f"{base_url}/fhv_tripdata_{year}-{month:02d}.parquet",
        'fhvhv': f"{base_url}/fhvhv_tripdata_{year}-{month:02d}.parquet"
    }
    
    for dataset_type, url_template in datasets.items():
        for month in months:
            url = url_template.format(year=year, month=month)
            filename = f"nyc_{dataset_type}_{year}_{month:02d}.parquet"
            
            print(f"Downloading {filename}...")
            df = pd.read_parquet(url)
            
            # Process and save
            df_processed = process_taxi_data(df, dataset_type)
            df_processed.to_parquet(f"data/real_datasets/{filename}")
            
            print(f" Saved {len(df_processed)} records to {filename}")

def process_taxi_data(df, dataset_type):
    """Clean and standardize taxi data"""
    # Standardize column names across datasets
    column_mapping = {
        'tpep_pickup_datetime': 'pickup_datetime',
        'tpep_dropoff_datetime': 'dropoff_datetime',
        'lpep_pickup_datetime': 'pickup_datetime',
        'lpep_dropoff_datetime': 'dropoff_datetime',
        'pickup_datetime': 'pickup_datetime',
        'dropOff_datetime': 'dropoff_datetime'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Filter valid coordinates (NYC area)
    if 'pickup_latitude' in df.columns:
        df = df[
            (df['pickup_latitude'].between(40.4, 41.0)) &
            (df['pickup_longitude'].between(-74.3, -73.7)) &
            (df['dropoff_latitude'].between(40.4, 41.0)) &
            (df['dropoff_longitude'].between(-74.3, -73.7))
        ]
    
    # Calculate trip metrics
    if 'pickup_datetime' in df.columns and 'dropoff_datetime' in df.columns:
        df['trip_duration_minutes'] = (
            pd.to_datetime(df['dropoff_datetime']) - 
            pd.to_datetime(df['pickup_datetime'])
        ).dt.total_seconds() / 60
        
        # Filter reasonable trip durations (1 min to 3 hours)
        df = df[
            (df['trip_duration_minutes'] >= 1) &
            (df['trip_duration_minutes'] <= 180)
        ]
    
    # Add dataset source
    df['data_source'] = dataset_type
    df['ingestion_timestamp'] = pd.Timestamp.now()
    
    return df
```

#### **Step 2: Multi-City Data Integration**
```python
def integrate_chicago_data():
    """Integrate Chicago taxi data via Socrata API"""
    import sodapy
    
    client = sodapy.Socrata("data.cityofchicago.org", None)
    
    # Get data in chunks
    limit = 50000
    offset = 0
    all_data = []
    
    while True:
        results = client.get("wrvz-psew", limit=limit, offset=offset)
        if not results:
            break
            
        all_data.extend(results)
        offset += limit
        print(f"Downloaded {len(all_data)} records...")
        
        if len(all_data) >= 1000000:  # Limit to 1M records
            break
    
    df = pd.DataFrame(all_data)
    df_processed = process_chicago_data(df)
    df_processed.to_parquet("data/real_datasets/chicago_taxi_2024.parquet")
    
    return df_processed
```

### Phase 2: Location Intelligence (Week 2-3)

#### **Step 1: POI Data Collection**
```python
def collect_poi_data(locations, radius_km=2):
    """Collect POI data using multiple sources"""
    import googlemaps
    import requests
    
    # Google Places API
    gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))
    
    # Foursquare API
    fsq_headers = {
        'Authorization': os.getenv('FOURSQUARE_API_KEY')
    }
    
    all_pois = []
    
    for location in locations:
        lat, lng = location['lat'], location['lng']
        
        # Google Places
        places_result = gmaps.places_nearby(
            location=(lat, lng),
            radius=radius_km * 1000,
            type='establishment'
        )
        
        for place in places_result.get('results', []):
            poi = {
                'location_id': location['location_id'],
                'poi_name': place['name'],
                'poi_type': ','.join(place.get('types', [])),
                'poi_rating': place.get('rating', 0),
                'poi_price_level': place.get('price_level', 0),
                'poi_lat': place['geometry']['location']['lat'],
                'poi_lng': place['geometry']['location']['lng'],
                'source': 'google_places'
            }
            all_pois.append(poi)
        
        # Foursquare Places
        fsq_url = "https://api.foursquare.com/v3/places/search"
        fsq_params = {
            'll': f"{lat},{lng}",
            'radius': radius_km * 1000,
            'limit': 50
        }
        
        fsq_response = requests.get(fsq_url, headers=fsq_headers, params=fsq_params)
        fsq_data = fsq_response.json()
        
        for place in fsq_data.get('results', []):
            poi = {
                'location_id': location['location_id'],
                'poi_name': place['name'],
                'poi_type': ','.join([cat['name'] for cat in place.get('categories', [])]),
                'poi_rating': place.get('rating', 0),
                'poi_price_level': place.get('price', 0),
                'poi_lat': place['geocodes']['main']['latitude'],
                'poi_lng': place['geocodes']['main']['longitude'],
                'source': 'foursquare'
            }
            all_pois.append(poi)
    
    return pd.DataFrame(all_pois)
```

#### **Step 2: Demographics Integration**
```python
def integrate_census_data(locations):
    """Integrate US Census demographic data"""
    import cenpy
    
    # American Community Survey 5-year estimates
    acs = cenpy.products.ACS(2022)
    
    demographic_data = []
    
    for location in locations:
        lat, lng = location['lat'], location['lng']
        
        # Get census tract for coordinates
        tract_data = acs.from_coordinates((lng, lat))
        
        if not tract_data.empty:
            demo_record = {
                'location_id': location['location_id'],
                'census_tract': tract_data.iloc[0]['GEOID'],
                'total_population': tract_data.iloc[0].get('B01003_001E', 0),
                'median_income': tract_data.iloc[0].get('B19013_001E', 0),
                'median_age': tract_data.iloc[0].get('B01002_001E', 0),
                'housing_units': tract_data.iloc[0].get('B25001_001E', 0),
                'population_density': tract_data.iloc[0].get('B01003_001E', 0) / 
                                   (tract_data.iloc[0].get('ALAND', 1) / 1000000)  # per sq km
            }
            demographic_data.append(demo_record)
    
    return pd.DataFrame(demographic_data)
```

### Phase 3: Visual Data Collection (Week 3-4)

#### **Step 1: Street View Image Collection**
```python
def collect_street_images_comprehensive(locations):
    """Comprehensive street image collection from multiple sources"""
    
    # Google Street View
    google_images = collect_google_streetview(locations)
    
    # Mapillary Images
    mapillary_images = collect_mapillary_images(locations)
    
    # Combine and process
    all_images = google_images + mapillary_images
    
    return process_image_metadata(all_images)

def collect_google_streetview(locations):
    """Collect Google Street View images"""
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    images_dir = Path("data/real_datasets/street_images/google")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    image_metadata = []
    
    for location in locations:
        lat, lng = location['lat'], location['lng']
        location_id = location['location_id']
        
        # Multiple angles and times
        for heading in [0, 45, 90, 135, 180, 225, 270, 315]:
            for pitch in [-10, 0, 10]:
                url = "https://maps.googleapis.com/maps/api/streetview"
                params = {
                    'size': '640x640',
                    'location': f"{lat},{lng}",
                    'heading': heading,
                    'pitch': pitch,
                    'fov': 90,
                    'key': api_key
                }
                
                try:
                    response = requests.get(url, params=params)
                    
                    if response.status_code == 200:
                        filename = f"{location_id}_h{heading}_p{pitch}.jpg"
                        filepath = images_dir / filename
                        
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        image_metadata.append({
                            'location_id': location_id,
                            'filename': filename,
                            'filepath': str(filepath),
                            'lat': lat,
                            'lng': lng,
                            'heading': heading,
                            'pitch': pitch,
                            'source': 'google_streetview',
                            'timestamp': datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    print(f"Error collecting Google Street View image: {e}")
    
    return image_metadata

def collect_mapillary_images(locations, radius_m=500):
    """Collect Mapillary street-level images"""
    access_token = os.getenv('MAPILLARY_ACCESS_TOKEN')
    images_dir = Path("data/real_datasets/street_images/mapillary")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    image_metadata = []
    
    for location in locations:
        lat, lng = location['lat'], location['lng']
        location_id = location['location_id']
        
        # Search for images near location
        url = "https://graph.mapillary.com/images"
        params = {
            'access_token': access_token,
            'fields': 'id,thumb_2048_url,computed_geometry,compass_angle,captured_at',
            'bbox': f"{lng-0.01},{lat-0.01},{lng+0.01},{lat+0.01}",
            'limit': 20
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            for img in data.get('data', []):
                # Download image
                img_url = img['thumb_2048_url']
                img_response = requests.get(img_url)
                
                if img_response.status_code == 200:
                    filename = f"{location_id}_mapillary_{img['id']}.jpg"
                    filepath = images_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                    
                    image_metadata.append({
                        'location_id': location_id,
                        'filename': filename,
                        'filepath': str(filepath),
                        'lat': img['computed_geometry']['coordinates'][1],
                        'lng': img['computed_geometry']['coordinates'][0],
                        'heading': img.get('compass_angle', 0),
                        'captured_at': img.get('captured_at'),
                        'source': 'mapillary',
                        'mapillary_id': img['id']
                    })
                    
        except Exception as e:
            print(f"Error collecting Mapillary images: {e}")
    
    return image_metadata
```

### Phase 4: Weather Data Integration (Week 4)

#### **Comprehensive Weather Data Collection**
```python
def integrate_comprehensive_weather(locations, days_back=365):
    """Integrate weather data from multiple sources"""
    
    # OpenWeatherMap historical data
    owm_data = collect_openweather_data(locations, days_back)
    
    # NOAA weather data
    noaa_data = collect_noaa_data(locations, days_back)
    
    # Combine and clean
    weather_df = pd.concat([owm_data, noaa_data], ignore_index=True)
    weather_df = clean_weather_data(weather_df)
    
    return weather_df

def collect_openweather_data(locations, days_back):
    """Collect OpenWeatherMap historical data"""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    weather_data = []
    
    for location in locations:
        lat, lng = location['lat'], location['lng']
        location_id = location['location_id']
        
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i)
            timestamp = int(date.timestamp())
            
            url = "http://api.openweathermap.org/data/2.5/onecall/timemachine"
            params = {
                'lat': lat,
                'lon': lng,
                'dt': timestamp,
                'appid': api_key,
                'units': 'metric'
            }
            
            try:
                response = requests.get(url, params=params)
                data = response.json()
                
                if 'current' in data:
                    weather_record = {
                        'location_id': location_id,
                        'timestamp': date.isoformat(),
                        'temperature': data['current']['temp'],
                        'feels_like': data['current']['feels_like'],
                        'humidity': data['current']['humidity'],
                        'pressure': data['current']['pressure'],
                        'weather_condition': data['current']['weather'][0]['main'].lower(),
                        'weather_description': data['current']['weather'][0]['description'],
                        'wind_speed': data['current']['wind_speed'],
                        'wind_direction': data['current'].get('wind_deg', 0),
                        'visibility': data['current'].get('visibility', 10000),
                        'uv_index': data['current'].get('uvi', 0),
                        'source': 'openweathermap'
                    }
                    weather_data.append(weather_record)
                    
            except Exception as e:
                print(f"Error fetching OpenWeather data: {e}")
                time.sleep(1)  # Rate limiting
    
    return pd.DataFrame(weather_data)

def collect_noaa_data(locations, days_back):
    """Collect NOAA weather station data"""
    # NOAA API integration
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
    headers = {'token': os.getenv('NOAA_API_TOKEN')}
    
    weather_data = []
    
    # Find nearest weather stations
    for location in locations:
        lat, lng = location['lat'], location['lng']
        location_id = location['location_id']
        
        # Get nearby stations
        stations_url = f"{base_url}/stations"
        stations_params = {
            'extent': f"{lat-0.1},{lng-0.1},{lat+0.1},{lng+0.1}",
            'limit': 5,
            'datasetid': 'GHCND'
        }
        
        try:
            stations_response = requests.get(stations_url, headers=headers, params=stations_params)
            stations_data = stations_response.json()
            
            if 'results' in stations_data:
                station_id = stations_data['results'][0]['id']
                
                # Get weather data for station
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                data_url = f"{base_url}/data"
                data_params = {
                    'datasetid': 'GHCND',
                    'stationid': station_id,
                    'startdate': start_date,
                    'enddate': end_date,
                    'datatypeid': 'TMAX,TMIN,PRCP,SNOW',
                    'limit': 1000
                }
                
                data_response = requests.get(data_url, headers=headers, params=data_params)
                weather_records = data_response.json()
                
                for record in weather_records.get('results', []):
                    weather_data.append({
                        'location_id': location_id,
                        'timestamp': record['date'],
                        'datatype': record['datatype'],
                        'value': record['value'],
                        'station_id': station_id,
                        'source': 'noaa'
                    })
                    
        except Exception as e:
            print(f"Error fetching NOAA data: {e}")
    
    return pd.DataFrame(weather_data)
```

---

##  Implementation Checklist

### **Week 1-2: Transportation Data**
- [ ] Set up NYC TLC data pipeline
- [ ] Download 12 months of taxi data (4 datasets × 12 months)
- [ ] Implement Chicago taxi data integration
- [ ] Add Uber Movement data
- [ ] Create unified transportation schema
- [ ] Validate data quality and completeness

### **Week 2-3: Location Intelligence**
- [ ] Set up Google Places API integration
- [ ] Implement Foursquare Places data collection
- [ ] Integrate US Census demographic data
- [ ] Add OpenStreetMap POI extraction
- [ ] Create location clustering algorithms
- [ ] Validate geographic coverage

### **Week 3-4: Visual Data**
- [ ] Implement Google Street View collection
- [ ] Set up Mapillary API integration
- [ ] Download Open Images dataset subset
- [ ] Create image preprocessing pipeline
- [ ] Implement crowd detection training data
- [ ] Validate image quality and coverage

### **Week 4: Weather & Economic Data**
- [ ] Set up OpenWeatherMap historical API
- [ ] Implement NOAA weather station integration
- [ ] Add economic indicators from FRED API
- [ ] Integrate BLS employment data
- [ ] Create weather impact models
- [ ] Validate temporal coverage

### **Week 5: Integration & Testing**
- [ ] Create unified data pipeline
- [ ] Implement data quality checks
- [ ] Set up automated data refresh
- [ ] Create monitoring and alerting
- [ ] Performance testing with full datasets
- [ ] Documentation and handover

---

## 🔧 Technical Requirements

### **API Keys Required**
```bash
# Set these environment variables
export GOOGLE_MAPS_API_KEY="your_google_maps_key"
export OPENWEATHER_API_KEY="your_openweather_key"
export FOURSQUARE_API_KEY="your_foursquare_key"
export MAPILLARY_ACCESS_TOKEN="your_mapillary_token"
export NOAA_API_TOKEN="your_noaa_token"
```

### **Storage Requirements**
- **Transportation Data**: ~50GB (1 year NYC + Chicago)
- **Street Images**: ~20GB (5 cities × 1000 images)
- **Weather Data**: ~5GB (1 year hourly data)
- **POI Data**: ~2GB (processed and indexed)
- **Total**: ~80GB for comprehensive dataset

### **Processing Requirements**
- **Memory**: 32GB+ recommended for large dataset processing
- **CPU**: Multi-core for parallel processing
- **Network**: High bandwidth for initial data downloads
- **Storage**: SSD recommended for fast I/O operations

---

## 📊 Expected Dataset Sizes

| Dataset Category | Records | Size | Update Frequency |
|-----------------|---------|------|------------------|
| NYC Taxi Data | 100M+ | 30GB | Monthly |
| Chicago Taxi | 50M+ | 15GB | Daily |
| Street Images | 50K+ | 20GB | Weekly |
| Weather Data | 2M+ | 5GB | Daily |
| POI Data | 500K+ | 2GB | Monthly |
| Demographics | 10K+ | 500MB | Annually |
| **Total** | **162M+** | **72GB+** | **Continuous** |

---

## 🚀 Next Steps

1. **Review and approve this implementation plan**
2. **Obtain necessary API keys and credentials**
3. **Set up development environment with storage**
4. **Begin with Phase 1 (Transportation Data)**
5. **Implement data quality monitoring**
6. **Scale to production infrastructure**

This comprehensive dataset integration will provide your geospatial pricing system with production-quality real-world data for training robust ML models and delivering accurate pricing intelligence.
