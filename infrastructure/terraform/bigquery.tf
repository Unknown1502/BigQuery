# BigQuery Resources for Dynamic Pricing Intelligence
# Provisions datasets, tables, and AI/ML resources for multimodal geospatial analysis

# Main dataset for ride intelligence
resource "google_bigquery_dataset" "ride_intelligence" {
  dataset_id                 = var.dataset_id
  friendly_name             = "Ride Intelligence Dataset"
  description               = "Dataset for dynamic pricing with multimodal geospatial intelligence"
  location                  = var.bigquery_location
  delete_contents_on_destroy = var.bigquery_delete_contents_on_destroy
  
  labels = var.labels
  
  access {
    role          = "OWNER"
    user_by_email = google_service_account.data_engineering.email
  }
  
  access {
    role           = "READER"
    special_group  = "projectReaders"
  }
  
  access {
    role           = "WRITER"
    special_group  = "projectWriters"
  }
}

# Street imagery table for visual intelligence
resource "google_bigquery_table" "street_imagery" {
  dataset_id          = google_bigquery_dataset.ride_intelligence.dataset_id
  table_id            = "street_imagery"
  deletion_protection = false
  
  description = "Street imagery data with visual analysis results"
  
  labels = var.labels
  
  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
  
  clustering = ["location_id", "camera_id"]
  
  schema = jsonencode([
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Image capture timestamp"
    },
    {
      name = "location_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Location identifier"
    },
    {
      name = "camera_id"
      type = "STRING"
      mode = "NULLABLE"
      description = "Camera identifier"
    },
    {
      name = "image_url"
      type = "STRING"
      mode = "REQUIRED"
      description = "Cloud Storage URL for the image"
    },
    {
      name = "crowd_count"
      type = "INTEGER"
      mode = "NULLABLE"
      description = "Number of people detected in the image"
    },
    {
      name = "activity_types"
      type = "STRING"
      mode = "REPEATED"
      description = "Types of activities detected"
    },
    {
      name = "accessibility_score"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Accessibility score for pickup/dropoff (0-1)"
    },
    {
      name = "weather_conditions"
      type = "STRING"
      mode = "NULLABLE"
      description = "Weather conditions visible in image"
    },
    {
      name = "confidence_score"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Overall confidence in analysis (0-1)"
    },
    {
      name = "processing_metadata"
      type = "JSON"
      mode = "NULLABLE"
      description = "Metadata about image processing"
    }
  ])
}

# Location profiles table
resource "google_bigquery_table" "location_profiles" {
  dataset_id          = google_bigquery_dataset.ride_intelligence.dataset_id
  table_id            = "location_profiles"
  deletion_protection = false
  
  description = "Location profiles with semantic embeddings and metadata"
  
  labels = var.labels
  
  clustering = ["location_type", "business_district"]
  
  schema = jsonencode([
    {
      name = "location_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Unique location identifier"
    },
    {
      name = "latitude"
      type = "FLOAT"
      mode = "REQUIRED"
      description = "Location latitude"
    },
    {
      name = "longitude"
      type = "FLOAT"
      mode = "REQUIRED"
      description = "Location longitude"
    },
    {
      name = "location_description"
      type = "STRING"
      mode = "NULLABLE"
      description = "Human-readable location description"
    },
    {
      name = "location_type"
      type = "STRING"
      mode = "NULLABLE"
      description = "Type of location (airport, mall, residential, etc.)"
    },
    {
      name = "business_types"
      type = "STRING"
      mode = "REPEATED"
      description = "Types of businesses in the area"
    },
    {
      name = "demographic_profile"
      type = "STRING"
      mode = "NULLABLE"
      description = "Demographic profile of the area"
    },
    {
      name = "typical_events"
      type = "STRING"
      mode = "REPEATED"
      description = "Typical events that occur at this location"
    },
    {
      name = "business_district"
      type = "BOOLEAN"
      mode = "NULLABLE"
      description = "Whether location is in a business district"
    },
    {
      name = "transportation_hubs"
      type = "STRING"
      mode = "REPEATED"
      description = "Nearby transportation hubs"
    },
    {
      name = "population_density"
      type = "STRING"
      mode = "NULLABLE"
      description = "Population density category"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Profile creation timestamp"
    },
    {
      name = "updated_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Profile last update timestamp"
    }
  ])
}

# Historical rides table
resource "google_bigquery_table" "historical_rides" {
  dataset_id          = google_bigquery_dataset.ride_intelligence.dataset_id
  table_id            = "historical_rides"
  deletion_protection = false
  
  description = "Historical ride data for demand forecasting and pricing analysis"
  
  labels = var.labels
  
  time_partitioning {
    type  = "DAY"
    field = "ride_timestamp"
  }
  
  clustering = ["location_id", "ride_status"]
  
  schema = jsonencode([
    {
      name = "ride_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Unique ride identifier"
    },
    {
      name = "ride_timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Ride request timestamp"
    },
    {
      name = "location_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Pickup location identifier"
    },
    {
      name = "destination_location_id"
      type = "STRING"
      mode = "NULLABLE"
      description = "Destination location identifier"
    },
    {
      name = "base_price"
      type = "FLOAT"
      mode = "REQUIRED"
      description = "Base price for the ride"
    },
    {
      name = "surge_multiplier"
      type = "FLOAT"
      mode = "REQUIRED"
      description = "Surge pricing multiplier applied"
    },
    {
      name = "final_price"
      type = "FLOAT"
      mode = "REQUIRED"
      description = "Final price charged to customer"
    },
    {
      name = "ride_status"
      type = "STRING"
      mode = "REQUIRED"
      description = "Status of the ride (completed, cancelled, etc.)"
    },
    {
      name = "wait_time_minutes"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Customer wait time in minutes"
    },
    {
      name = "ride_duration_minutes"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Ride duration in minutes"
    },
    {
      name = "distance_km"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Ride distance in kilometers"
    },
    {
      name = "weather_conditions"
      type = "STRING"
      mode = "NULLABLE"
      description = "Weather conditions during ride"
    },
    {
      name = "traffic_conditions"
      type = "STRING"
      mode = "NULLABLE"
      description = "Traffic conditions during ride"
    },
    {
      name = "customer_rating"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Customer rating for the ride (1-5)"
    },
    {
      name = "driver_rating"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Driver rating for the ride (1-5)"
    }
  ])
}

# Real-time events table for streaming data
resource "google_bigquery_table" "realtime_events" {
  dataset_id          = google_bigquery_dataset.ride_intelligence.dataset_id
  table_id            = "realtime_events"
  deletion_protection = false
  
  description = "Real-time events from multimodal data streams"
  
  labels = var.labels
  
  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
  
  clustering = ["event_type", "location_id"]
  
  schema = jsonencode([
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Event timestamp"
    },
    {
      name = "processing_timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Processing timestamp"
    },
    {
      name = "processing_latency_ms"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Processing latency in milliseconds"
    },
    {
      name = "event_type"
      type = "STRING"
      mode = "REQUIRED"
      description = "Type of event (street_analysis, weather_update, etc.)"
    },
    {
      name = "location_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Location identifier"
    },
    {
      name = "crowd_density"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Crowd density score"
    },
    {
      name = "accessibility_score"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Accessibility score"
    },
    {
      name = "activity_level"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Activity level score"
    },
    {
      name = "visual_factors"
      type = "STRING"
      mode = "REPEATED"
      description = "Visual factors detected"
    },
    {
      name = "confidence_score"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Confidence score for analysis"
    },
    {
      name = "temperature"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Temperature in Celsius"
    },
    {
      name = "humidity"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Humidity percentage"
    },
    {
      name = "precipitation"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Precipitation amount"
    },
    {
      name = "weather_impact_score"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Weather impact on demand"
    },
    {
      name = "congestion_level"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Traffic congestion level (0-1)"
    },
    {
      name = "average_speed"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Average traffic speed"
    },
    {
      name = "incident_count"
      type = "INTEGER"
      mode = "NULLABLE"
      description = "Number of traffic incidents"
    },
    {
      name = "traffic_impact_score"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Traffic impact on demand"
    },
    {
      name = "avg_competitor_price"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Average competitor price"
    },
    {
      name = "min_competitor_price"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Minimum competitor price"
    },
    {
      name = "max_competitor_price"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Maximum competitor price"
    },
    {
      name = "competitor_count"
      type = "INTEGER"
      mode = "NULLABLE"
      description = "Number of competitors"
    },
    {
      name = "market_position"
      type = "STRING"
      mode = "NULLABLE"
      description = "Market position relative to competitors"
    },
    {
      name = "data_quality_score"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Data quality score (0-1)"
    }
  ])
}

# API pricing requests table for analytics
resource "google_bigquery_table" "api_pricing_requests" {
  dataset_id          = google_bigquery_dataset.ride_intelligence.dataset_id
  table_id            = "api_pricing_requests"
  deletion_protection = false
  
  description = "API pricing requests for performance analytics"
  
  labels = var.labels
  
  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
  
  clustering = ["location_id", "api_version"]
  
  schema = jsonencode([
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Request timestamp"
    },
    {
      name = "request_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Unique request identifier"
    },
    {
      name = "location_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Location identifier"
    },
    {
      name = "base_price"
      type = "FLOAT"
      mode = "REQUIRED"
      description = "Base price calculated"
    },
    {
      name = "surge_multiplier"
      type = "FLOAT"
      mode = "REQUIRED"
      description = "Surge multiplier applied"
    },
    {
      name = "final_price"
      type = "FLOAT"
      mode = "REQUIRED"
      description = "Final price returned"
    },
    {
      name = "confidence_score"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Confidence in pricing decision"
    },
    {
      name = "visual_factors"
      type = "STRING"
      mode = "NULLABLE"
      description = "Visual factors JSON"
    },
    {
      name = "processing_time_ms"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Processing time in milliseconds"
    },
    {
      name = "api_version"
      type = "STRING"
      mode = "REQUIRED"
      description = "API version used"
    },
    {
      name = "user_agent"
      type = "STRING"
      mode = "NULLABLE"
      description = "User agent string"
    },
    {
      name = "ip_address"
      type = "STRING"
      mode = "NULLABLE"
      description = "Client IP address"
    }
  ])
}

# Location embeddings table for semantic similarity
resource "google_bigquery_table" "location_embeddings" {
  dataset_id          = google_bigquery_dataset.ride_intelligence.dataset_id
  table_id            = "location_embeddings"
  deletion_protection = false
  
  description = "Location embeddings for semantic similarity matching"
  
  labels = var.labels
  
  clustering = ["location_id"]
  
  schema = jsonencode([
    {
      name = "location_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Location identifier"
    },
    {
      name = "location_description"
      type = "STRING"
      mode = "REQUIRED"
      description = "Location description used for embedding"
    },
    {
      name = "embedding_vector"
      type = "FLOAT"
      mode = "REPEATED"
      description = "Embedding vector (768 dimensions)"
    },
    {
      name = "embedding_model"
      type = "STRING"
      mode = "REQUIRED"
      description = "Model used to generate embedding"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Embedding creation timestamp"
    },
    {
      name = "updated_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Embedding last update timestamp"
    }
  ])
}

# Competitor analysis table
resource "google_bigquery_table" "competitor_analysis" {
  dataset_id          = google_bigquery_dataset.ride_intelligence.dataset_id
  table_id            = "competitor_analysis"
  deletion_protection = false
  
  description = "Competitor pricing analysis data"
  
  labels = var.labels
  
  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
  
  clustering = ["location_id", "competitor_name"]
  
  schema = jsonencode([
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Analysis timestamp"
    },
    {
      name = "location_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Location identifier"
    },
    {
      name = "competitor_name"
      type = "STRING"
      mode = "REQUIRED"
      description = "Competitor company name"
    },
    {
      name = "competitor_price"
      type = "FLOAT"
      mode = "REQUIRED"
      description = "Competitor price"
    },
    {
      name = "our_price"
      type = "FLOAT"
      mode = "REQUIRED"
      description = "Our price at the same time"
    },
    {
      name = "price_difference_pct"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Price difference percentage"
    },
    {
      name = "market_share_impact"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Estimated market share impact"
    },
    {
      name = "estimated_wait_time"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Competitor estimated wait time"
    },
    {
      name = "service_quality_score"
      type = "FLOAT"
      mode = "NULLABLE"
      description = "Competitor service quality score"
    }
  ])
}

# Real-time pricing table (referenced in pubsub.tf)
resource "google_bigquery_table" "realtime_pricing" {
  dataset_id          = google_bigquery_dataset.ride_intelligence.dataset_id
  table_id            = "realtime_pricing"
  deletion_protection = false
  
  description = "Real-time pricing calculations for streaming"
  
  labels = var.labels
  
  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }
  
  clustering = ["location_id"]
  
  schema = jsonencode([
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
      description = "Pricing calculation timestamp"
    },
    {
      name = "location_id"
      type = "STRING"
      mode = "REQUIRED"
      description = "Location identifier"
    },
    {
      name = "base_price"
      type = "FLOAT64"
      mode = "REQUIRED"
      description = "Base price calculated"
    },
    {
      name = "surge_multiplier"
      type = "FLOAT64"
      mode = "REQUIRED"
      description = "Surge multiplier applied"
    },
    {
      name = "final_price"
      type = "FLOAT64"
      mode = "REQUIRED"
      description = "Final price"
    },
    {
      name = "confidence_score"
      type = "FLOAT64"
      mode = "REQUIRED"
      description = "Confidence score for pricing decision"
    },
    {
      name = "factors"
      type = "STRING"
      mode = "REPEATED"
      description = "Factors influencing pricing decision"
    },
    {
      name = "message_id"
      type = "STRING"
      mode = "NULLABLE"
      description = "Pub/Sub message ID (required for write_metadata)"
    },
    {
      name = "publish_time"
      type = "TIMESTAMP"
      mode = "NULLABLE"
      description = "Pub/Sub message publish time (required for write_metadata)"
    },
    {
      name = "subscription_name"
      type = "STRING"
      mode = "NULLABLE"
      description = "Pub/Sub subscription name (required for write_metadata)"
    },
    {
      name = "attributes"
      type = "JSON"
      mode = "NULLABLE"
      description = "Pub/Sub message attributes (required for write_metadata)"
    }
  ])
}

# External table for street imagery in Cloud Storage
resource "google_bigquery_table" "street_imagery_external" {
  dataset_id          = google_bigquery_dataset.ride_intelligence.dataset_id
  table_id            = "street_imagery_external"
  deletion_protection = false
  
  description = "External table for street imagery stored in Cloud Storage"
  
  labels = var.labels
  
  external_data_configuration {
    autodetect    = false
    source_format = "CSV"
    
    source_uris = [
      "gs://${var.project_id}-street-imagery/*"
    ]
    
    csv_options {
      quote                 = "\""
      allow_jagged_rows     = false
      allow_quoted_newlines = false
      skip_leading_rows     = 1
    }
    
    schema = jsonencode([
      {
        name = "image_path"
        type = "STRING"
        mode = "REQUIRED"
      },
      {
        name = "location_id"
        type = "STRING"
        mode = "REQUIRED"
      },
      {
        name = "timestamp"
        type = "TIMESTAMP"
        mode = "REQUIRED"
      },
      {
        name = "camera_id"
        type = "STRING"
        mode = "NULLABLE"
      }
    ])
  }
}

# Output values
output "dataset_id" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.ride_intelligence.dataset_id
}

output "dataset_location" {
  description = "BigQuery dataset location"
  value       = google_bigquery_dataset.ride_intelligence.location
}

output "table_ids" {
  description = "List of created table IDs"
  value = [
    google_bigquery_table.street_imagery.table_id,
    google_bigquery_table.location_profiles.table_id,
    google_bigquery_table.historical_rides.table_id,
    google_bigquery_table.realtime_events.table_id,
    google_bigquery_table.api_pricing_requests.table_id,
    google_bigquery_table.location_embeddings.table_id,
    google_bigquery_table.competitor_analysis.table_id,
    google_bigquery_table.street_imagery_external.table_id
  ]
}
