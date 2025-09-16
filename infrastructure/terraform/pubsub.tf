# Pub/Sub Resources for Dynamic Pricing Intelligence
# Provisions topics and subscriptions for real-time data streaming

# Street imagery analysis topic
resource "google_pubsub_topic" "street_imagery_analysis" {
  name = "street-imagery-analysis"
  
  labels = merge(var.additional_labels, {
    component = "messaging"
    purpose   = "street-imagery"
    owner     = var.owner
  })
  
  message_retention_duration = "604800s" # 7 days
  
  schema_settings {
    schema   = google_pubsub_schema.street_imagery_schema.id
    encoding = "JSON"
  }
}

# Real-time events topic
resource "google_pubsub_topic" "realtime_events" {
  name = "realtime-events"
  
  labels = merge(var.additional_labels, {
    component = "messaging"
    purpose   = "realtime-events"
    owner     = var.owner
  })
  
  message_retention_duration = "86400s" # 1 day
  
  schema_settings {
    schema   = google_pubsub_schema.realtime_events_schema.id
    encoding = "JSON"
  }
}

# Pricing calculations topic
resource "google_pubsub_topic" "pricing_calculations" {
  name = "pricing-calculations"
  
  labels = merge(var.additional_labels, {
    component = "messaging"
    purpose   = "pricing-calculations"
    owner     = var.owner
  })
  
  message_retention_duration = "259200s" # 3 days
  
  schema_settings {
    schema   = google_pubsub_schema.pricing_calculations_schema.id
    encoding = "JSON"
  }
}

# Dead letter topic for failed messages
resource "google_pubsub_topic" "dead_letter" {
  name = "dead-letter-queue"
  
  labels = merge(var.additional_labels, {
    component = "messaging"
    purpose   = "dead-letter"
    owner     = var.owner
  })
  
  message_retention_duration = "2592000s" # 30 days
}

# Pub/Sub Schemas
resource "google_pubsub_schema" "street_imagery_schema" {
  name = "street-imagery-schema"
  type = "AVRO"
  
  definition = jsonencode({
    type = "record"
    name = "StreetImageryEvent"
    fields = [
      {
        name = "timestamp"
        type = "long"
      },
      {
        name = "location_id"
        type = "string"
      },
      {
        name = "image_url"
        type = "string"
      },
      {
        name = "camera_id"
        type = "string"
      },
      {
        name = "coordinates"
        type = {
          type = "record"
          name = "Coordinates"
          fields = [
            { name = "latitude", type = "double" },
            { name = "longitude", type = "double" }
          ]
        }
      },
      {
        name = "metadata"
        type = {
          type = "map"
          values = "string"
        }
      }
    ]
  })
}

resource "google_pubsub_schema" "realtime_events_schema" {
  name = "realtime-events-schema"
  type = "AVRO"
  
  definition = jsonencode({
    type = "record"
    name = "RealtimeEvent"
    fields = [
      {
        name = "timestamp"
        type = "long"
      },
      {
        name = "event_type"
        type = "string"
      },
      {
        name = "location_id"
        type = "string"
      },
      {
        name = "data"
        type = {
          type = "map"
          values = "string"
        }
      },
      {
        name = "source"
        type = "string"
      }
    ]
  })
}

resource "google_pubsub_schema" "pricing_calculations_schema" {
  name = "pricing-calculations-schema"
  type = "AVRO"
  
  definition = jsonencode({
    type = "record"
    name = "PricingCalculation"
    fields = [
      {
        name = "timestamp"
        type = {
          type = "long"
          logicalType = "timestamp-micros"
        }
      },
      {
        name = "location_id"
        type = "string"
      },
      {
        name = "base_price"
        type = "double"
      },
      {
        name = "surge_multiplier"
        type = "double"
      },
      {
        name = "final_price"
        type = "double"
      },
      {
        name = "confidence_score"
        type = "double"
      },
      {
        name = "factors"
        type = {
          type = "array"
          items = "string"
        }
      }
    ]
  })
}

# Subscriptions for image processing service
resource "google_pubsub_subscription" "image_processor_subscription" {
  name  = "image-processor-subscription"
  topic = google_pubsub_topic.street_imagery_analysis.name
  
  labels = merge(var.additional_labels, {
    component = "subscription"
    service   = "image-processor"
    owner     = var.owner
  })
  
  ack_deadline_seconds = 300
  
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }
  
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dead_letter.id
    max_delivery_attempts = 5
  }
  
  expiration_policy {
    ttl = "2678400s" # 31 days
  }
}

# Subscription for pricing engine service
resource "google_pubsub_subscription" "pricing_engine_subscription" {
  name  = "pricing-engine-subscription"
  topic = google_pubsub_topic.realtime_events.name
  
  labels = merge(var.additional_labels, {
    component = "subscription"
    service   = "pricing-engine"
    owner     = var.owner
  })
  
  ack_deadline_seconds = 60
  
  retry_policy {
    minimum_backoff = "5s"
    maximum_backoff = "300s"
  }
  
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dead_letter.id
    max_delivery_attempts = 5
  }
}

# Subscription for BigQuery streaming
resource "google_pubsub_subscription" "bigquery_streaming_subscription" {
  name  = "bigquery-streaming-subscription"
  topic = google_pubsub_topic.pricing_calculations.name
  
  labels = merge(var.additional_labels, {
    component = "subscription"
    service   = "bigquery-streaming"
    owner     = var.owner
  })
  
  ack_deadline_seconds = 120
  
  bigquery_config {
    table = "${google_bigquery_table.realtime_pricing.project}.${google_bigquery_table.realtime_pricing.dataset_id}.${google_bigquery_table.realtime_pricing.table_id}"
    
    use_topic_schema = true
    write_metadata   = true
  }
  
  depends_on = [
    google_bigquery_table.realtime_pricing,
    google_project_iam_member.pubsub_bigquery_permissions
  ]
}

# IAM bindings for Pub/Sub
resource "google_pubsub_topic_iam_member" "publisher_binding" {
  for_each = toset([
    google_pubsub_topic.street_imagery_analysis.name,
    google_pubsub_topic.realtime_events.name,
    google_pubsub_topic.pricing_calculations.name
  ])
  
  topic  = each.value
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_pubsub_subscription_iam_member" "subscriber_binding" {
  for_each = toset([
    google_pubsub_subscription.image_processor_subscription.name,
    google_pubsub_subscription.pricing_engine_subscription.name,
    google_pubsub_subscription.bigquery_streaming_subscription.name
  ])
  
  subscription = each.value
  role         = "roles/pubsub.subscriber"
  member       = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# Output values
output "pubsub_topics" {
  description = "Created Pub/Sub topic names"
  value = {
    street_imagery_analysis = google_pubsub_topic.street_imagery_analysis.name
    realtime_events        = google_pubsub_topic.realtime_events.name
    pricing_calculations   = google_pubsub_topic.pricing_calculations.name
    dead_letter           = google_pubsub_topic.dead_letter.name
  }
}

output "pubsub_subscriptions" {
  description = "Created Pub/Sub subscription names"
  value = {
    image_processor      = google_pubsub_subscription.image_processor_subscription.name
    pricing_engine      = google_pubsub_subscription.pricing_engine_subscription.name
    bigquery_streaming  = google_pubsub_subscription.bigquery_streaming_subscription.name
  }
}
