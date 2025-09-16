# Vertex AI Resources for Dynamic Pricing Intelligence
# Provisions ML infrastructure using supported Terraform resources

# Cloud Storage bucket for training data
resource "google_storage_bucket" "training_data" {
  name          = "${var.project_id}-training-data"
  location      = var.storage_location
  force_destroy = var.storage_force_destroy
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
  
  labels = var.additional_labels
}

# Model metadata storage
resource "google_storage_bucket_object" "crowd_detection_model_metadata" {
  name   = "ml-models/crowd-detection/model_metadata.json"
  bucket = "${var.project_id}-ml-models"
  content = jsonencode({
    display_name = "crowd-detection-model"
    description  = "Computer vision model for crowd detection and density analysis"
    model_type   = "image_classification"
    framework    = "tensorflow"
    version      = "v1.0"
    input_spec = {
      image_bytes = {
        type = "image"
        format = "base64"
      }
    }
    output_spec = {
      crowd_count = {
        type = "integer"
        description = "Number of people detected"
      }
      crowd_density = {
        type = "float"
        description = "Crowd density score"
      }
    }
    labels = var.additional_labels
  })
}

resource "google_storage_bucket_object" "activity_classification_model_metadata" {
  name   = "ml-models/activity-classification/model_metadata.json"
  bucket = "${var.project_id}-ml-models"
  content = jsonencode({
    display_name = "activity-classification-model"
    description  = "Computer vision model for activity and scene classification"
    model_type   = "image_classification"
    framework    = "tensorflow"
    version      = "v1.0"
    input_spec = {
      image_bytes = {
        type = "image"
        format = "base64"
      }
    }
    output_spec = {
      activity_type = {
        type = "string"
        description = "Detected activity type"
      }
      confidence_score = {
        type = "float"
        description = "Confidence score for prediction"
      }
    }
    labels = var.additional_labels
  })
}

resource "google_storage_bucket_object" "demand_forecasting_model_metadata" {
  name   = "ml-models/demand-forecasting/model_metadata.json"
  bucket = "${var.project_id}-ml-models"
  content = jsonencode({
    display_name = "demand-forecasting-model"
    description  = "ML model for demand forecasting based on multimodal data"
    model_type   = "tabular_regression"
    framework    = "scikit-learn"
    version      = "v1.0"
    input_spec = {
      features = {
        type = "array"
        description = "Feature vector for demand prediction"
      }
    }
    output_spec = {
      demand_forecast = {
        type = "float"
        description = "Predicted demand value"
      }
      confidence_interval = {
        type = "array"
        description = "Confidence interval for prediction"
      }
    }
    labels = var.additional_labels
  })
}

# Cloud Run services (commented out - requires container images to be built and pushed first)
# 
# # Cloud Run service for Crowd Detection
# resource "google_cloud_run_service" "crowd_detection_service" {
#   name     = "crowd-detection-service"
#   location = var.region
#   
#   template {
#     spec {
#       containers {
#         image = "gcr.io/${var.project_id}/crowd-detection:latest"
#         
#         ports {
#           container_port = 8080
#         }
#         
#         env {
#           name  = "MODEL_PATH"
#           value = "gs://${var.project_id}-ml-models/crowd-detection/model/"
#         }
#         
#         env {
#           name  = "PROJECT_ID"
#           value = var.project_id
#         }
#         
#         resources {
#           limits = {
#             cpu    = "2000m"
#             memory = "4Gi"
#           }
#         }
#       }
#       
#       container_concurrency = 10
#       timeout_seconds      = 300
#     }
#     
#     metadata {
#       annotations = {
#         "autoscaling.knative.dev/minScale" = "1"
#         "autoscaling.knative.dev/maxScale" = "10"
#         "run.googleapis.com/cpu-throttling" = "false"
#       }
#     }
#   }
#   
#   traffic {
#     percent         = 100
#     latest_revision = true
#   }
# }
# 
# # Cloud Run service for Activity Classification
# resource "google_cloud_run_service" "activity_classification_service" {
#   name     = "activity-classification-service"
#   location = var.region
#   
#   template {
#     spec {
#       containers {
#         image = "gcr.io/${var.project_id}/activity-classification:latest"
#         
#         ports {
#           container_port = 8080
#         }
#         
#         env {
#           name  = "MODEL_PATH"
#           value = "gs://${var.project_id}-ml-models/activity-classification/model/"
#         }
#         
#         env {
#           name  = "PROJECT_ID"
#           value = var.project_id
#         }
#         
#         resources {
#           limits = {
#             cpu    = "2000m"
#             memory = "4Gi"
#           }
#         }
#       }
#       
#       container_concurrency = 10
#       timeout_seconds      = 300
#     }
#     
#     metadata {
#       annotations = {
#         "autoscaling.knative.dev/minScale" = "1"
#         "autoscaling.knative.dev/maxScale" = "10"
#         "run.googleapis.com/cpu-throttling" = "false"
#       }
#     }
#   }
#   
#   traffic {
#     percent         = 100
#     latest_revision = true
#   }
# }
# 
# # Cloud Run service for Demand Forecasting
# resource "google_cloud_run_service" "demand_forecasting_service" {
#   name     = "demand-forecasting-service"
#   location = var.region
#   
#   template {
#     spec {
#       containers {
#         image = "gcr.io/${var.project_id}/demand-forecasting:latest"
#         
#         ports {
#           container_port = 8080
#         }
#         
#         env {
#           name  = "MODEL_PATH"
#           value = "gs://${var.project_id}-ml-models/demand-forecasting/model/"
#         }
#         
#         env {
#           name  = "PROJECT_ID"
#           value = var.project_id
#         }
#         
#         resources {
#           limits = {
#             cpu    = "1000m"
#             memory = "2Gi"
#           }
#         }
#       }
#       
#       container_concurrency = 20
#       timeout_seconds      = 300
#     }
#     
#     metadata {
#       annotations = {
#         "autoscaling.knative.dev/minScale" = "1"
#         "autoscaling.knative.dev/maxScale" = "5"
#         "run.googleapis.com/cpu-throttling" = "false"
#       }
#     }
#   }
#   
#   traffic {
#     percent         = 100
#     latest_revision = true
#   }
# }
# 
# # IAM bindings for Cloud Run services
# resource "google_cloud_run_service_iam_binding" "crowd_detection_invoker" {
#   location = google_cloud_run_service.crowd_detection_service.location
#   project  = google_cloud_run_service.crowd_detection_service.project
#   service  = google_cloud_run_service.crowd_detection_service.name
#   role     = "roles/run.invoker"
#   
#   members = [
#     "serviceAccount:${var.project_id}@appspot.gserviceaccount.com",
#   ]
# }
# 
# resource "google_cloud_run_service_iam_binding" "activity_classification_invoker" {
#   location = google_cloud_run_service.activity_classification_service.location
#   project  = google_cloud_run_service.activity_classification_service.project
#   service  = google_cloud_run_service.activity_classification_service.name
#   role     = "roles/run.invoker"
#   
#   members = [
#     "serviceAccount:${var.project_id}@appspot.gserviceaccount.com",
#   ]
# }
# 
# resource "google_cloud_run_service_iam_binding" "demand_forecasting_invoker" {
#   location = google_cloud_run_service.demand_forecasting_service.location
#   project  = google_cloud_run_service.demand_forecasting_service.project
#   service  = google_cloud_run_service.demand_forecasting_service.name
#   role     = "roles/run.invoker"
#   
#   members = [
#     "serviceAccount:${var.project_id}@appspot.gserviceaccount.com",
#   ]
# }

# Cloud Build trigger for Model Retraining (commented out - requires repository setup)
# resource "google_cloudbuild_trigger" "model_retraining_trigger" {
#   name        = "model-retraining-trigger"
#   description = "Automated trigger for retraining ML models"
#   
#   github {
#     owner = "your-github-username"
#     name  = "dynamic-pricing-intelligence"
#     push {
#       branch = "^main$"
#     }
#   }
#   
#   build {
#     step {
#       name = "gcr.io/cloud-builders/docker"
#       args = [
#         "build",
#         "-t", "gcr.io/${var.project_id}/model-training:latest",
#         "-f", "infrastructure/docker/Dockerfile.training",
#         "."
#       ]
#     }
#     
#     step {
#       name = "gcr.io/cloud-builders/docker"
#       args = ["push", "gcr.io/${var.project_id}/model-training:latest"]
#     }
#     
#     options {
#       machine_type = "E2_HIGHCPU_8"
#     }
#     
#     timeout = "3600s"
#   }
# }

# Cloud Monitoring alert policy for model performance
resource "google_monitoring_alert_policy" "model_performance_alert" {
  display_name = "Model Performance Alert"
  combiner     = "OR"
  
  conditions {
    display_name = "Model Prediction Latency"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"crowd-detection-service\" AND metric.type=\"run.googleapis.com/request_latencies\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5000  # 5 seconds in milliseconds
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_PERCENTILE_95"
      }
    }
  }
  
  conditions {
    display_name = "Model Error Rate"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"crowd-detection-service\" AND metric.type=\"run.googleapis.com/request_count\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 10  # 10 requests per minute
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }
  
  notification_channels = var.notification_channels
  
  alert_strategy {
    auto_close = "1800s"
  }
}

# Output values
output "ml_models_bucket" {
  description = "ML models storage bucket name"
  value       = "${var.project_id}-ml-models"
}

output "training_data_bucket" {
  description = "Training data storage bucket name"
  value       = google_storage_bucket.training_data.name
}

# Cloud Run service URLs (commented out - services are disabled)
# output "crowd_detection_service_url" {
#   description = "Crowd detection service URL"
#   value       = google_cloud_run_service.crowd_detection_service.status[0].url
# }
# 
# output "activity_classification_service_url" {
#   description = "Activity classification service URL"
#   value       = google_cloud_run_service.activity_classification_service.status[0].url
# }
# 
# output "demand_forecasting_service_url" {
#   description = "Demand forecasting service URL"
#   value       = google_cloud_run_service.demand_forecasting_service.status[0].url
# }
