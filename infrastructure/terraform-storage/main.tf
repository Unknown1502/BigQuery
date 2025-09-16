# Storage-Only Terraform Configuration
# This file contains only the storage bucket resources for initial deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Configure the Google Cloud Provider
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Enable required APIs for storage
resource "google_project_service" "storage_apis" {
  for_each = toset([
    "storage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_dependent_services = false
  disable_on_destroy        = false
}

# Street imagery storage bucket
resource "google_storage_bucket" "street_imagery" {
  name          = "${var.project_id}-street-imagery"
  location      = var.storage_location
  force_destroy = var.storage_force_destroy
  
  labels = merge(var.additional_labels, {
    component = "data-storage"
    purpose   = "street-imagery"
    owner     = var.owner
  })
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  depends_on = [google_project_service.storage_apis]
}

# ML models storage bucket
resource "google_storage_bucket" "ml_models" {
  name          = "${var.project_id}-ml-models"
  location      = var.storage_location
  force_destroy = var.storage_force_destroy
  
  labels = merge(var.additional_labels, {
    component = "ml-storage"
    purpose   = "ml-models"
    owner     = var.owner
  })
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  depends_on = [google_project_service.storage_apis]
}

# Data processing temporary storage
resource "google_storage_bucket" "data_processing" {
  name          = "${var.project_id}-data-processing"
  location      = var.storage_location
  force_destroy = var.storage_force_destroy
  
  labels = merge(var.additional_labels, {
    component = "data-processing"
    purpose   = "temporary-storage"
    owner     = var.owner
  })
  
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }
  
  depends_on = [google_project_service.storage_apis]
}

# Analytics storage bucket
resource "google_storage_bucket" "analytics" {
  name          = "${var.project_id}-analytics"
  location      = var.storage_location
  force_destroy = var.storage_force_destroy
  
  labels = merge(var.additional_labels, {
    component = "analytics-storage"
    purpose   = "analytics-data"
    owner     = var.owner
  })
  
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
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  depends_on = [google_project_service.storage_apis]
}

# Backup storage bucket
resource "google_storage_bucket" "backups" {
  name          = "${var.project_id}-backups"
  location      = var.storage_location
  force_destroy = false
  
  labels = merge(var.additional_labels, {
    component = "backup-storage"
    purpose   = "system-backups"
    owner     = var.owner
  })
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = var.backup_retention_days
    }
    action {
      type = "Delete"
    }
  }
  
  depends_on = [google_project_service.storage_apis]
}

# Output values
output "storage_buckets" {
  description = "Created storage bucket names"
  value = {
    street_imagery   = google_storage_bucket.street_imagery.name
    ml_models        = google_storage_bucket.ml_models.name
    data_processing  = google_storage_bucket.data_processing.name
    analytics        = google_storage_bucket.analytics.name
    backups         = google_storage_bucket.backups.name
  }
}

output "storage_bucket_urls" {
  description = "Storage bucket URLs"
  value = {
    street_imagery   = google_storage_bucket.street_imagery.url
    ml_models        = google_storage_bucket.ml_models.url
    data_processing  = google_storage_bucket.data_processing.url
    analytics        = google_storage_bucket.analytics.url
    backups         = google_storage_bucket.backups.url
  }
}
