# Cloud Storage Resources for Dynamic Pricing Intelligence
# Provisions storage buckets for street imagery, model artifacts, and data processing

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
}

# Bucket IAM bindings
resource "google_storage_bucket_iam_member" "street_imagery_readers" {
  bucket = google_storage_bucket.street_imagery.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_storage_bucket_iam_member" "street_imagery_writers" {
  bucket = google_storage_bucket.street_imagery.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_storage_bucket_iam_member" "ml_models_access" {
  bucket = google_storage_bucket.ml_models.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_storage_bucket_iam_member" "analytics_access" {
  bucket = google_storage_bucket.analytics.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_storage_bucket_iam_member" "data_processing_access" {
  bucket = google_storage_bucket.data_processing.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.gke_nodes.email}"
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
