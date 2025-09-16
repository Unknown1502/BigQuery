# Dynamic Pricing Intelligence - Main Terraform Configuration
# Provisions GCP resources for the multimodal geospatial intelligence system

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
  
  # Using local backend for initial deployment
  # Can be migrated to GCS backend later after buckets are created
}

# Configure the Google Cloud Provider
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Local values for resource naming and tagging
locals {
  name_prefix = "dynamic-pricing"
  
  common_labels = {
    project     = "dynamic-pricing-intelligence"
    environment = var.environment
    managed_by  = "terraform"
    team        = "data-engineering"
  }
  
  # Network configuration
  network_name    = "${local.name_prefix}-vpc"
  subnet_name     = "${local.name_prefix}-subnet"
  
  # BigQuery configuration
  dataset_id      = "ride_intelligence"
  
  # GKE configuration
  cluster_name    = "${local.name_prefix}-cluster"
  
  # Pub/Sub topics
  pubsub_topics = [
    "street-imagery",
    "weather-data", 
    "traffic-data",
    "competitor-pricing",
    "pricing-triggers",
    "pricing-results"
  ]
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "bigquery.googleapis.com",
    "pubsub.googleapis.com",
    "cloudfunctions.googleapis.com",
    "dataflow.googleapis.com",
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
    "servicenetworking.googleapis.com",
    "cloudbuild.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_dependent_services = false
  disable_on_destroy        = false
}

# Create VPC network
resource "google_compute_network" "vpc" {
  name                    = local.network_name
  auto_create_subnetworks = false
  mtu                     = 1460
  
  depends_on = [google_project_service.required_apis]
}

# Create subnet for GKE cluster
resource "google_compute_subnetwork" "subnet" {
  name          = local.subnet_name
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc.id
  
  # Enable private Google access for nodes without external IPs
  private_ip_google_access = true
  
  # Secondary IP ranges for GKE pods and services
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }
}

# Note: Firewall rules are defined in networking.tf

# Cloud NAT for outbound internet access
resource "google_compute_router" "router" {
  name    = "${local.name_prefix}-router"
  region  = var.region
  network = google_compute_network.vpc.id
}

resource "google_compute_router_nat" "nat" {
  name                               = "${local.name_prefix}-nat"
  router                            = google_compute_router.router.name
  region                            = var.region
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Service Account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "${local.name_prefix}-gke-nodes"
  display_name = "GKE Nodes Service Account"
  description  = "Service account for GKE cluster nodes"
}

# Service Account for data engineering team
resource "google_service_account" "data_engineering" {
  account_id   = "data-engineering"
  display_name = "Data Engineering Service Account"
  description  = "Service account for data engineering operations"
}

# Service Account for Terraform operations
resource "google_service_account" "terraform_sa" {
  account_id   = "terraform-operations"
  display_name = "Terraform Operations Service Account"
  description  = "Service account for Terraform infrastructure operations"
}

# IAM bindings for Terraform service account
resource "google_project_iam_member" "terraform_sa_roles" {
  for_each = toset([
    "roles/pubsub.admin",
    "roles/bigquery.admin",
    "roles/storage.admin",
    "roles/compute.admin",
    "roles/container.admin",
    "roles/iam.serviceAccountAdmin",
    "roles/resourcemanager.projectIamAdmin"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.terraform_sa.email}"
}

# IAM bindings for data engineering service account
resource "google_project_iam_member" "data_engineering_roles" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/pubsub.editor",
    "roles/storage.objectAdmin"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.data_engineering.email}"
}

# IAM permissions for Pub/Sub to BigQuery integration
resource "google_project_iam_member" "pubsub_bigquery_permissions" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.metadataViewer"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

# Get current project information
data "google_project" "current" {
  project_id = var.project_id
}

resource "google_project_iam_member" "gke_nodes_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/storage.objectViewer",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/pubsub.editor",
    "roles/aiplatform.user"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# GKE Cluster
resource "google_container_cluster" "primary" {
  name     = local.cluster_name
  location = var.region
  
  # Network configuration
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
  
  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1
  
  # Networking configuration
  networking_mode = "VPC_NATIVE"
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  
  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.master_cidr
  }
  
  # Master authorized networks
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All networks"
    }
  }
  
  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }
    
    horizontal_pod_autoscaling {
      disabled = false
    }
    
    network_policy_config {
      disabled = false
    }
    
    gcp_filestore_csi_driver_config {
      enabled = true
    }
  }
  
  # Network policy
  network_policy {
    enabled = true
  }
  
  # Maintenance policy - using daily window instead of recurring window to avoid date issues
  maintenance_policy {
    daily_maintenance_window {
      start_time = "02:00"
    }
  }
  
  # Logging and monitoring
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"
  
  depends_on = [
    google_project_service.required_apis,
    google_compute_subnetwork.subnet
  ]
}

# Primary node pool for general workloads
resource "google_container_node_pool" "primary_nodes" {
  name       = "${local.cluster_name}-primary-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  
  # Auto-scaling configuration
  autoscaling {
    min_node_count = var.min_nodes
    max_node_count = var.max_nodes
  }
  
  # Node configuration
  node_config {
    preemptible  = var.use_preemptible_nodes
    machine_type = var.node_machine_type
    disk_size_gb = 50
    disk_type    = "pd-ssd"
    
    # Service account
    service_account = google_service_account.gke_nodes.email
    
    # OAuth scopes
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    # Labels
    labels = merge(local.common_labels, {
      node_pool = "primary"
    })
    
    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
    
    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Shielded instance configuration
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
  
  # Node management
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  # Upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# High-memory node pool for ML workloads
resource "google_container_node_pool" "ml_nodes" {
  name       = "${local.cluster_name}-ml-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  
  # Auto-scaling configuration
  autoscaling {
    min_node_count = 0
    max_node_count = 3
  }
  
  # Node configuration
  node_config {
    preemptible  = var.use_preemptible_nodes
    machine_type = "n1-highmem-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    # Service account
    service_account = google_service_account.gke_nodes.email
    
    # OAuth scopes
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    # Labels
    labels = merge(local.common_labels, {
      node_pool = "ml"
      workload  = "machine-learning"
    })
    
    # Taints for ML workloads
    taint {
      key    = "workload"
      value  = "ml"
      effect = "NO_SCHEDULE"
    }
    
    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
    
    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Shielded instance configuration
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
  
  # Node management
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  # Upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# Note: Cloud Storage resources are defined in cloud-storage.tf
# Note: Bucket IAM bindings are defined in cloud-storage.tf

# Output important values
output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.vpc.name
}

# Note: Subnet output is defined in outputs.tf

# Note: BigQuery module output removed for initial storage-only deployment
# output "bigquery_dataset_id" {
#   description = "BigQuery dataset ID"
#   value       = module.bigquery.dataset_id
# }

# Note: Storage bucket outputs are defined in cloud-storage.tf

# Note: Pub/Sub module output removed for initial storage-only deployment
# output "pubsub_topics" {
#   description = "Pub/Sub topic names"
#   value       = module.pubsub.topic_names
# }
