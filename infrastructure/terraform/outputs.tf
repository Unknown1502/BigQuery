# Dynamic Pricing Intelligence - Terraform Outputs
# Output values for the multimodal geospatial intelligence system

output "project_id" {
  description = "The GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "The GCP region"
  value       = var.region
}

output "environment" {
  description = "The environment name"
  value       = var.environment
}

# Network Outputs
output "vpc_network_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.vpc.name
}

output "vpc_network_id" {
  description = "ID of the VPC network"
  value       = google_compute_network.vpc.id
}

output "subnet_name" {
  description = "Name of the subnet"
  value       = google_compute_subnetwork.subnet.name
}

output "subnet_cidr" {
  description = "CIDR range of the subnet"
  value       = google_compute_subnetwork.subnet.ip_cidr_range
}

# GKE Cluster Outputs
output "gke_cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.primary.name
}

output "gke_cluster_endpoint" {
  description = "Endpoint of the GKE cluster"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "gke_cluster_ca_certificate" {
  description = "CA certificate of the GKE cluster"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "gke_cluster_location" {
  description = "Location of the GKE cluster"
  value       = google_container_cluster.primary.location
}

# Service Account Outputs
output "gke_service_account_email" {
  description = "Email of the GKE service account"
  value       = google_service_account.gke_nodes.email
}

# BigQuery Outputs
output "bigquery_dataset_id" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.ride_intelligence.dataset_id
}

output "bigquery_dataset_location" {
  description = "BigQuery dataset location"
  value       = google_bigquery_dataset.ride_intelligence.location
}

output "bigquery_tables" {
  description = "List of BigQuery table IDs"
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

# Connection Information
output "kubectl_connection_command" {
  description = "Command to connect kubectl to the GKE cluster"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.region} --project ${var.project_id}"
}

output "bigquery_connection_info" {
  description = "BigQuery connection information"
  value = {
    project_id = var.project_id
    dataset_id = google_bigquery_dataset.ride_intelligence.dataset_id
    location   = google_bigquery_dataset.ride_intelligence.location
  }
}

# API Endpoints (when deployed)
output "api_endpoints" {
  description = "API endpoints for the system"
  value = {
    pricing_api    = "https://api.${var.project_id}.com/api/v1/pricing"
    analytics_api  = "https://api.${var.project_id}.com/api/v1/analytics"
    admin_api      = "https://api.${var.project_id}.com/api/v1/admin"
    health_check   = "https://api.${var.project_id}.com/health"
    documentation  = "https://api.${var.project_id}.com/docs"
  }
}

# Monitoring and Logging
output "monitoring_dashboard_url" {
  description = "URL for the monitoring dashboard"
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${var.project_id}"
}

output "logging_url" {
  description = "URL for Cloud Logging"
  value       = "https://console.cloud.google.com/logs/query?project=${var.project_id}"
}

# Security Information
output "network_security" {
  description = "Network security configuration"
  value = {
    private_cluster     = true
    network_policy      = var.enable_network_policy
    master_cidr        = var.master_cidr
    authorized_networks = "0.0.0.0/0"
  }
}

# Cost Information
output "cost_optimization_features" {
  description = "Enabled cost optimization features"
  value = {
    preemptible_nodes = var.use_preemptible_nodes
    auto_scaling     = var.auto_scaling_enabled
    cost_monitoring  = var.enable_cost_optimization
  }
}

# Deployment Information
output "deployment_info" {
  description = "Information for deployment"
  value = {
    terraform_version = ">=1.0"
    provider_versions = {
      google      = "~>5.0"
      google-beta = "~>5.0"
    }
    deployment_timestamp = timestamp()
  }
}
