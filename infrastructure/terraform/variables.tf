# Dynamic Pricing Intelligence - Terraform Variables
# Configuration variables for the multimodal geospatial intelligence system

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone for zonal resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# Network Configuration
variable "subnet_cidr" {
  description = "CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "pods_cidr" {
  description = "CIDR range for GKE pods"
  type        = string
  default     = "10.1.0.0/16"
}

variable "services_cidr" {
  description = "CIDR range for GKE services"
  type        = string
  default     = "10.2.0.0/16"
}

variable "master_cidr" {
  description = "CIDR range for GKE master nodes"
  type        = string
  default     = "10.3.0.0/28"
}

# GKE Configuration
variable "node_machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "e2-standard-4"
}

variable "min_nodes" {
  description = "Minimum number of nodes in the node pool"
  type        = number
  default     = 1
}

variable "max_nodes" {
  description = "Maximum number of nodes in the node pool"
  type        = number
  default     = 10
}

variable "use_preemptible_nodes" {
  description = "Use preemptible nodes for cost savings"
  type        = bool
  default     = true
}

# BigQuery Configuration
variable "bigquery_location" {
  description = "Location for BigQuery datasets"
  type        = string
  default     = "US"
}

variable "bigquery_delete_contents_on_destroy" {
  description = "Delete BigQuery dataset contents on destroy"
  type        = bool
  default     = false
}

# Cloud Storage Configuration
variable "storage_location" {
  description = "Location for Cloud Storage buckets"
  type        = string
  default     = "US"
}

variable "storage_force_destroy" {
  description = "Force destroy Cloud Storage buckets"
  type        = bool
  default     = false
}

# Pub/Sub Configuration
variable "pubsub_message_retention_duration" {
  description = "Message retention duration for Pub/Sub topics"
  type        = string
  default     = "604800s" # 7 days
}

# Cloud Functions Configuration
variable "functions_runtime" {
  description = "Runtime for Cloud Functions"
  type        = string
  default     = "python39"
}

variable "functions_memory" {
  description = "Memory allocation for Cloud Functions"
  type        = number
  default     = 512
}

variable "functions_timeout" {
  description = "Timeout for Cloud Functions in seconds"
  type        = number
  default     = 540
}

# Vertex AI Configuration
variable "vertex_ai_region" {
  description = "Region for Vertex AI resources"
  type        = string
  default     = "us-central1"
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable monitoring and alerting"
  type        = bool
  default     = true
}

variable "notification_channels" {
  description = "List of notification channels for alerts"
  type        = list(string)
  default     = []
}

# Security Configuration
variable "enable_binary_authorization" {
  description = "Enable Binary Authorization for GKE"
  type        = bool
  default     = false
}

variable "enable_network_policy" {
  description = "Enable network policy for GKE"
  type        = bool
  default     = true
}

variable "enable_pod_security_policy" {
  description = "Enable Pod Security Policy for GKE"
  type        = bool
  default     = false
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "enable_point_in_time_recovery" {
  description = "Enable point-in-time recovery for applicable resources"
  type        = bool
  default     = true
}

# Cost Optimization
variable "enable_cost_optimization" {
  description = "Enable cost optimization features"
  type        = bool
  default     = true
}

variable "auto_scaling_enabled" {
  description = "Enable auto-scaling for applicable resources"
  type        = bool
  default     = true
}

# Development Configuration
variable "enable_debug_logging" {
  description = "Enable debug logging"
  type        = bool
  default     = false
}

variable "enable_development_features" {
  description = "Enable development-specific features"
  type        = bool
  default     = false
}

# Resource Tagging
variable "additional_labels" {
  description = "Additional labels to apply to resources"
  type        = map(string)
  default     = {}
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}

# BigQuery Configuration
variable "dataset_id" {
  description = "BigQuery dataset ID"
  type        = string
  default     = "ride_intelligence"
}

# Network Configuration Variables
variable "gke_subnet_cidr" {
  description = "CIDR range for GKE subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "gke_pods_cidr" {
  description = "CIDR range for GKE pods"
  type        = string
  default     = "10.1.0.0/16"
}

variable "gke_services_cidr" {
  description = "CIDR range for GKE services"
  type        = string
  default     = "10.2.0.0/16"
}

variable "services_subnet_cidr" {
  description = "CIDR range for services subnet"
  type        = string
  default     = "10.3.0.0/24"
}

variable "network_name" {
  description = "Name of the VPC network"
  type        = string
  default     = "pricing-intelligence-network"
}

# SSL Configuration
variable "ssl_domains" {
  description = "Domains for SSL certificate"
  type        = list(string)
  default     = []
}

variable "domain_name" {
  description = "Domain name for DNS configuration"
  type        = string
  default     = "pricing-intelligence.local"
}

# Security Configuration
variable "ssh_source_ranges" {
  description = "Source IP ranges allowed for SSH"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "blocked_ip_ranges" {
  description = "IP ranges to block in security policy"
  type        = list(string)
  default     = []
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "data-engineering-team"
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "engineering"
}

# API Configuration
variable "api_gateway_config" {
  description = "Configuration for API Gateway"
  type = object({
    min_instances = number
    max_instances = number
    cpu_limit     = string
    memory_limit  = string
  })
  default = {
    min_instances = 1
    max_instances = 10
    cpu_limit     = "1000m"
    memory_limit  = "2Gi"
  }
}

# Data Processing Configuration
variable "dataflow_config" {
  description = "Configuration for Dataflow jobs"
  type = object({
    max_workers        = number
    machine_type       = string
    disk_size_gb      = number
    use_public_ips    = bool
  })
  default = {
    max_workers        = 10
    machine_type       = "n1-standard-2"
    disk_size_gb      = 50
    use_public_ips    = false
  }
}

# ML Configuration
variable "ml_config" {
  description = "Configuration for ML workloads"
  type = object({
    enable_gpu           = bool
    gpu_type            = string
    gpu_count           = number
    model_serving_nodes = number
  })
  default = {
    enable_gpu           = false
    gpu_type            = "nvidia-tesla-t4"
    gpu_count           = 1
    model_serving_nodes = 2
  }
}

# Disaster Recovery Configuration
variable "disaster_recovery_config" {
  description = "Configuration for disaster recovery"
  type = object({
    enable_cross_region_backup = bool
    backup_region             = string
    rpo_hours                 = number
    rto_hours                 = number
  })
  default = {
    enable_cross_region_backup = true
    backup_region             = "us-east1"
    rpo_hours                 = 4
    rto_hours                 = 2
  }
}

# Cloud Functions Configuration Variables
variable "log_level" {
  description = "Log level for Cloud Functions and services"
  type        = string
  default     = "INFO"
  
  validation {
    condition     = contains(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], var.log_level)
    error_message = "Log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL."
  }
}

variable "vertex_ai_endpoint" {
  description = "Vertex AI endpoint URL for ML model predictions"
  type        = string
  default     = ""
}

variable "redis_host" {
  description = "Redis host for caching and session storage"
  type        = string
  default     = ""
}

variable "model_performance_threshold" {
  description = "Performance threshold for triggering model retraining"
  type        = number
  default     = 0.85
  
  validation {
    condition     = var.model_performance_threshold >= 0.0 && var.model_performance_threshold <= 1.0
    error_message = "Model performance threshold must be between 0.0 and 1.0."
  }
}

variable "alert_notification_channel" {
  description = "Notification channel ID for monitoring alerts"
  type        = string
  default     = ""
}

variable "api_gateway_internal_ip" {
  description = "Internal IP address for API Gateway DNS record"
  type        = string
  default     = "10.0.0.100"
  
  validation {
    condition     = can(regex("^(?:[0-9]{1,3}\\.){3}[0-9]{1,3}$", var.api_gateway_internal_ip))
    error_message = "API Gateway internal IP must be a valid IPv4 address."
  }
}
