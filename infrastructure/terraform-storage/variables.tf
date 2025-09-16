# Storage-Only Terraform Variables
# Configuration variables for storage bucket deployment

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

# Backup Configuration
variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

# Resource Tagging
variable "additional_labels" {
  description = "Additional labels to apply to resources"
  type        = map(string)
  default     = {}
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "data-engineering-team"
}
