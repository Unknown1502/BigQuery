# Networking Resources for Dynamic Pricing Intelligence
# Provisions VPC, subnets, firewall rules, and load balancers

# VPC Network
resource "google_compute_network" "main_vpc" {
  name                    = "${var.project_id}-vpc"
  auto_create_subnetworks = false
  description            = "Main VPC for dynamic pricing intelligence system"
  
  routing_mode = "REGIONAL"
}

# Subnet for GKE cluster
resource "google_compute_subnetwork" "gke_subnet" {
  name          = "${var.project_id}-gke-subnet"
  ip_cidr_range = var.gke_subnet_cidr
  region        = var.region
  network       = google_compute_network.main_vpc.id
  description   = "Subnet for GKE cluster nodes"
  
  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = var.gke_pods_cidr
  }
  
  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = var.gke_services_cidr
  }
  
  private_ip_google_access = true
  
  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling       = 0.5
    metadata           = "INCLUDE_ALL_METADATA"
  }
}

# Subnet for Cloud Functions and other services
resource "google_compute_subnetwork" "services_subnet" {
  name          = "${var.project_id}-services-subnet"
  ip_cidr_range = var.services_subnet_cidr
  region        = var.region
  network       = google_compute_network.main_vpc.id
  description   = "Subnet for Cloud Functions and other services"
  
  private_ip_google_access = true
  
  log_config {
    aggregation_interval = "INTERVAL_5_MIN"
    flow_sampling       = 0.3
    metadata           = "INCLUDE_ALL_METADATA"
  }
}

# Cloud Router for NAT Gateway
resource "google_compute_router" "main_router" {
  name    = "${var.project_id}-router"
  region  = var.region
  network = google_compute_network.main_vpc.id
  
  description = "Router for NAT gateway and VPN connections"
}

# Cloud NAT for outbound internet access
resource "google_compute_router_nat" "main_nat" {
  name                               = "${var.project_id}-nat"
  router                            = google_compute_router.main_router.name
  region                            = var.region
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
  
  min_ports_per_vm = 64
  max_ports_per_vm = 512
}

# Global Load Balancer IP
resource "google_compute_global_address" "lb_ip" {
  name         = "${var.project_id}-lb-ip"
  description  = "Static IP for global load balancer"
  address_type = "EXTERNAL"
}

# SSL Certificate for HTTPS (conditional on domains being provided)
resource "google_compute_managed_ssl_certificate" "ssl_cert" {
  count = length(var.ssl_domains) > 0 ? 1 : 0
  
  name = "${var.project_id}-ssl-cert"
  
  managed {
    domains = var.ssl_domains
  }
  
  description = "Managed SSL certificate for dynamic pricing API"
}

# Firewall Rules

# Allow internal communication within VPC
resource "google_compute_firewall" "allow_internal" {
  name    = "${var.project_id}-allow-internal"
  network = google_compute_network.main_vpc.name
  
  description = "Allow internal communication within VPC"
  
  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "icmp"
  }
  
  source_ranges = [
    var.gke_subnet_cidr,
    var.gke_pods_cidr,
    var.gke_services_cidr,
    var.services_subnet_cidr
  ]
  
  priority = 1000
}

# Allow HTTPS traffic from internet
resource "google_compute_firewall" "allow_https" {
  name    = "${var.project_id}-allow-https"
  network = google_compute_network.main_vpc.name
  
  description = "Allow HTTPS traffic from internet"
  
  allow {
    protocol = "tcp"
    ports    = ["443"]
  }
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["https-server"]
  
  priority = 1000
}

# Allow HTTP traffic for health checks
resource "google_compute_firewall" "allow_http_health_check" {
  name    = "${var.project_id}-allow-http-health-check"
  network = google_compute_network.main_vpc.name
  
  description = "Allow HTTP traffic for health checks"
  
  allow {
    protocol = "tcp"
    ports    = ["8080", "8000"]
  }
  
  source_ranges = [
    "130.211.0.0/22",
    "35.191.0.0/16"
  ]
  target_tags = ["health-check"]
  
  priority = 1000
}

# Allow SSH access for debugging (restricted)
resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.project_id}-allow-ssh"
  network = google_compute_network.main_vpc.name
  
  description = "Allow SSH access for debugging"
  
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  
  source_ranges = var.ssh_source_ranges
  target_tags   = ["ssh-access"]
  
  priority = 1000
}

# Deny all other traffic
resource "google_compute_firewall" "deny_all" {
  name    = "${var.project_id}-deny-all"
  network = google_compute_network.main_vpc.name
  
  description = "Deny all other traffic"
  
  deny {
    protocol = "all"
  }
  
  source_ranges = ["0.0.0.0/0"]
  
  priority = 65534
}

# Network Security Policy
resource "google_compute_security_policy" "api_security_policy" {
  name        = "${var.project_id}-api-security-policy"
  description = "Security policy for API endpoints"
  
  # Rate limiting rule
  rule {
    action   = "rate_based_ban"
    priority = "1000"
    
    match {
      versioned_expr = "SRC_IPS_V1"
      
      config {
        src_ip_ranges = ["*"]
      }
    }
    
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      
      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }
      
      ban_duration_sec = 300
    }
    
    description = "Rate limit: 100 requests per minute per IP"
  }
  
  # Block known bad IPs (only if blocked_ip_ranges is not empty)
  dynamic "rule" {
    for_each = length(var.blocked_ip_ranges) > 0 ? [1] : []
    content {
      action   = "deny(403)"
      priority = "2000"
      
      match {
        versioned_expr = "SRC_IPS_V1"
        
        config {
          src_ip_ranges = var.blocked_ip_ranges
        }
      }
      
      description = "Block known malicious IP ranges"
    }
  }
  
  # Allow all other traffic
  rule {
    action   = "allow"
    priority = "2147483647"
    
    match {
      versioned_expr = "SRC_IPS_V1"
      
      config {
        src_ip_ranges = ["*"]
      }
    }
    
    description = "Default allow rule"
  }
}

# Private Service Connection for managed services
resource "google_compute_global_address" "private_service_connection" {
  name          = "${var.project_id}-private-service-connection"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.main_vpc.id
  
  description = "IP range for private service connection"
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.main_vpc.id
  service                = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_service_connection.name]
}

# DNS Zone for internal services
resource "google_dns_managed_zone" "internal_zone" {
  name        = "${var.project_id}-internal-zone"
  dns_name    = "internal.${var.domain_name}."
  description = "Internal DNS zone for service discovery"
  
  visibility = "private"
  
  private_visibility_config {
    networks {
      network_url = google_compute_network.main_vpc.id
    }
  }
  
  labels = merge(var.additional_labels, {
    component = "networking"
    purpose   = "internal-dns"
    owner     = var.owner
  })
}

# DNS records for internal services
resource "google_dns_record_set" "api_gateway_internal" {
  name = "api-gateway.internal.${var.domain_name}."
  type = "A"
  ttl  = 300
  
  managed_zone = google_dns_managed_zone.internal_zone.name
  
  rrdatas = [var.api_gateway_internal_ip]
}

resource "google_dns_record_set" "bigquery_internal" {
  name = "bigquery.internal.${var.domain_name}."
  type = "CNAME"
  ttl  = 300
  
  managed_zone = google_dns_managed_zone.internal_zone.name
  
  rrdatas = ["bigquery.googleapis.com."]
}

# Network Endpoint Group for load balancer
resource "google_compute_global_network_endpoint_group" "api_neg" {
  name                  = "${var.project_id}-api-neg"
  network_endpoint_type = "INTERNET_FQDN_PORT"
  default_port          = 443
  
  description = "Network endpoint group for API services"
}

# Output values
output "network_info" {
  description = "Network configuration details"
  value = {
    vpc_name           = google_compute_network.main_vpc.name
    vpc_id            = google_compute_network.main_vpc.id
    gke_subnet_name   = google_compute_subnetwork.gke_subnet.name
    services_subnet_name = google_compute_subnetwork.services_subnet.name
    router_name       = google_compute_router.main_router.name
    nat_name          = google_compute_router_nat.main_nat.name
  }
}

output "load_balancer_ip" {
  description = "Global load balancer IP address"
  value       = google_compute_global_address.lb_ip.address
}

output "ssl_certificate" {
  description = "SSL certificate details"
  value = length(var.ssl_domains) > 0 ? {
    name    = google_compute_managed_ssl_certificate.ssl_cert[0].name
    domains = google_compute_managed_ssl_certificate.ssl_cert[0].managed[0].domains
  } : {
    name    = ""
    domains = []
  }
}

output "security_policy" {
  description = "Security policy details"
  value = {
    name = google_compute_security_policy.api_security_policy.name
    id   = google_compute_security_policy.api_security_policy.id
  }
}

output "dns_zone" {
  description = "Internal DNS zone details"
  value = {
    name     = google_dns_managed_zone.internal_zone.name
    dns_name = google_dns_managed_zone.internal_zone.dns_name
  }
}
