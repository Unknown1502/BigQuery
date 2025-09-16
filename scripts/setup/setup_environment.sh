#!/bin/bash

# Dynamic Pricing Intelligence - Environment Setup Script
# This script sets up the complete development and production environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=""
REGION="us-central1"
ZONE="us-central1-a"
ENVIRONMENT="development"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command_exists gcloud; then
        missing_tools+=("gcloud")
    fi
    
    if ! command_exists kubectl; then
        missing_tools+=("kubectl")
    fi
    
    if ! command_exists terraform; then
        missing_tools+=("terraform")
    fi
    
    if ! command_exists docker; then
        missing_tools+=("docker")
    fi
    
    if ! command_exists python3; then
        missing_tools+=("python3")
    fi
    
    if ! command_exists pip3; then
        missing_tools+=("pip3")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_error "Please install the missing tools and run this script again."
        exit 1
    fi
    
    print_success "All prerequisites are installed"
}

# Function to get user input
get_user_input() {
    print_status "Gathering configuration information..."
    
    if [ -z "$PROJECT_ID" ]; then
        read -p "Enter your GCP Project ID: " PROJECT_ID
    fi
    
    if [ -z "$PROJECT_ID" ]; then
        print_error "Project ID is required"
        exit 1
    fi
    
    read -p "Enter region (default: us-central1): " input_region
    REGION=${input_region:-$REGION}
    
    read -p "Enter environment (development/staging/production) [default: development]: " input_env
    ENVIRONMENT=${input_env:-$ENVIRONMENT}
    
    print_success "Configuration gathered successfully"
    print_status "Project ID: $PROJECT_ID"
    print_status "Region: $REGION"
    print_status "Environment: $ENVIRONMENT"
}

# Function to setup GCP authentication
setup_gcp_auth() {
    print_status "Setting up GCP authentication..."
    
    # Check if already authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        print_success "Already authenticated with GCP"
    else
        print_status "Please authenticate with GCP..."
        gcloud auth login
    fi
    
    # Set project
    gcloud config set project "$PROJECT_ID"
    gcloud config set compute/region "$REGION"
    gcloud config set compute/zone "$ZONE"
    
    print_success "GCP authentication configured"
}

# Function to enable required APIs
enable_apis() {
    print_status "Enabling required GCP APIs..."
    
    local apis=(
        "bigquery.googleapis.com"
        "storage-api.googleapis.com"
        "pubsub.googleapis.com"
        "cloudfunctions.googleapis.com"
        "aiplatform.googleapis.com"
        "container.googleapis.com"
        "compute.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "iam.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "cloudbuild.googleapis.com"
        "artifactregistry.googleapis.com"
        "servicenetworking.googleapis.com"
        "dns.googleapis.com"
        "cloudscheduler.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        print_status "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID"
    done
    
    print_success "All required APIs enabled"
}

# Function to create service accounts
create_service_accounts() {
    print_status "Creating service accounts..."
    
    # GKE nodes service account
    if ! gcloud iam service-accounts describe "gke-nodes@$PROJECT_ID.iam.gserviceaccount.com" >/dev/null 2>&1; then
        gcloud iam service-accounts create gke-nodes \
            --display-name="GKE Nodes Service Account" \
            --description="Service account for GKE cluster nodes"
        
        # Grant necessary roles
        local roles=(
            "roles/storage.objectAdmin"
            "roles/bigquery.dataEditor"
            "roles/pubsub.editor"
            "roles/aiplatform.user"
            "roles/monitoring.metricWriter"
            "roles/logging.logWriter"
        )
        
        for role in "${roles[@]}"; do
            gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                --member="serviceAccount:gke-nodes@$PROJECT_ID.iam.gserviceaccount.com" \
                --role="$role"
        done
    fi
    
    # Cloud Functions service account
    if ! gcloud iam service-accounts describe "cloud-functions-sa@$PROJECT_ID.iam.gserviceaccount.com" >/dev/null 2>&1; then
        gcloud iam service-accounts create cloud-functions-sa \
            --display-name="Cloud Functions Service Account" \
            --description="Service account for Cloud Functions"
    fi
    
    print_success "Service accounts created"
}

# Function to setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip and install build tools for PEP 517 compliance
    pip install --upgrade pip setuptools wheel
    
    # Install requirements with PEP 517 compliance
    if [ -f "requirements.txt" ]; then
        pip install --use-pep517 -r requirements.txt
    fi
    
    if [ -f "requirements-dev.txt" ] && [ "$ENVIRONMENT" = "development" ]; then
        pip install --use-pep517 -r requirements-dev.txt
    fi
    
    print_success "Python environment setup complete"
}

# Function to initialize Terraform
setup_terraform() {
    print_status "Setting up Terraform..."
    
    cd infrastructure/terraform
    
    # Create terraform.tfvars if it doesn't exist
    if [ ! -f "terraform.tfvars" ]; then
        cat > terraform.tfvars <<EOF
project_id = "$PROJECT_ID"
region = "$REGION"
environment = "$ENVIRONMENT"
owner = "$(whoami)"

# Network configuration
gke_subnet_cidr = "10.0.0.0/24"
gke_pods_cidr = "10.1.0.0/16"
gke_services_cidr = "10.2.0.0/16"
services_subnet_cidr = "10.3.0.0/24"

# Storage configuration
storage_location = "US"
storage_force_destroy = true

# Security configuration
ssh_source_ranges = ["$(curl -s ifconfig.me)/32"]
blocked_ip_ranges = []

# SSL configuration
ssl_domains = ["api.$PROJECT_ID.com"]
domain_name = "$PROJECT_ID.com"

# Monitoring configuration
alert_emails = ["admin@$PROJECT_ID.com"]
alert_notification_channel = ""

# Performance configuration
backup_retention_days = 30
model_performance_threshold = 0.85
EOF
    fi
    
    # Initialize Terraform
    terraform init
    
    # Validate configuration
    terraform validate
    
    print_success "Terraform initialized successfully"
    
    cd ../..
}

# Function to create BigQuery datasets
setup_bigquery() {
    print_status "Setting up BigQuery datasets..."
    
    # Create main dataset
    if ! bq ls -d "$PROJECT_ID:ride_intelligence" >/dev/null 2>&1; then
        bq mk --dataset \
            --description="Dynamic Pricing Intelligence Dataset" \
            --location=US \
            "$PROJECT_ID:ride_intelligence"
    fi
    
    # Create external tables for street imagery
    bq mk --external_table_definition=@data/schemas/bigquery_schemas.json \
        "$PROJECT_ID:ride_intelligence.street_imagery" || true
    
    print_success "BigQuery datasets created"
}

# Function to setup local development environment
setup_local_dev() {
    if [ "$ENVIRONMENT" = "development" ]; then
        print_status "Setting up local development environment..."
        
        # Copy environment template
        if [ ! -f ".env" ]; then
            cp .env.template .env
            print_warning "Please edit .env file with your specific configuration"
        fi
        
        # Start local services with Docker Compose
        if command_exists docker-compose; then
            print_status "Starting local development services..."
            docker-compose up -d redis postgres
            print_success "Local development services started"
        fi
        
        # Setup pre-commit hooks
        if command_exists pre-commit; then
            pre-commit install
            print_success "Pre-commit hooks installed"
        fi
    fi
}

# Function to run tests
run_tests() {
    if [ "$ENVIRONMENT" = "development" ]; then
        print_status "Running tests..."
        
        source venv/bin/activate
        
        # Run unit tests
        python -m pytest tests/unit/ -v
        
        # Run linting
        python -m flake8 src/ --max-line-length=88
        
        # Run type checking
        python -m mypy src/ --ignore-missing-imports
        
        print_success "All tests passed"
    fi
}

# Function to display next steps
display_next_steps() {
    print_success "Environment setup completed successfully!"
    echo
    print_status "Next steps:"
    echo "1. Review and edit the .env file with your specific configuration"
    echo "2. Review and customize infrastructure/terraform/terraform.tfvars"
    echo "3. Deploy infrastructure: cd infrastructure/terraform && terraform plan && terraform apply"
    echo "4. Deploy BigQuery functions: make deploy-bigquery"
    echo "5. Build and deploy services: make deploy-$ENVIRONMENT"
    echo "6. Run the comprehensive demo: python scripts/demo/comprehensive_demo.py"
    echo
    print_status "For development:"
    echo "- Activate Python environment: source venv/bin/activate"
    echo "- Start local services: docker-compose up -d"
    echo "- Run tests: make test"
    echo "- Start API server: make run-api"
    echo
    print_status "Documentation:"
    echo "- Architecture: docs/architecture/"
    echo "- API Documentation: docs/api/"
    echo "- Deployment Guide: docs/deployment/"
}

# Main execution
main() {
    echo "=========================================="
    echo "Dynamic Pricing Intelligence Setup"
    echo "=========================================="
    echo
    
    check_prerequisites
    get_user_input
    setup_gcp_auth
    enable_apis
    create_service_accounts
    setup_python_env
    setup_terraform
    setup_bigquery
    setup_local_dev
    
    if [ "$ENVIRONMENT" = "development" ]; then
        run_tests
    fi
    
    display_next_steps
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --project-id PROJECT_ID    GCP Project ID"
            echo "  --region REGION           GCP Region (default: us-central1)"
            echo "  --environment ENV         Environment (development/staging/production)"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main
