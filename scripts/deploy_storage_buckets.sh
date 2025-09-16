#!/bin/bash

# Deploy Google Cloud Storage Buckets Script
# This script deploys the updated Terraform configuration for Cloud Storage buckets

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform is not installed. Please install Terraform first."
        exit 1
    fi
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        print_error "Google Cloud SDK is not installed. Please install gcloud first."
        exit 1
    fi
    
    # Check if authenticated with gcloud
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_error "Not authenticated with Google Cloud. Please run 'gcloud auth login' first."
        exit 1
    fi
    
    print_success "All prerequisites met!"
}

# Set project variables
set_project_variables() {
    print_status "Setting project variables..."
    
    # Set project ID
    export TF_VAR_project_id="lucid-dynamo-413113"
    
    # Set other common variables
    export TF_VAR_region="us-central1"
    export TF_VAR_environment="dev"
    
    print_success "Project variables set:"
    echo "  - Project ID: $TF_VAR_project_id"
    echo "  - Region: $TF_VAR_region"
    echo "  - Environment: $TF_VAR_environment"
}

# Navigate to terraform directory
navigate_to_terraform() {
    print_status "Navigating to Terraform directory..."
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    TERRAFORM_DIR="$SCRIPT_DIR/../infrastructure/terraform"
    
    if [ ! -d "$TERRAFORM_DIR" ]; then
        print_error "Terraform directory not found: $TERRAFORM_DIR"
        exit 1
    fi
    
    cd "$TERRAFORM_DIR"
    print_success "Changed to directory: $(pwd)"
}

# Initialize Terraform
init_terraform() {
    print_status "Initializing Terraform..."
    
    if terraform init; then
        print_success "Terraform initialized successfully!"
    else
        print_error "Failed to initialize Terraform"
        exit 1
    fi
}

# Plan Terraform changes
plan_terraform() {
    print_status "Planning Terraform changes..."
    
    echo ""
    print_warning "The following changes will be made:"
    echo "  - Add new analytics bucket: lucid-dynamo-413113-analytics"
    echo "  - Rename model-artifacts bucket to: lucid-dynamo-413113-ml-models"
    echo "  - Update all references in Cloud Functions and Vertex AI"
    echo ""
    
    if terraform plan -out=tfplan; then
        print_success "Terraform plan completed successfully!"
        return 0
    else
        print_error "Failed to create Terraform plan"
        exit 1
    fi
}

# Apply Terraform changes
apply_terraform() {
    print_status "Applying Terraform changes..."
    
    echo ""
    print_warning "This will create/modify Google Cloud resources. Continue? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        if terraform apply tfplan; then
            print_success "Terraform apply completed successfully!"
        else
            print_error "Failed to apply Terraform changes"
            exit 1
        fi
    else
        print_warning "Deployment cancelled by user"
        exit 0
    fi
}

# Verify bucket creation
verify_buckets() {
    print_status "Verifying bucket creation..."
    
    local buckets=(
        "lucid-dynamo-413113-street-imagery"
        "lucid-dynamo-413113-ml-models"
        "lucid-dynamo-413113-data-processing"
        "lucid-dynamo-413113-analytics"
        "lucid-dynamo-413113-backups"
    )
    
    echo ""
    print_status "Checking bucket existence..."
    
    for bucket in "${buckets[@]}"; do
        if gsutil ls "gs://$bucket" &> /dev/null; then
            print_success "✓ $bucket exists"
        else
            print_warning "✗ $bucket not found (may not have been created yet)"
        fi
    done
}

# Main execution
main() {
    echo ""
    print_status "Starting Google Cloud Storage Buckets Deployment"
    echo "=================================================="
    
    check_prerequisites
    set_project_variables
    navigate_to_terraform
    init_terraform
    plan_terraform
    apply_terraform
    verify_buckets
    
    echo ""
    print_success "Deployment completed successfully!"
    echo ""
    print_status "Your buckets are now ready:"
    echo "  - lucid-dynamo-413113-street-imagery (for street imagery data)"
    echo "  - lucid-dynamo-413113-ml-models (for ML models and artifacts)"
    echo "  - lucid-dynamo-413113-data-processing (for temporary processing data)"
    echo "  - lucid-dynamo-413113-analytics (for analytics data)"
    echo "  - lucid-dynamo-413113-backups (for system backups)"
    echo ""
    print_status "You can now use these buckets in your applications!"
}

# Run main function
main "$@"
