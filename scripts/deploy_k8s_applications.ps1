# Deploy Kubernetes Applications for Dynamic Pricing Intelligence
# PowerShell script for Windows users

param(
    [string]$ProjectId = "geosptial-471213",
    [string]$Region = "us-central1",
    [string]$ClusterName = "dynamic-pricing-cluster",
    [string]$Namespace = "dynamic-pricing"
)

# Configuration
Write-Host "Starting Kubernetes Application Deployment" -ForegroundColor Blue
Write-Host "Project ID: $ProjectId"
Write-Host "Region: $Region"
Write-Host "Cluster: $ClusterName"
Write-Host "Namespace: $Namespace"
Write-Host ""

# Function to print status
function Write-Success {
    param([string]$Message)
    Write-Host "SUCCESS: $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "WARNING: $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
}

try {
    # Step 1: Configure Docker for GCR
    Write-Host "Configuring Docker for Google Container Registry..." -ForegroundColor Blue
    gcloud auth configure-docker --quiet
    if ($LASTEXITCODE -ne 0) { throw "Failed to configure Docker for GCR" }
    Write-Success "Docker configured for GCR"

    # Step 2: Build Docker Images
    Write-Host "Building Docker Images..." -ForegroundColor Blue

    # Build API Gateway image
    Write-Host "Building API Gateway image..."
    docker build -f infrastructure/docker/Dockerfile.api --target production -t "gcr.io/$ProjectId/api-gateway:latest" -t "gcr.io/$ProjectId/api-gateway:v1.0.0" .
    if ($LASTEXITCODE -ne 0) { throw "Failed to build API Gateway image" }
    Write-Success "API Gateway image built"

    # Build Image Processor image
    Write-Host "Building Image Processor image..."
    docker build -f infrastructure/docker/Dockerfile.processor --target production -t "gcr.io/$ProjectId/image-processor:latest" -t "gcr.io/$ProjectId/image-processor:v1.0.0" .
    if ($LASTEXITCODE -ne 0) { throw "Failed to build Image Processor image" }
    Write-Success "Image Processor image built"

    # Step 3: Push Images to GCR
    Write-Host "Pushing Images to Google Container Registry..." -ForegroundColor Blue

    docker push "gcr.io/$ProjectId/api-gateway:latest"
    if ($LASTEXITCODE -ne 0) { throw "Failed to push API Gateway image" }

    docker push "gcr.io/$ProjectId/api-gateway:v1.0.0"
    docker push "gcr.io/$ProjectId/image-processor:latest"
    docker push "gcr.io/$ProjectId/image-processor:v1.0.0"
    Write-Success "All images pushed to GCR"

    # Step 4: Update Kubernetes manifests with correct PROJECT_ID
    Write-Host "Updating Kubernetes manifests..." -ForegroundColor Blue

    # Create temporary directory for updated manifests
    $TempDir = "$env:TEMP\k8s-manifests"
    if (Test-Path $TempDir) { Remove-Item -Recurse -Force $TempDir }
    New-Item -ItemType Directory -Path $TempDir | Out-Null
    Copy-Item -Recurse "infrastructure\kubernetes\*" $TempDir

    # Replace PROJECT_ID placeholders in all YAML files
    Get-ChildItem $TempDir -Recurse -Filter "*.yaml" | ForEach-Object {
        (Get-Content $_.FullName) -replace 'PROJECT_ID', $ProjectId | Set-Content $_.FullName
    }
    Write-Success "Kubernetes manifests updated with project ID"

    # Step 5: Create necessary secrets and configmaps
    Write-Host "Creating Kubernetes secrets and configmaps..." -ForegroundColor Blue

    # Create GCP service account key secret (if credentials file exists)
    $CredFile = Get-ChildItem "credentials\$ProjectId-*.json" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($CredFile) {
        kubectl create secret generic gcp-service-account-key --from-file="key.json=$($CredFile.FullName)" --namespace=$Namespace --dry-run=client -o yaml | kubectl apply -f -
        Write-Success "GCP service account secret created"
    } else {
        Write-Warning "GCP service account key file not found, skipping secret creation"
    }

    # Create Redis credentials secret (placeholder)
    kubectl create secret generic redis-credentials --from-literal=url="redis://localhost:6379" --namespace=$Namespace --dry-run=client -o yaml | kubectl apply -f -
    Write-Success "Redis credentials secret created"

    # Create API Gateway config map
    kubectl create configmap api-gateway-config --from-literal=environment="production" --from-literal=log_level="INFO" --from-literal=bigquery_dataset="ride_intelligence" --namespace=$Namespace --dry-run=client -o yaml | kubectl apply -f -
    Write-Success "API Gateway config map created"

    # Step 6: Deploy applications
    Write-Host "Deploying applications to Kubernetes..." -ForegroundColor Blue

    # Deploy namespace first (if not already deployed)
    kubectl apply -f "$TempDir\namespace.yaml"
    Write-Success "Namespace deployed"

    # Deploy API Gateway
    kubectl apply -f "$TempDir\api-gateway\"
    Write-Success "API Gateway deployed"

    # Deploy Pricing Engine
    kubectl apply -f "$TempDir\pricing-engine\"
    Write-Success "Pricing Engine deployed"

    # Deploy Real-time Processor
    kubectl apply -f "$TempDir\real-time-processor\"
    Write-Success "Real-time Processor deployed"

    # Deploy Monitoring
    kubectl apply -f "$TempDir\monitoring\"
    Write-Success "Monitoring deployed"

    # Step 7: Wait for deployments to be ready
    Write-Host "Waiting for deployments to be ready..." -ForegroundColor Blue

    kubectl wait --for=condition=available --timeout=300s deployment/api-gateway -n $Namespace
    if ($LASTEXITCODE -eq 0) {
        Write-Success "API Gateway is ready"
    } else {
        Write-Warning "API Gateway may not be ready yet"
    }

    kubectl wait --for=condition=available --timeout=300s deployment/pricing-engine -n $Namespace
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Pricing Engine is ready"
    } else {
        Write-Warning "Pricing Engine may not be ready yet"
    }

    kubectl wait --for=condition=available --timeout=300s deployment/real-time-processor -n $Namespace
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Real-time Processor is ready"
    } else {
        Write-Warning "Real-time Processor may not be ready yet"
    }

    # Step 8: Display deployment status
    Write-Host ""
    Write-Host "Deployment Status:" -ForegroundColor Blue
    kubectl get pods -n $Namespace
    Write-Host ""
    kubectl get services -n $Namespace
    Write-Host ""

    # Step 9: Display next steps
    Write-Success "Deployment completed successfully!"
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Blue
    Write-Host "1. Test the API endpoints"
    Write-Host "2. Monitor application logs: kubectl logs -f deployment/api-gateway -n $Namespace"
    Write-Host "3. Check application metrics in GCP Console"
    Write-Host "4. Set up external load balancer if needed"
    Write-Host ""
    Write-Host "To access the applications:" -ForegroundColor Blue
    Write-Host "kubectl port-forward service/api-gateway-service 8080:80 -n $Namespace"
    Write-Host "Then visit: http://localhost:8080"

    # Cleanup temporary files
    Remove-Item -Recurse -Force $TempDir
    Write-Success "Cleanup completed"

} catch {
    Write-Error $_.Exception.Message
    exit 1
}

Write-Host ""
Write-Host "Deployment script finished successfully!" -ForegroundColor Green
