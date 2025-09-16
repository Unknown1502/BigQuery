@echo off
REM Deploy Kubernetes Applications for Dynamic Pricing Intelligence
REM This script builds Docker images, pushes to GCR, and deploys to GKE

setlocal enabledelayedexpansion

REM Configuration
set PROJECT_ID=geosptial-471213
set REGION=us-central1
set CLUSTER_NAME=dynamic-pricing-cluster
set NAMESPACE=dynamic-pricing

echo Starting Kubernetes Application Deployment
echo Project ID: %PROJECT_ID%
echo Region: %REGION%
echo Cluster: %CLUSTER_NAME%
echo Namespace: %NAMESPACE%
echo.

REM Step 1: Configure Docker for GCR
echo Configuring Docker for Google Container Registry...
gcloud auth configure-docker --quiet
if %errorlevel% neq 0 (
    echo ERROR: Failed to configure Docker for GCR
    exit /b 1
)
echo SUCCESS: Docker configured for GCR

REM Step 2: Build Docker Images
echo Building Docker Images...

REM Build API Gateway image
echo Building API Gateway image...
docker build -f infrastructure/docker/Dockerfile.api --target production -t gcr.io/%PROJECT_ID%/api-gateway:latest -t gcr.io/%PROJECT_ID%/api-gateway:v1.0.0 .
if %errorlevel% neq 0 (
    echo ERROR: Failed to build API Gateway image
    exit /b 1
)
echo SUCCESS: API Gateway image built

REM Build Image Processor image
echo Building Image Processor image...
docker build -f infrastructure/docker/Dockerfile.processor --target production -t gcr.io/%PROJECT_ID%/image-processor:latest -t gcr.io/%PROJECT_ID%/image-processor:v1.0.0 .
if %errorlevel% neq 0 (
    echo ERROR: Failed to build Image Processor image
    exit /b 1
)
echo SUCCESS: Image Processor image built

REM Step 3: Push Images to GCR
echo Pushing Images to Google Container Registry...

docker push gcr.io/%PROJECT_ID%/api-gateway:latest
if %errorlevel% neq 0 (
    echo ERROR: Failed to push API Gateway image
    exit /b 1
)

docker push gcr.io/%PROJECT_ID%/api-gateway:v1.0.0
docker push gcr.io/%PROJECT_ID%/image-processor:latest
docker push gcr.io/%PROJECT_ID%/image-processor:v1.0.0
echo SUCCESS: All images pushed to GCR

REM Step 4: Update Kubernetes manifests with correct PROJECT_ID
echo Updating Kubernetes manifests...

REM Create temporary directory for updated manifests
if exist "%TEMP%\k8s-manifests" rmdir /s /q "%TEMP%\k8s-manifests"
mkdir "%TEMP%\k8s-manifests"
xcopy /s /e "infrastructure\kubernetes\*" "%TEMP%\k8s-manifests\"

REM Replace PROJECT_ID placeholders in all YAML files using PowerShell
powershell -Command "(Get-ChildItem '%TEMP%\k8s-manifests' -Recurse -Filter '*.yaml') | ForEach-Object { (Get-Content $_.FullName) -replace 'PROJECT_ID', '%PROJECT_ID%' | Set-Content $_.FullName }"
echo SUCCESS: Kubernetes manifests updated with project ID

REM Step 5: Create necessary secrets and configmaps
echo Creating Kubernetes secrets and configmaps...

REM Create GCP service account key secret (if credentials file exists)
if exist "credentials\%PROJECT_ID%-*.json" (
    for %%f in (credentials\%PROJECT_ID%-*.json) do (
        kubectl create secret generic gcp-service-account-key --from-file=key.json="%%f" --namespace=%NAMESPACE% --dry-run=client -o yaml | kubectl apply -f -
    )
    echo SUCCESS: GCP service account secret created
) else (
    echo WARNING: GCP service account key file not found, skipping secret creation
)

REM Create Redis credentials secret (placeholder)
kubectl create secret generic redis-credentials --from-literal=url="redis://localhost:6379" --namespace=%NAMESPACE% --dry-run=client -o yaml | kubectl apply -f -
echo SUCCESS: Redis credentials secret created

REM Create API Gateway config map
kubectl create configmap api-gateway-config --from-literal=environment="production" --from-literal=log_level="INFO" --from-literal=bigquery_dataset="ride_intelligence" --namespace=%NAMESPACE% --dry-run=client -o yaml | kubectl apply -f -
echo SUCCESS: API Gateway config map created

REM Step 6: Deploy applications
echo Deploying applications to Kubernetes...

REM Deploy namespace first (if not already deployed)
kubectl apply -f "%TEMP%\k8s-manifests\namespace.yaml"
echo SUCCESS: Namespace deployed

REM Deploy API Gateway
kubectl apply -f "%TEMP%\k8s-manifests\api-gateway\"
echo SUCCESS: API Gateway deployed

REM Deploy Pricing Engine
kubectl apply -f "%TEMP%\k8s-manifests\pricing-engine\"
echo SUCCESS: Pricing Engine deployed

REM Deploy Real-time Processor
kubectl apply -f "%TEMP%\k8s-manifests\real-time-processor\"
echo SUCCESS: Real-time Processor deployed

REM Deploy Monitoring
kubectl apply -f "%TEMP%\k8s-manifests\monitoring\"
echo SUCCESS: Monitoring deployed

REM Step 7: Wait for deployments to be ready
echo Waiting for deployments to be ready...

kubectl wait --for=condition=available --timeout=300s deployment/api-gateway -n %NAMESPACE%
if %errorlevel% equ 0 (
    echo SUCCESS: API Gateway is ready
) else (
    echo WARNING: API Gateway may not be ready yet
)

kubectl wait --for=condition=available --timeout=300s deployment/pricing-engine -n %NAMESPACE%
if %errorlevel% equ 0 (
    echo SUCCESS: Pricing Engine is ready
) else (
    echo WARNING: Pricing Engine may not be ready yet
)

kubectl wait --for=condition=available --timeout=300s deployment/real-time-processor -n %NAMESPACE%
if %errorlevel% equ 0 (
    echo SUCCESS: Real-time Processor is ready
) else (
    echo WARNING: Real-time Processor may not be ready yet
)

REM Step 8: Display deployment status
echo.
echo Deployment Status:
kubectl get pods -n %NAMESPACE%
echo.
kubectl get services -n %NAMESPACE%
echo.

REM Step 9: Display next steps
echo SUCCESS: Deployment completed!
echo.
echo Next steps:
echo 1. Test the API endpoints
echo 2. Monitor application logs: kubectl logs -f deployment/api-gateway -n %NAMESPACE%
echo 3. Check application metrics in GCP Console
echo 4. Set up external load balancer if needed
echo.
echo To access the applications:
echo kubectl port-forward service/api-gateway-service 8080:80 -n %NAMESPACE%
echo Then visit: http://localhost:8080

REM Cleanup temporary files
rmdir /s /q "%TEMP%\k8s-manifests"
echo SUCCESS: Cleanup completed

echo.
echo Deployment script finished successfully!
pause
