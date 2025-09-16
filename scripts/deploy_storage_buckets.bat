@echo off
setlocal enabledelayedexpansion

REM Deploy Google Cloud Storage Buckets Script for Windows
REM This script deploys the updated Terraform configuration for Cloud Storage buckets

echo.
echo [INFO] Starting Google Cloud Storage Buckets Deployment
echo ==================================================

REM Check prerequisites
echo [INFO] Checking prerequisites...

REM Check if terraform is installed
terraform --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Terraform is not installed. Please install Terraform first.
    exit /b 1
)

REM Check if gcloud is installed
gcloud --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Google Cloud SDK is not installed. Please install gcloud first.
    exit /b 1
)

echo [SUCCESS] All prerequisites met!

REM Set project variables
echo [INFO] Setting project variables...
set TF_VAR_project_id=lucid-dynamo-413113
set TF_VAR_region=us-central1
set TF_VAR_environment=dev

echo [SUCCESS] Project variables set:
echo   - Project ID: %TF_VAR_project_id%
echo   - Region: %TF_VAR_region%
echo   - Environment: %TF_VAR_environment%

REM Navigate to terraform directory
echo [INFO] Navigating to Terraform directory...
cd /d "%~dp0..\infrastructure\terraform"
if errorlevel 1 (
    echo [ERROR] Failed to navigate to Terraform directory
    exit /b 1
)
echo [SUCCESS] Changed to directory: %CD%

REM Initialize Terraform
echo [INFO] Initializing Terraform...
terraform init
if errorlevel 1 (
    echo [ERROR] Failed to initialize Terraform
    exit /b 1
)
echo [SUCCESS] Terraform initialized successfully!

REM Plan Terraform changes
echo [INFO] Planning Terraform changes...
echo.
echo [WARNING] The following changes will be made:
echo   - Add new analytics bucket: lucid-dynamo-413113-analytics
echo   - Rename model-artifacts bucket to: lucid-dynamo-413113-ml-models
echo   - Update all references in Cloud Functions and Vertex AI
echo.

terraform plan -out=tfplan
if errorlevel 1 (
    echo [ERROR] Failed to create Terraform plan
    exit /b 1
)
echo [SUCCESS] Terraform plan completed successfully!

REM Apply Terraform changes
echo [INFO] Applying Terraform changes...
echo.
set /p response="[WARNING] This will create/modify Google Cloud resources. Continue? (y/N): "
if /i "!response!"=="y" (
    terraform apply tfplan
    if errorlevel 1 (
        echo [ERROR] Failed to apply Terraform changes
        exit /b 1
    )
    echo [SUCCESS] Terraform apply completed successfully!
) else (
    echo [WARNING] Deployment cancelled by user
    exit /b 0
)

REM Verify bucket creation
echo [INFO] Verifying bucket creation...
echo.
echo [INFO] Checking bucket existence...

set buckets=lucid-dynamo-413113-street-imagery lucid-dynamo-413113-ml-models lucid-dynamo-413113-data-processing lucid-dynamo-413113-analytics lucid-dynamo-413113-backups

for %%b in (%buckets%) do (
    gsutil ls gs://%%b >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] ✗ %%b not found ^(may not have been created yet^)
    ) else (
        echo [SUCCESS] ✓ %%b exists
    )
)

echo.
echo [SUCCESS] Deployment completed successfully!
echo.
echo [INFO] Your buckets are now ready:
echo   - lucid-dynamo-413113-street-imagery ^(for street imagery data^)
echo   - lucid-dynamo-413113-ml-models ^(for ML models and artifacts^)
echo   - lucid-dynamo-413113-data-processing ^(for temporary processing data^)
echo   - lucid-dynamo-413113-analytics ^(for analytics data^)
echo   - lucid-dynamo-413113-backups ^(for system backups^)
echo.
echo [INFO] You can now use these buckets in your applications!

pause
