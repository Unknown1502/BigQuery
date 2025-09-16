@echo off
setlocal enabledelayedexpansion

REM Improved Deploy Google Cloud Storage Buckets Script for Windows
REM This script handles various Terraform installation scenarios

echo.
echo [INFO] Starting Google Cloud Storage Buckets Deployment
echo ==================================================

REM Check prerequisites with improved Terraform detection
echo [INFO] Checking prerequisites...

REM Function to find Terraform executable
set TERRAFORM_CMD=
set TERRAFORM_FOUND=0

REM Check if terraform is in PATH
terraform --version >nul 2>&1
if not errorlevel 1 (
    set TERRAFORM_CMD=terraform
    set TERRAFORM_FOUND=1
    echo [SUCCESS] Found Terraform in PATH
    goto :terraform_found
)

REM Check common installation locations
set LOCATIONS=C:\terraform\terraform.exe %USERPROFILE%\terraform\terraform.exe C:\Tools\Terraform\terraform.exe "C:\Program Files\Terraform\terraform.exe" C:\HashiCorp\Terraform\terraform.exe

for %%L in (%LOCATIONS%) do (
    if exist "%%L" (
        set TERRAFORM_CMD=%%L
        set TERRAFORM_FOUND=1
        echo [SUCCESS] Found Terraform at: %%L
        goto :terraform_found
    )
)

REM Check if Chocolatey installed Terraform
where terraform >nul 2>&1
if not errorlevel 1 (
    set TERRAFORM_CMD=terraform
    set TERRAFORM_FOUND=1
    echo [SUCCESS] Found Terraform via Chocolatey
    goto :terraform_found
)

:terraform_not_found
echo [ERROR] Terraform is not installed or not found in common locations.
echo.
echo [INFO] Please install Terraform using one of these methods:
echo.
echo Method 1 - Manual Installation:
echo   1. Download from: https://releases.hashicorp.com/terraform/1.13.1/
echo   2. Extract terraform.exe to C:\terraform\
echo   3. Add C:\terraform to your PATH environment variable
echo   4. Restart command prompt and try again
echo.
echo Method 2 - Using Chocolatey:
echo   choco install terraform --version=1.13.1
echo.
echo Method 3 - Using our PowerShell script:
echo   powershell -ExecutionPolicy Bypass -File scripts\install_terraform.ps1
echo.
pause
exit /b 1

:terraform_found
REM Test Terraform
echo [INFO] Testing Terraform installation...
"%TERRAFORM_CMD%" --version
if errorlevel 1 (
    echo [ERROR] Terraform test failed
    exit /b 1
)

REM Check if gcloud is installed
gcloud --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Google Cloud SDK is not installed. Please install gcloud first.
    echo [INFO] Download from: https://cloud.google.com/sdk/docs/install
    pause
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
echo   - Terraform: %TERRAFORM_CMD%

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
"%TERRAFORM_CMD%" init
if errorlevel 1 (
    echo [ERROR] Failed to initialize Terraform
    echo [INFO] This might be due to missing credentials or network issues
    echo [INFO] Make sure you're authenticated with: gcloud auth login
    pause
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

"%TERRAFORM_CMD%" plan -out=tfplan
if errorlevel 1 (
    echo [ERROR] Failed to create Terraform plan
    echo [INFO] Check your GCP credentials and project permissions
    pause
    exit /b 1
)
echo [SUCCESS] Terraform plan completed successfully!

REM Apply Terraform changes
echo [INFO] Applying Terraform changes...
echo.
set /p response="[WARNING] This will create/modify Google Cloud resources. Continue? (y/N): "
if /i "!response!"=="y" (
    "%TERRAFORM_CMD%" apply tfplan
    if errorlevel 1 (
        echo [ERROR] Failed to apply Terraform changes
        pause
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
