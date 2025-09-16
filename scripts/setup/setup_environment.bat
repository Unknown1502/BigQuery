@echo off
setlocal enabledelayedexpansion

REM Dynamic Pricing Intelligence - Windows Environment Setup Script
REM This script sets up the complete development and production environment on Windows

echo.
echo ==========================================
echo Dynamic Pricing Intelligence Setup
echo ==========================================
echo.

REM Configuration
set PROJECT_ID=
set REGION=us-central1
set ZONE=us-central1-a
set ENVIRONMENT=development
set TERRAFORM_VERSION=1.13.1

REM Colors for output (using echo with different prefixes)
set INFO_PREFIX=[INFO]
set SUCCESS_PREFIX=[SUCCESS]
set WARNING_PREFIX=[WARNING]
set ERROR_PREFIX=[ERROR]

REM Function to check if command exists
:check_command
where %1 >nul 2>&1
exit /b %errorlevel%

REM Function to print status messages
:print_info
echo %INFO_PREFIX% %~1
exit /b 0

:print_success
echo %SUCCESS_PREFIX% %~1
exit /b 0

:print_warning
echo %WARNING_PREFIX% %~1
exit /b 0

:print_error
echo %ERROR_PREFIX% %~1
exit /b 0

REM Function to check prerequisites and install missing tools
:check_prerequisites
call :print_info "Checking prerequisites..."

set missing_tools=
set need_terraform=0
set need_gcloud=0
set need_python=0
set need_git=0

REM Check gcloud
call :check_command gcloud
if errorlevel 1 (
    set missing_tools=!missing_tools! gcloud
    set need_gcloud=1
)

REM Check terraform
call :check_command terraform
if errorlevel 1 (
    set missing_tools=!missing_tools! terraform
    set need_terraform=1
)

REM Check python
call :check_command python
if errorlevel 1 (
    set missing_tools=!missing_tools! python
    set need_python=1
)

REM Check git
call :check_command git
if errorlevel 1 (
    set missing_tools=!missing_tools! git
    set need_git=1
)

if not "!missing_tools!"=="" (
    call :print_warning "Missing tools detected:!missing_tools!"
    call :print_info "Attempting to install missing tools..."
    call :install_missing_tools
) else (
    call :print_success "All prerequisites are installed"
)

exit /b 0

REM Function to install missing tools
:install_missing_tools

REM Check if Chocolatey is installed
call :check_command choco
if errorlevel 1 (
    call :print_info "Installing Chocolatey package manager..."
    powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    
    REM Refresh environment variables
    call refreshenv
    
    REM Check if chocolatey installation was successful
    call :check_command choco
    if errorlevel 1 (
        call :print_error "Failed to install Chocolatey. Please install it manually from https://chocolatey.org/install"
        pause
        exit /b 1
    )
    call :print_success "Chocolatey installed successfully"
)

REM Install Terraform
if %need_terraform%==1 (
    call :print_info "Installing Terraform..."
    choco install terraform --version=%TERRAFORM_VERSION% -y
    if errorlevel 1 (
        call :print_warning "Chocolatey installation failed. Trying manual installation..."
        call :install_terraform_manual
    ) else (
        call :print_success "Terraform installed successfully via Chocolatey"
    )
)

REM Install Google Cloud SDK
if %need_gcloud%==1 (
    call :print_info "Installing Google Cloud SDK..."
    choco install gcloudsdk -y
    if errorlevel 1 (
        call :print_error "Failed to install Google Cloud SDK via Chocolatey"
        call :print_info "Please install manually from https://cloud.google.com/sdk/docs/install"
        pause
        exit /b 1
    )
    call :print_success "Google Cloud SDK installed successfully"
)

REM Install Python
if %need_python%==1 (
    call :print_info "Installing Python..."
    choco install python -y
    if errorlevel 1 (
        call :print_error "Failed to install Python via Chocolatey"
        call :print_info "Please install manually from https://python.org"
        pause
        exit /b 1
    )
    call :print_success "Python installed successfully"
)

REM Install Git
if %need_git%==1 (
    call :print_info "Installing Git..."
    choco install git -y
    if errorlevel 1 (
        call :print_error "Failed to install Git via Chocolatey"
        call :print_info "Please install manually from https://git-scm.com"
        pause
        exit /b 1
    )
    call :print_success "Git installed successfully"
)

REM Refresh environment variables after installations
call refreshenv

call :print_success "All missing tools have been installed"
exit /b 0

REM Function to manually install Terraform
:install_terraform_manual
call :print_info "Downloading Terraform manually..."

REM Create temp directory
if not exist "%TEMP%\terraform_install" mkdir "%TEMP%\terraform_install"
cd /d "%TEMP%\terraform_install"

REM Download Terraform
powershell -Command "Invoke-WebRequest -Uri 'https://releases.hashicorp.com/terraform/%TERRAFORM_VERSION%/terraform_%TERRAFORM_VERSION%_windows_amd64.zip' -OutFile 'terraform.zip'"

if not exist "terraform.zip" (
    call :print_error "Failed to download Terraform"
    exit /b 1
)

REM Extract Terraform
powershell -Command "Expand-Archive -Path 'terraform.zip' -DestinationPath '.' -Force"

REM Create Terraform directory in Program Files
if not exist "C:\Program Files\Terraform" mkdir "C:\Program Files\Terraform"

REM Copy terraform.exe
copy terraform.exe "C:\Program Files\Terraform\terraform.exe"

REM Add to PATH
setx PATH "%PATH%;C:\Program Files\Terraform" /M

call :print_success "Terraform installed manually to C:\Program Files\Terraform"

REM Clean up
cd /d "%~dp0"
rmdir /s /q "%TEMP%\terraform_install"

exit /b 0

REM Function to get user input
:get_user_input
call :print_info "Gathering configuration information..."

if "%PROJECT_ID%"=="" (
    set /p PROJECT_ID="Enter your GCP Project ID: "
)

if "%PROJECT_ID%"=="" (
    call :print_error "Project ID is required"
    pause
    exit /b 1
)

set /p input_region="Enter region (default: us-central1): "
if not "%input_region%"=="" set REGION=%input_region%

set /p input_env="Enter environment (development/staging/production) [default: development]: "
if not "%input_env%"=="" set ENVIRONMENT=%input_env%

call :print_success "Configuration gathered successfully"
call :print_info "Project ID: %PROJECT_ID%"
call :print_info "Region: %REGION%"
call :print_info "Environment: %ENVIRONMENT%"

exit /b 0

REM Function to setup GCP authentication
:setup_gcp_auth
call :print_info "Setting up GCP authentication..."

REM Check if already authenticated
gcloud auth list --filter=status:ACTIVE --format="value(account)" | findstr "@" >nul
if errorlevel 1 (
    call :print_info "Please authenticate with GCP..."
    gcloud auth login
) else (
    call :print_success "Already authenticated with GCP"
)

REM Set project
gcloud config set project "%PROJECT_ID%"
gcloud config set compute/region "%REGION%"
gcloud config set compute/zone "%ZONE%"

call :print_success "GCP authentication configured"
exit /b 0

REM Function to enable required APIs
:enable_apis
call :print_info "Enabling required GCP APIs..."

set apis=bigquery.googleapis.com storage-api.googleapis.com pubsub.googleapis.com cloudfunctions.googleapis.com aiplatform.googleapis.com container.googleapis.com compute.googleapis.com cloudresourcemanager.googleapis.com iam.googleapis.com monitoring.googleapis.com logging.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com servicenetworking.googleapis.com dns.googleapis.com cloudscheduler.googleapis.com

for %%a in (%apis%) do (
    call :print_info "Enabling %%a..."
    gcloud services enable %%a --project="%PROJECT_ID%"
)

call :print_success "All required APIs enabled"
exit /b 0

REM Function to setup Python environment
:setup_python_env
call :print_info "Setting up Python environment..."

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip and install build tools
python -m pip install --upgrade pip setuptools wheel

REM Install requirements
if exist "requirements.txt" (
    pip install -r requirements.txt
)

if exist "requirements-dev.txt" (
    if "%ENVIRONMENT%"=="development" (
        pip install -r requirements-dev.txt
    )
)

call :print_success "Python environment setup complete"
exit /b 0

REM Function to initialize Terraform
:setup_terraform
call :print_info "Setting up Terraform..."

cd /d "infrastructure\terraform"

REM Create terraform.tfvars if it doesn't exist
if not exist "terraform.tfvars" (
    (
        echo project_id = "%PROJECT_ID%"
        echo region = "%REGION%"
        echo environment = "%ENVIRONMENT%"
        echo owner = "%USERNAME%"
        echo.
        echo # Network configuration
        echo gke_subnet_cidr = "10.0.0.0/24"
        echo gke_pods_cidr = "10.1.0.0/16"
        echo gke_services_cidr = "10.2.0.0/16"
        echo services_subnet_cidr = "10.3.0.0/24"
        echo.
        echo # Storage configuration
        echo storage_location = "US"
        echo storage_force_destroy = true
        echo.
        echo # Security configuration
        echo ssh_source_ranges = ["0.0.0.0/0"]
        echo blocked_ip_ranges = []
        echo.
        echo # SSL configuration
        echo ssl_domains = ["api.%PROJECT_ID%.com"]
        echo domain_name = "%PROJECT_ID%.com"
        echo.
        echo # Monitoring configuration
        echo alert_emails = ["admin@%PROJECT_ID%.com"]
        echo alert_notification_channel = ""
        echo.
        echo # Performance configuration
        echo backup_retention_days = 30
        echo model_performance_threshold = 0.85
    ) > terraform.tfvars
)

REM Initialize Terraform
terraform init
if errorlevel 1 (
    call :print_error "Failed to initialize Terraform"
    cd /d "%~dp0..\.."
    exit /b 1
)

REM Validate configuration
terraform validate
if errorlevel 1 (
    call :print_error "Terraform configuration validation failed"
    cd /d "%~dp0..\.."
    exit /b 1
)

call :print_success "Terraform initialized successfully"

cd /d "%~dp0..\.."
exit /b 0

REM Function to display next steps
:display_next_steps
call :print_success "Environment setup completed successfully!"
echo.
call :print_info "Next steps:"
echo 1. Review and edit the .env file with your specific configuration
echo 2. Review and customize infrastructure\terraform\terraform.tfvars
echo 3. Deploy infrastructure: cd infrastructure\terraform ^&^& terraform plan ^&^& terraform apply
echo 4. Deploy storage buckets: scripts\deploy_storage_buckets.bat
echo 5. Build and deploy services as needed
echo.
call :print_info "For development:"
echo - Activate Python environment: venv\Scripts\activate.bat
echo - Run tests: python -m pytest tests\
echo.
call :print_info "Troubleshooting:"
echo - If commands are not recognized, restart your command prompt
echo - Ensure all tools are in your PATH environment variable
echo.
exit /b 0

REM Main execution
:main
call :check_prerequisites
if errorlevel 1 exit /b 1

call :get_user_input
if errorlevel 1 exit /b 1

call :setup_gcp_auth
if errorlevel 1 exit /b 1

call :enable_apis
if errorlevel 1 exit /b 1

call :setup_python_env
if errorlevel 1 exit /b 1

call :setup_terraform
if errorlevel 1 exit /b 1

call :display_next_steps

exit /b 0

REM Parse command line arguments
:parse_args
if "%~1"=="--project-id" (
    set PROJECT_ID=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--region" (
    set REGION=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--environment" (
    set ENVIRONMENT=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--help" (
    echo Usage: %0 [OPTIONS]
    echo Options:
    echo   --project-id PROJECT_ID    GCP Project ID
    echo   --region REGION           GCP Region (default: us-central1)
    echo   --environment ENV         Environment (development/staging/production)
    echo   --help                    Show this help message
    exit /b 0
)
if not "%~1"=="" (
    call :print_error "Unknown option: %~1"
    exit /b 1
)
exit /b 0

REM Entry point
call :parse_args %*
call :main

pause
