@echo off
setlocal enabledelayedexpansion

echo [INFO] Installing Terraform for Windows...
echo.

REM Set Terraform version
set TERRAFORM_VERSION=1.13.1

REM Create a temporary directory
set TEMP_DIR=%TEMP%\terraform_install_%RANDOM%
mkdir "%TEMP_DIR%"
cd /d "%TEMP_DIR%"

echo [INFO] Downloading Terraform %TERRAFORM_VERSION%...

REM Download Terraform using PowerShell
powershell -Command "try { Invoke-WebRequest -Uri 'https://releases.hashicorp.com/terraform/%TERRAFORM_VERSION%/terraform_%TERRAFORM_VERSION%_windows_amd64.zip' -OutFile 'terraform.zip' -UseBasicParsing; Write-Host '[SUCCESS] Download completed' } catch { Write-Host '[ERROR] Download failed:' $_.Exception.Message; exit 1 }"

if not exist "terraform.zip" (
    echo [ERROR] Failed to download Terraform
    cd /d "%~dp0"
    rmdir /s /q "%TEMP_DIR%"
    pause
    exit /b 1
)

echo [INFO] Extracting Terraform...

REM Extract the zip file
powershell -Command "try { Expand-Archive -Path 'terraform.zip' -DestinationPath '.' -Force; Write-Host '[SUCCESS] Extraction completed' } catch { Write-Host '[ERROR] Extraction failed:' $_.Exception.Message; exit 1 }"

if not exist "terraform.exe" (
    echo [ERROR] Failed to extract terraform.exe
    cd /d "%~dp0"
    rmdir /s /q "%TEMP_DIR%"
    pause
    exit /b 1
)

echo [INFO] Installing Terraform to system...

REM Create Terraform directory
set TERRAFORM_DIR=C:\Tools\Terraform
if not exist "%TERRAFORM_DIR%" mkdir "%TERRAFORM_DIR%"

REM Copy terraform.exe
copy terraform.exe "%TERRAFORM_DIR%\terraform.exe" >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy terraform.exe to %TERRAFORM_DIR%
    echo [INFO] Trying alternative location...
    
    REM Try user directory instead
    set TERRAFORM_DIR=%USERPROFILE%\terraform
    if not exist "%TERRAFORM_DIR%" mkdir "%TERRAFORM_DIR%"
    copy terraform.exe "%TERRAFORM_DIR%\terraform.exe" >nul
    if errorlevel 1 (
        echo [ERROR] Failed to install Terraform
        cd /d "%~dp0"
        rmdir /s /q "%TEMP_DIR%"
        pause
        exit /b 1
    )
)

echo [SUCCESS] Terraform installed to: %TERRAFORM_DIR%

REM Add to PATH for current session
set PATH=%PATH%;%TERRAFORM_DIR%

echo [INFO] Adding Terraform to system PATH...

REM Add to system PATH permanently
powershell -Command "try { $env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'User'); if ($env:PATH -notlike '*%TERRAFORM_DIR%*') { [Environment]::SetEnvironmentVariable('PATH', $env:PATH + ';%TERRAFORM_DIR%', 'User'); Write-Host '[SUCCESS] Added to user PATH' } else { Write-Host '[INFO] Already in PATH' } } catch { Write-Host '[WARNING] Could not add to system PATH. You may need to add manually.' }"

REM Clean up
cd /d "%~dp0"
rmdir /s /q "%TEMP_DIR%"

echo.
echo [SUCCESS] Terraform installation completed!
echo [INFO] Terraform location: %TERRAFORM_DIR%\terraform.exe
echo.
echo [INFO] Testing Terraform installation...

REM Test the installation
"%TERRAFORM_DIR%\terraform.exe" --version
if errorlevel 1 (
    echo [WARNING] Terraform test failed. You may need to restart your command prompt.
) else (
    echo [SUCCESS] Terraform is working correctly!
)

echo.
echo [INFO] If terraform command is not recognized, please:
echo 1. Restart your command prompt/PowerShell
echo 2. Or manually add %TERRAFORM_DIR% to your PATH environment variable
echo.

pause
