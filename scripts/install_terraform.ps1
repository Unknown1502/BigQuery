# PowerShell script to install Terraform on Windows
# Run this script as Administrator for best results

Write-Host "[INFO] Installing Terraform for Windows..." -ForegroundColor Blue

# Configuration
$TerraformVersion = "1.13.1"
$TerraformUrl = "https://releases.hashicorp.com/terraform/$TerraformVersion/terraform_${TerraformVersion}_windows_amd64.zip"
$TempDir = "$env:TEMP\terraform_install_$(Get-Random)"
$InstallDir = "$env:USERPROFILE\terraform"

try {
    # Create temporary directory
    Write-Host "[INFO] Creating temporary directory..." -ForegroundColor Blue
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
    Set-Location $TempDir

    # Download Terraform
    Write-Host "[INFO] Downloading Terraform $TerraformVersion..." -ForegroundColor Blue
    Invoke-WebRequest -Uri $TerraformUrl -OutFile "terraform.zip" -UseBasicParsing
    
    if (-not (Test-Path "terraform.zip")) {
        throw "Failed to download Terraform"
    }
    Write-Host "[SUCCESS] Download completed" -ForegroundColor Green

    # Extract Terraform
    Write-Host "[INFO] Extracting Terraform..." -ForegroundColor Blue
    Expand-Archive -Path "terraform.zip" -DestinationPath "." -Force
    
    if (-not (Test-Path "terraform.exe")) {
        throw "Failed to extract terraform.exe"
    }
    Write-Host "[SUCCESS] Extraction completed" -ForegroundColor Green

    # Create installation directory
    Write-Host "[INFO] Installing Terraform to $InstallDir..." -ForegroundColor Blue
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }

    # Copy terraform.exe
    Copy-Item "terraform.exe" "$InstallDir\terraform.exe" -Force
    Write-Host "[SUCCESS] Terraform installed to $InstallDir" -ForegroundColor Green

    # Add to PATH
    Write-Host "[INFO] Adding Terraform to PATH..." -ForegroundColor Blue
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -notlike "*$InstallDir*") {
        $newPath = $currentPath + ";" + $InstallDir
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
        Write-Host "[SUCCESS] Added to user PATH" -ForegroundColor Green
    } else {
        Write-Host "[INFO] Already in PATH" -ForegroundColor Yellow
    }

    # Test installation
    Write-Host "[INFO] Testing Terraform installation..." -ForegroundColor Blue
    $env:PATH = $env:PATH + ";" + $InstallDir
    & "$InstallDir\terraform.exe" --version
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Terraform is working correctly!" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Terraform test failed" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "[SUCCESS] Terraform installation completed!" -ForegroundColor Green
    Write-Host "[INFO] Terraform location: $InstallDir\terraform.exe" -ForegroundColor Blue
    Write-Host "[INFO] You may need to restart your command prompt for PATH changes to take effect" -ForegroundColor Yellow

} catch {
    Write-Host "[ERROR] Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    # Clean up
    if (Test-Path $TempDir) {
        Set-Location $env:USERPROFILE
        Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
