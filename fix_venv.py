"""
Virtual Environment Diagnostic and Repair Script
This script will diagnose and fix common virtual environment issues.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a command and return the result."""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode == 0, "", ""
    except Exception as e:
        return False, "", str(e)

def check_python():
    """Check Python installation."""
    print("1. Checking Python installation...")
    success, stdout, stderr = run_command("python --version")
    if success:
        print(f"   Python found: {stdout.strip()}")
    else:
        print("   ERROR: Python not found in PATH")
        return False
    
    # Check if Python 3.11 is available
    success, stdout, stderr = run_command("py -3.11 --version")
    if success:
        print(f"   Python 3.11 found: {stdout.strip()}")
    else:
        print("   WARNING: Python 3.11 not found via py launcher")
    
    return True

def check_venv():
    """Check if virtual environment exists and is valid."""
    print("\n2. Checking virtual environment...")
    venv_path = Path("venv")
    
    if not venv_path.exists():
        print("   ERROR: venv directory does not exist")
        return False
    
    # Check for essential venv files
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
        activate = venv_path / "Scripts" / "activate.bat"
    else:
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
        activate = venv_path / "bin" / "activate"
    
    missing_files = []
    if not python_exe.exists():
        missing_files.append(str(python_exe))
    if not pip_exe.exists():
        missing_files.append(str(pip_exe))
    if not activate.exists():
        missing_files.append(str(activate))
    
    if missing_files:
        print(f"   ERROR: Missing essential files: {', '.join(missing_files)}")
        return False
    
    print("   Virtual environment structure looks good")
    
    # Test venv Python
    success, stdout, stderr = run_command(f'"{python_exe}" --version')
    if success:
        print(f"   Venv Python version: {stdout.strip()}")
    else:
        print(f"   ERROR: Cannot run venv Python: {stderr}")
        return False
    
    return True

def check_pip():
    """Check if pip is working in the virtual environment."""
    print("\n3. Checking pip in virtual environment...")
    
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip.exe"
    else:
        pip_cmd = "venv/bin/pip"
    
    success, stdout, stderr = run_command(f'"{pip_cmd}" --version')
    if success:
        print(f"   Pip version: {stdout.strip()}")
    else:
        print(f"   ERROR: Pip not working: {stderr}")
        return False
    
    # Check installed packages
    success, stdout, stderr = run_command(f'"{pip_cmd}" list')
    if success:
        lines = stdout.strip().split('\n')
        print(f"   Installed packages: {len(lines) - 2 if len(lines) > 2 else 0}")
    else:
        print(f"   ERROR: Cannot list packages: {stderr}")
        return False
    
    return True

def fix_venv():
    """Attempt to fix the virtual environment."""
    print("\n4. Attempting to fix virtual environment...")
    
    # First, try to upgrade pip
    if sys.platform == "win32":
        python_exe = "venv\\Scripts\\python.exe"
        pip_exe = "venv\\Scripts\\pip.exe"
    else:
        python_exe = "venv/bin/python"
        pip_exe = "venv/bin/pip"
    
    print("   Upgrading pip...")
    success, stdout, stderr = run_command(f'"{python_exe}" -m pip install --upgrade pip')
    if success:
        print("   Pip upgraded successfully")
    else:
        print(f"   WARNING: Could not upgrade pip: {stderr}")
    
    # Install setuptools and wheel
    print("   Installing setuptools and wheel...")
    success, stdout, stderr = run_command(f'"{pip_exe}" install --upgrade setuptools wheel')
    if success:
        print("   Setuptools and wheel installed successfully")
    else:
        print(f"   WARNING: Could not install setuptools/wheel: {stderr}")
    
    return True

def recreate_venv():
    """Recreate the virtual environment from scratch."""
    print("\n5. Recreating virtual environment...")
    
    # Backup requirements if they exist
    requirements_files = ["requirements.txt", "requirements-dev.txt"]
    
    # Remove old venv
    venv_path = Path("venv")
    if venv_path.exists():
        print("   Removing old virtual environment...")
        try:
            shutil.rmtree(venv_path)
            print("   Old venv removed")
        except Exception as e:
            print(f"   ERROR: Could not remove old venv: {e}")
            return False
    
    # Create new venv
    print("   Creating new virtual environment...")
    success, stdout, stderr = run_command("python -m venv venv")
    if not success:
        # Try with py launcher
        success, stdout, stderr = run_command("py -3.11 -m venv venv")
        if not success:
            print(f"   ERROR: Could not create venv: {stderr}")
            return False
    
    print("   Virtual environment created successfully")
    
    # Upgrade pip in new venv
    if sys.platform == "win32":
        python_exe = "venv\\Scripts\\python.exe"
        pip_exe = "venv\\Scripts\\pip.exe"
    else:
        python_exe = "venv/bin/python"
        pip_exe = "venv/bin/pip"
    
    print("   Upgrading pip in new environment...")
    run_command(f'"{python_exe}" -m pip install --upgrade pip setuptools wheel')
    
    return True

def install_requirements():
    """Install requirements in the virtual environment."""
    print("\n6. Installing requirements...")
    
    if sys.platform == "win32":
        pip_exe = "venv\\Scripts\\pip.exe"
    else:
        pip_exe = "venv/bin/pip"
    
    requirements_files = ["requirements.txt", "requirements-dev.txt"]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            print(f"   Installing {req_file}...")
            success, stdout, stderr = run_command(f'"{pip_exe}" install -r {req_file}', capture_output=False)
            if success:
                print(f"   {req_file} installed successfully")
            else:
                print(f"   WARNING: Some packages from {req_file} may have failed to install")
    
    return True

def create_activation_scripts():
    """Create helper scripts for activating the virtual environment."""
    print("\n7. Creating activation helper scripts...")
    
    # Windows batch file
    activate_bat = """@echo off
echo Activating virtual environment...
call venv\\Scripts\\activate.bat
echo Virtual environment activated!
echo.
echo To deactivate, type: deactivate
"""
    
    # PowerShell script
    activate_ps1 = """Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\\venv\\Scripts\\Activate.ps1
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "To deactivate, type: deactivate" -ForegroundColor Yellow
"""
    
    # Create the scripts
    with open("activate_venv.bat", "w") as f:
        f.write(activate_bat)
    print("   Created activate_venv.bat")
    
    with open("activate_venv.ps1", "w") as f:
        f.write(activate_ps1)
    print("   Created activate_venv.ps1")
    
    return True

def main():
    """Main diagnostic and repair function."""
    print("="*60)
    print("Virtual Environment Diagnostic and Repair Tool")
    print("="*60)
    
    # Check Python
    if not check_python():
        print("\nPython installation issues detected. Please install Python 3.11 or later.")
        return
    
    # Check venv
    venv_ok = check_venv()
    pip_ok = False
    
    if venv_ok:
        pip_ok = check_pip()
    
    if not venv_ok or not pip_ok:
        print("\nVirtual environment issues detected.")
        response = input("Do you want to recreate the virtual environment? (y/n): ")
        if response.lower() == 'y':
            if recreate_venv():
                check_venv()
                check_pip()
                install_requirements()
        else:
            print("Attempting to fix existing environment...")
            fix_venv()
            check_pip()
    else:
        print("\nVirtual environment appears to be working correctly!")
        response = input("Do you want to reinstall requirements? (y/n): ")
        if response.lower() == 'y':
            install_requirements()
    
    # Create activation scripts
    create_activation_scripts()
    
    print("\n" + "="*60)
    print("Diagnostic complete!")
    print("="*60)
    print("\nTo activate your virtual environment:")
    print("  - Command Prompt: activate_venv.bat")
    print("  - PowerShell: .\\activate_venv.ps1")
    print("  - Or directly: venv\\Scripts\\activate")
    print("\nTo test if everything is working:")
    print("  1. Activate the virtual environment")
    print("  2. Run: python -c \"import fastapi; print('FastAPI imported successfully!')\"")

if __name__ == "__main__":
    main()
