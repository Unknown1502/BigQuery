@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Virtual environment activated successfully!
echo Python version:
python --version
echo.
echo Key packages installed:
echo - FastAPI 0.104.1
echo - TensorFlow 2.15.0
echo - PyTorch 2.1.1
echo - Google Cloud SDK packages
echo.
echo To deactivate, type: deactivate
echo To run the API server: uvicorn src.services.api-gateway.main:app --reload
echo To run tests: pytest
