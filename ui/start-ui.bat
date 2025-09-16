@echo off
echo ========================================
echo Dynamic Pricing Intelligence UI System
echo BigQuery AI Competition 2025
echo ========================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Node.js version:
node --version
echo.

REM Check if we're in the correct directory
if not exist "package.json" (
    echo ERROR: package.json not found
    echo Please run this script from the ui directory
    pause
    exit /b 1
)

REM Install backend dependencies if needed
if not exist "node_modules" (
    echo Installing backend dependencies...
    npm install
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install backend dependencies
        pause
        exit /b 1
    )
    echo.
)

REM Install frontend dependencies if needed
if not exist "client\node_modules" (
    echo Installing frontend dependencies...
    cd client
    npm install
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install frontend dependencies
        pause
        exit /b 1
    )
    cd ..
    echo.
)

REM Check for environment configuration
if not exist ".env" (
    echo WARNING: .env file not found
    echo Creating sample .env file...
    echo # Google Cloud Configuration > .env
    echo GOOGLE_CLOUD_PROJECT_ID=your-project-id >> .env
    echo GOOGLE_APPLICATION_CREDENTIALS=../credentials/geosptial-471213-105ec6eaf6bd.json >> .env
    echo. >> .env
    echo # BigQuery Configuration >> .env
    echo BIGQUERY_DATASET_ID=ride_intelligence >> .env
    echo BIGQUERY_LOCATION=US >> .env
    echo. >> .env
    echo # Server Configuration >> .env
    echo PORT=5000 >> .env
    echo NODE_ENV=development >> .env
    echo. >> .env
    echo # CORS Configuration >> .env
    echo CORS_ORIGIN=http://localhost:3000 >> .env
    echo.
    echo Please update the .env file with your actual configuration
    echo.
)

REM Check if port 3000 is already in use
netstat -an | find "3000" >nul 2>&1
if %errorlevel% equ 0 (
    echo WARNING: Port 3000 is already in use
    echo Please stop any existing applications on port 3000 or the frontend will use an alternative port
    echo.
)

REM Check if port 5000 is already in use
netstat -an | find "5000" >nul 2>&1
if %errorlevel% equ 0 (
    echo WARNING: Port 5000 is already in use
    echo Please stop any existing applications on port 5000 or the backend may fail to start
    echo.
)

echo Starting the Dynamic Pricing Intelligence UI System...
echo.
echo Backend will start on: http://localhost:5000
echo Frontend will start on: http://localhost:3000 (or next available port)
echo.
echo Press Ctrl+C to stop the servers
echo.

REM Start both backend and frontend concurrently
start "Backend Server" cmd /k "npm run dev"
timeout /t 3 /nobreak >nul
start "Frontend Server" cmd /k "cd client && npm start"

echo.
echo Both servers are starting...
echo Check the opened terminal windows for detailed logs
echo.
echo If port 3000 was occupied, React will automatically use the next available port
echo API Health Check: http://localhost:5000/api/health
echo Main Application: http://localhost:3000 (or check terminal for actual port)
echo.
echo Note: If you see port conflicts, please close other applications using those ports
echo and restart this script for optimal performance.
echo.
pause
