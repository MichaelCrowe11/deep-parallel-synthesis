@echo off
echo ===============================================
echo    DEEP PARALLEL SYNTHESIS - QUICK START
echo ===============================================
echo.

REM Step 1: Verify Docker
echo [1/5] Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker not found. Please install Docker Desktop first.
    echo Download from: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)
echo SUCCESS: Docker is ready!
echo.

REM Step 2: Build the image
echo [2/5] Building DPS Docker image (this may take 5-10 minutes)...
docker build -t dps:latest . 
if %errorlevel% neq 0 (
    echo ERROR: Build failed
    pause
    exit /b 1
)
echo SUCCESS: Image built!
echo.

REM Step 3: Start services
echo [3/5] Starting all services...
docker-compose -f docker-compose.dev.yml up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start services
    pause
    exit /b 1
)
echo SUCCESS: Services started!
echo.

REM Step 4: Wait for initialization
echo [4/5] Waiting for services to initialize (30 seconds)...
timeout /t 30 /nobreak >nul
echo.

REM Step 5: Test the deployment
echo [5/5] Testing deployment...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo SUCCESS: API is responding!
) else (
    echo WARNING: API not responding yet (may still be starting)
)
echo.

echo ===============================================
echo    DEPLOYMENT COMPLETE!
echo ===============================================
echo.
echo Access your services:
echo   API:        http://localhost:8000
echo   Grafana:    http://localhost:3000 (admin/admin)
echo   Prometheus: http://localhost:9090
echo.
echo View logs:
echo   docker-compose -f docker-compose.dev.yml logs -f
echo.
echo Stop services:
echo   docker-compose -f docker-compose.dev.yml down
echo.
pause