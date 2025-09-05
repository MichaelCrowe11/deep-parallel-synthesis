@echo off
echo === Deep Parallel Synthesis - Local Deployment ===
echo.

REM Check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo.
    echo Please install Docker Desktop for Windows:
    echo https://www.docker.com/products/docker-desktop/
    echo.
    echo After installation:
    echo 1. Start Docker Desktop
    echo 2. Wait for it to fully start
    echo 3. Run this script again
    echo.
    pause
    exit /b 1
)

echo [1/6] Docker found: 
docker --version
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Desktop is not running
    echo Please start Docker Desktop and wait for it to fully start
    pause
    exit /b 1
)

echo [2/6] Building Docker image...
docker build -t dps:latest . 
if %errorlevel% neq 0 (
    echo ERROR: Docker build failed
    pause
    exit /b 1
)
echo SUCCESS: Image built
echo.

echo [3/6] Starting services with Docker Compose...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Docker Compose failed to start services
    pause
    exit /b 1
)
echo SUCCESS: Services started
echo.

echo [4/6] Waiting for services to initialize (30 seconds)...
timeout /t 30 /nobreak >nul
echo.

echo [5/6] Checking service health...
curl -s http://localhost:8000/health
echo.

echo [6/6] Services Status:
docker-compose ps
echo.

echo === Deployment Complete ===
echo.
echo Access points:
echo - API:        http://localhost:8000
echo - Grafana:    http://localhost:3000 (admin/admin)
echo - Prometheus: http://localhost:9090
echo.
echo Test inference with:
echo curl -X POST http://localhost:8000/v1/generate -H "Content-Type: application/json" -d "{\"prompt\": \"Hello\", \"max_tokens\": 50}"
echo.
echo To stop services:
echo docker-compose down
echo.
pause