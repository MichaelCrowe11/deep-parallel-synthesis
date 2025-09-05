# Deep Parallel Synthesis - Local Deployment Script

Write-Host "=== Deep Parallel Synthesis - Local Deployment ===" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
try {
    $dockerVersion = docker --version
    Write-Host "[✓] Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "[✗] Docker is not installed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Docker Desktop for Windows:" -ForegroundColor Yellow
    Write-Host "https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After installation:" -ForegroundColor Yellow
    Write-Host "1. Start Docker Desktop" -ForegroundColor Yellow
    Write-Host "2. Wait for it to fully start" -ForegroundColor Yellow
    Write-Host "3. Run this script again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "[✓] Docker Desktop is running" -ForegroundColor Green
} catch {
    Write-Host "[✗] Docker Desktop is not running" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and wait for it to fully start" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[1/6] Building Docker image..." -ForegroundColor Yellow
try {
    docker build -t dps:latest .
    Write-Host "[✓] Docker image built successfully" -ForegroundColor Green
} catch {
    Write-Host "[✗] Docker build failed" -ForegroundColor Red
    Write-Host $_.Exception.Message
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[2/6] Starting services with Docker Compose..." -ForegroundColor Yellow
try {
    docker-compose -f docker-compose.dev.yml up -d
    Write-Host "[✓] Services started successfully" -ForegroundColor Green
} catch {
    Write-Host "[✗] Failed to start services" -ForegroundColor Red
    Write-Host $_.Exception.Message
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[3/6] Waiting for services to initialize (30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

Write-Host ""
Write-Host "[4/6] Checking service health..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    Write-Host "[✓] Service is healthy: $($health | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "[⚠] Service health check failed (may still be starting)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[5/6] Services Status:" -ForegroundColor Yellow
docker-compose -f docker-compose.dev.yml ps

Write-Host ""
Write-Host "[6/6] Testing inference endpoint..." -ForegroundColor Yellow
try {
    $testPayload = @{
        prompt = "What is 2+2?"
        max_tokens = 10
        temperature = 0.7
    } | ConvertTo-Json

    $response = Invoke-RestMethod -Uri "http://localhost:8000/v1/generate" -Method Post -Body $testPayload -ContentType "application/json"
    Write-Host "[✓] Inference test successful" -ForegroundColor Green
} catch {
    Write-Host "[⚠] Inference test failed (service may still be loading model)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Deployment Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Access points:" -ForegroundColor Cyan
Write-Host "  • API:        http://localhost:8000" -ForegroundColor White
Write-Host "  • Grafana:    http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "  • Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  • View logs:    docker-compose -f docker-compose.dev.yml logs -f" -ForegroundColor White
Write-Host "  • Stop services: docker-compose -f docker-compose.dev.yml down" -ForegroundColor White
Write-Host "  • Restart:      docker-compose -f docker-compose.dev.yml restart" -ForegroundColor White
Write-Host ""
Write-Host "Test inference with:" -ForegroundColor Cyan
Write-Host 'Invoke-RestMethod -Uri "http://localhost:8000/v1/generate" -Method Post -Body (ConvertTo-Json @{prompt="Hello"; max_tokens=50}) -ContentType "application/json"' -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to exit"