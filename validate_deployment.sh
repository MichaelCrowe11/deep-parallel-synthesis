#!/bin/bash

echo "=== DPS Deployment Validation ==="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Docker
echo "1. Checking Docker installation..."
if command_exists docker; then
    echo "   ✓ Docker version: $(docker --version)"
else
    echo "   ✗ Docker not found. Please install Docker Desktop"
    echo "   Download from: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check Docker Compose
echo ""
echo "2. Checking Docker Compose..."
if command_exists docker-compose || docker compose version >/dev/null 2>&1; then
    echo "   ✓ Docker Compose is available"
else
    echo "   ✗ Docker Compose not found"
    exit 1
fi

# Check services
echo ""
echo "3. Checking running services..."
docker-compose ps 2>/dev/null || docker compose ps 2>/dev/null

# Check health endpoint
echo ""
echo "4. Checking health endpoint..."
if curl -s http://localhost:8000/health 2>/dev/null | grep -q "healthy"; then
    echo "   ✓ Health check passed"
else
    echo "   ✗ Service not responding or not healthy"
fi

# Check metrics endpoint
echo ""
echo "5. Checking metrics endpoint..."
if curl -s http://localhost:8000/metrics 2>/dev/null | head -1 | grep -q "#"; then
    echo "   ✓ Metrics endpoint accessible"
else
    echo "   ✗ Metrics endpoint not accessible"
fi

# Test inference
echo ""
echo "6. Testing inference endpoint..."
response=$(curl -s -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "max_tokens": 10,
    "temperature": 0.7
  }' 2>/dev/null)

if echo "$response" | grep -q "generated_text"; then
    echo "   ✓ Inference endpoint working"
    echo "   Response: $response" | head -1
else
    echo "   ✗ Inference endpoint not working"
fi

# Check monitoring services
echo ""
echo "7. Checking monitoring services..."
if curl -s http://localhost:3000 2>/dev/null | grep -q "Grafana"; then
    echo "   ✓ Grafana is accessible at http://localhost:3000"
else
    echo "   ✗ Grafana not accessible"
fi

if curl -s http://localhost:9090 2>/dev/null | grep -q "Prometheus"; then
    echo "   ✓ Prometheus is accessible at http://localhost:9090"
else
    echo "   ✗ Prometheus not accessible"
fi

echo ""
echo "=== Validation Complete ==="
echo ""
echo "Summary:"
echo "- Docker: $(command_exists docker && echo "✓ Installed" || echo "✗ Not installed")"
echo "- Services: $(docker-compose ps 2>/dev/null | grep -q "Up" && echo "✓ Running" || echo "✗ Not running")"
echo "- API Health: $(curl -s http://localhost:8000/health 2>/dev/null | grep -q "healthy" && echo "✓ Healthy" || echo "✗ Not healthy")"
echo ""

# Instructions if Docker is not installed
if ! command_exists docker; then
    echo "=== Next Steps ==="
    echo "1. Install Docker Desktop for Windows:"
    echo "   https://www.docker.com/products/docker-desktop/"
    echo ""
    echo "2. After installation, restart your terminal and run:"
    echo "   ./start_deployment.bat"
fi