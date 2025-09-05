#!/bin/bash

set -e

ENVIRONMENT=${1:-production}
ACTION=${2:-deploy}
NAMESPACE="dps-system"
REGISTRY=${REGISTRY:-"your-registry.com"}
VERSION=${VERSION:-$(git rev-parse --short HEAD)}

echo "================================"
echo "Deep Parallel Synthesis Deployment"
echo "Environment: $ENVIRONMENT"
echo "Action: $ACTION"
echo "Version: $VERSION"
echo "================================"

function check_requirements() {
    echo "Checking requirements..."
    
    command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed."; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed."; exit 1; }
    
    if [ "$ENVIRONMENT" == "production" ]; then
        command -v helm >/dev/null 2>&1 || { echo "Helm is required for production deployment."; exit 1; }
    fi
    
    echo "✓ All requirements satisfied"
}

function build_image() {
    echo "Building Docker image..."
    
    docker build -t dps:$VERSION .
    docker tag dps:$VERSION $REGISTRY/dps:$VERSION
    docker tag dps:$VERSION $REGISTRY/dps:latest
    
    echo "✓ Docker image built: $REGISTRY/dps:$VERSION"
}

function push_image() {
    echo "Pushing image to registry..."
    
    docker push $REGISTRY/dps:$VERSION
    docker push $REGISTRY/dps:latest
    
    echo "✓ Image pushed to registry"
}

function download_model() {
    echo "Downloading model weights..."
    
    MODEL_PATH="./models/dps_model"
    
    if [ ! -d "$MODEL_PATH" ]; then
        mkdir -p $MODEL_PATH
        
        # Download from S3 or model hub
        if [ -n "$MODEL_S3_BUCKET" ]; then
            aws s3 sync s3://$MODEL_S3_BUCKET/dps-model/ $MODEL_PATH/
        else
            echo "Warning: No model source configured. Please set MODEL_S3_BUCKET or download manually."
        fi
    fi
    
    echo "✓ Model ready at $MODEL_PATH"
}

function deploy_kubernetes() {
    echo "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl apply -f deploy/k8s/namespace.yaml
    
    # Apply configurations
    kubectl apply -f deploy/k8s/storage.yaml
    
    # Update image in deployment
    kubectl set image deployment/dps-inference dps-inference=$REGISTRY/dps:$VERSION -n $NAMESPACE
    
    # Apply deployments
    kubectl apply -f deploy/k8s/deployment.yaml
    kubectl apply -f deploy/k8s/hpa.yaml
    
    # Wait for rollout
    kubectl rollout status deployment/dps-inference -n $NAMESPACE --timeout=600s
    
    echo "✓ Kubernetes deployment complete"
}

function deploy_docker_compose() {
    echo "Deploying with Docker Compose..."
    
    export DPS_VERSION=$VERSION
    docker-compose up -d
    
    echo "✓ Docker Compose deployment complete"
}

function health_check() {
    echo "Performing health check..."
    
    if [ "$ENVIRONMENT" == "kubernetes" ]; then
        SERVICE_IP=$(kubectl get service dps-inference-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        HEALTH_URL="http://$SERVICE_IP/health"
    else
        HEALTH_URL="http://localhost:8000/health"
    fi
    
    for i in {1..30}; do
        if curl -s $HEALTH_URL >/dev/null 2>&1; then
            echo "✓ Service is healthy"
            return 0
        fi
        echo "Waiting for service to be ready... ($i/30)"
        sleep 10
    done
    
    echo "✗ Health check failed"
    return 1
}

function rollback() {
    echo "Rolling back deployment..."
    
    if [ "$ENVIRONMENT" == "kubernetes" ]; then
        kubectl rollout undo deployment/dps-inference -n $NAMESPACE
        kubectl rollout status deployment/dps-inference -n $NAMESPACE
    else
        docker-compose down
        docker-compose up -d
    fi
    
    echo "✓ Rollback complete"
}

function scale() {
    REPLICAS=${3:-3}
    echo "Scaling to $REPLICAS replicas..."
    
    if [ "$ENVIRONMENT" == "kubernetes" ]; then
        kubectl scale deployment/dps-inference --replicas=$REPLICAS -n $NAMESPACE
    else
        docker-compose up -d --scale dps-inference=$REPLICAS
    fi
    
    echo "✓ Scaled to $REPLICAS replicas"
}

function monitor() {
    echo "Opening monitoring dashboards..."
    
    if [ "$ENVIRONMENT" == "kubernetes" ]; then
        kubectl port-forward -n $NAMESPACE service/grafana 3000:3000 &
        kubectl port-forward -n $NAMESPACE service/prometheus 9090:9090 &
        echo "Grafana: http://localhost:3000 (admin/admin)"
        echo "Prometheus: http://localhost:9090"
    else
        echo "Grafana: http://localhost:3000 (admin/admin)"
        echo "Prometheus: http://localhost:9090"
    fi
}

# Main execution
check_requirements

case $ACTION in
    deploy)
        build_image
        push_image
        download_model
        
        if [ "$ENVIRONMENT" == "kubernetes" ]; then
            deploy_kubernetes
        else
            deploy_docker_compose
        fi
        
        health_check
        ;;
    
    rollback)
        rollback
        health_check
        ;;
    
    scale)
        scale
        ;;
    
    monitor)
        monitor
        ;;
    
    build)
        build_image
        ;;
    
    push)
        push_image
        ;;
    
    *)
        echo "Unknown action: $ACTION"
        echo "Usage: ./deploy.sh [environment] [action]"
        echo "Environments: local, kubernetes, production"
        echo "Actions: deploy, rollback, scale, monitor, build, push"
        exit 1
        ;;
esac

echo "================================"
echo "Deployment complete!"
echo "================================"