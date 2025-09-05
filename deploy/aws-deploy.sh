#!/bin/bash

set -e

AWS_REGION=${AWS_REGION:-"us-west-2"}
CLUSTER_NAME=${CLUSTER_NAME:-"dps-cluster"}
INSTANCE_TYPE=${INSTANCE_TYPE:-"p3.8xlarge"}
MIN_NODES=${MIN_NODES:-2}
MAX_NODES=${MAX_NODES:-10}
S3_BUCKET=${S3_BUCKET:-"dps-models"}

echo "================================"
echo "DPS AWS EKS Deployment"
echo "Region: $AWS_REGION"
echo "Cluster: $CLUSTER_NAME"
echo "================================"

function setup_eks_cluster() {
    echo "Setting up EKS cluster..."
    
    cat <<EOF > eks-cluster-config.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $AWS_REGION
  version: "1.27"

iam:
  withOIDC: true

managedNodeGroups:
  - name: gpu-nodes
    instanceType: $INSTANCE_TYPE
    minSize: $MIN_NODES
    maxSize: $MAX_NODES
    desiredCapacity: $MIN_NODES
    volumeSize: 500
    volumeType: gp3
    volumeIOPS: 16000
    volumeThroughput: 1000
    labels:
      workload: gpu
      node.kubernetes.io/gpu: "true"
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
    preBootstrapCommands:
      - |
        #!/bin/bash
        /etc/eks/bootstrap.sh $CLUSTER_NAME
        nvidia-smi
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        ebs: true
        efs: true

  - name: cpu-nodes
    instanceType: c5.4xlarge
    minSize: 1
    maxSize: 5
    desiredCapacity: 2
    volumeSize: 100
    labels:
      workload: cpu

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver
EOF

    eksctl create cluster -f eks-cluster-config.yaml
    
    # Install NVIDIA device plugin
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
    
    echo "✓ EKS cluster created"
}

function setup_s3_model_storage() {
    echo "Setting up S3 model storage..."
    
    # Create S3 bucket if not exists
    aws s3api create-bucket \
        --bucket $S3_BUCKET \
        --region $AWS_REGION \
        --create-bucket-configuration LocationConstraint=$AWS_REGION || true
    
    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket $S3_BUCKET \
        --versioning-configuration Status=Enabled
    
    # Upload model if local exists
    if [ -d "./models/dps_model" ]; then
        aws s3 sync ./models/dps_model/ s3://$S3_BUCKET/dps-model/ \
            --storage-class INTELLIGENT_TIERING
    fi
    
    echo "✓ S3 model storage configured"
}

function setup_efs() {
    echo "Setting up EFS for shared storage..."
    
    # Get VPC ID
    VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --region $AWS_REGION \
        --query "cluster.resourcesVpcConfig.vpcId" --output text)
    
    # Create EFS
    EFS_ID=$(aws efs create-file-system \
        --region $AWS_REGION \
        --performance-mode generalPurpose \
        --throughput-mode elastic \
        --encrypted \
        --tags "Key=Name,Value=dps-efs" \
        --query "FileSystemId" --output text)
    
    # Wait for EFS to be available
    aws efs describe-file-systems --file-system-id $EFS_ID \
        --region $AWS_REGION --query "FileSystems[0].LifeCycleState" \
        --output text | grep -q available || sleep 30
    
    # Install EFS CSI driver
    kubectl apply -k "github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.5"
    
    echo "✓ EFS configured: $EFS_ID"
}

function setup_alb() {
    echo "Setting up Application Load Balancer..."
    
    # Install AWS Load Balancer Controller
    eksctl utils associate-iam-oidc-provider \
        --cluster $CLUSTER_NAME \
        --region $AWS_REGION \
        --approve
    
    # Create IAM policy
    curl -o iam_policy.json https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.5.4/docs/install/iam_policy.json
    
    aws iam create-policy \
        --policy-name AWSLoadBalancerControllerIAMPolicy \
        --policy-document file://iam_policy.json || true
    
    # Create service account
    eksctl create iamserviceaccount \
        --cluster=$CLUSTER_NAME \
        --namespace=kube-system \
        --name=aws-load-balancer-controller \
        --attach-policy-arn=arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/AWSLoadBalancerControllerIAMPolicy \
        --override-existing-serviceaccounts \
        --approve
    
    # Install controller
    helm repo add eks https://aws.github.io/eks-charts
    helm repo update
    helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName=$CLUSTER_NAME \
        --set serviceAccount.create=false \
        --set serviceAccount.name=aws-load-balancer-controller
    
    echo "✓ ALB controller installed"
}

function deploy_dps() {
    echo "Deploying DPS to EKS..."
    
    # Update storage class for AWS
    cat <<EOF > dps-storageclass.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "16000"
  throughput: "1000"
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
volumeBindingMode: WaitForFirstConsumer
EOF
    
    kubectl apply -f dps-storageclass.yaml
    
    # Deploy DPS
    kubectl apply -f deploy/k8s/namespace.yaml
    kubectl apply -f deploy/k8s/storage.yaml
    kubectl apply -f deploy/k8s/deployment.yaml
    kubectl apply -f deploy/k8s/hpa.yaml
    
    # Create ingress
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: dps-ingress
  namespace: dps-system
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/certificate-arn: $SSL_CERT_ARN
spec:
  rules:
  - host: dps.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: dps-inference-service
            port:
              number: 80
EOF
    
    echo "✓ DPS deployed to EKS"
}

function setup_monitoring() {
    echo "Setting up monitoring with CloudWatch..."
    
    # Install CloudWatch Container Insights
    ClusterName=$CLUSTER_NAME
    RegionName=$AWS_REGION
    FluentBitHttpPort='2020'
    FluentBitReadFromHead='Off'
    
    curl https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/quickstart/cwagent-fluent-bit-quickstart.yaml | \
        sed "s/{{cluster_name}}/${ClusterName}/;s/{{region_name}}/${RegionName}/;s/{{http_server_port}}/${FluentBitHttpPort}/;s/{{read_from_head}}/${FluentBitReadFromHead}/" | \
        kubectl apply -f -
    
    echo "✓ CloudWatch monitoring configured"
}

function setup_autoscaling() {
    echo "Setting up Cluster Autoscaler..."
    
    # Deploy Cluster Autoscaler
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml
    
    kubectl -n kube-system annotate deployment.apps/cluster-autoscaler \
        cluster-autoscaler.kubernetes.io/safe-to-evict="false"
    
    kubectl -n kube-system set image deployment.apps/cluster-autoscaler \
        cluster-autoscaler=k8s.gcr.io/autoscaling/cluster-autoscaler:v1.27.0
    
    kubectl -n kube-system edit deployment.apps/cluster-autoscaler
    
    echo "✓ Cluster Autoscaler configured"
}

function create_backup() {
    echo "Setting up backup with AWS Backup..."
    
    # Create backup plan
    aws backup create-backup-plan \
        --backup-plan '{
            "BackupPlanName": "dps-backup-plan",
            "Rules": [{
                "RuleName": "DailyBackups",
                "TargetBackupVaultName": "Default",
                "ScheduleExpression": "cron(0 5 ? * * *)",
                "StartWindowMinutes": 60,
                "CompletionWindowMinutes": 120,
                "Lifecycle": {
                    "DeleteAfterDays": 30
                }
            }]
        }'
    
    echo "✓ Backup configured"
}

# Main deployment flow
case ${1:-full} in
    full)
        setup_eks_cluster
        setup_s3_model_storage
        setup_efs
        setup_alb
        deploy_dps
        setup_monitoring
        setup_autoscaling
        create_backup
        ;;
    
    cluster)
        setup_eks_cluster
        ;;
    
    deploy)
        deploy_dps
        ;;
    
    monitoring)
        setup_monitoring
        ;;
    
    destroy)
        eksctl delete cluster --name $CLUSTER_NAME --region $AWS_REGION
        ;;
    
    *)
        echo "Usage: ./aws-deploy.sh [full|cluster|deploy|monitoring|destroy]"
        exit 1
        ;;
esac

echo "================================"
echo "AWS Deployment Complete!"
echo "Cluster: $CLUSTER_NAME"
echo "Region: $AWS_REGION"
echo "================================"