# Prepare AWS Deployment Script

Write-Host "=== Preparing for AWS Deployment ===" -ForegroundColor Cyan
Write-Host ""

# Function to check if AWS CLI is installed
function Test-AWSCLIInstalled {
    try {
        aws --version | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Check AWS CLI
if (Test-AWSCLIInstalled) {
    $awsVersion = aws --version
    Write-Host "[✓] AWS CLI found: $awsVersion" -ForegroundColor Green
} else {
    Write-Host "[✗] AWS CLI not installed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Installing AWS CLI..." -ForegroundColor Yellow
    
    # Download AWS CLI installer
    $installerUrl = "https://awscli.amazonaws.com/AWSCLIV2.msi"
    $installerPath = "$env:TEMP\AWSCLIV2.msi"
    
    Write-Host "Downloading AWS CLI installer..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
    
    Write-Host "Running installer..." -ForegroundColor Yellow
    Start-Process msiexec.exe -ArgumentList "/i", $installerPath, "/quiet" -Wait
    
    Write-Host "[✓] AWS CLI installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "Configuring AWS credentials..." -ForegroundColor Yellow
Write-Host "You'll need:" -ForegroundColor Cyan
Write-Host "  • AWS Access Key ID" -ForegroundColor White
Write-Host "  • AWS Secret Access Key" -ForegroundColor White
Write-Host "  • Default region (recommended: us-west-2)" -ForegroundColor White
Write-Host ""

# Check if credentials already exist
if (Test-Path "$env:USERPROFILE\.aws\credentials") {
    Write-Host "[✓] AWS credentials file exists" -ForegroundColor Green
    $reconfigure = Read-Host "Do you want to reconfigure AWS credentials? (y/n)"
    if ($reconfigure -eq 'y') {
        aws configure
    }
} else {
    Write-Host "Setting up AWS credentials..." -ForegroundColor Yellow
    aws configure
}

Write-Host ""
Write-Host "Creating deployment configuration..." -ForegroundColor Yellow

# Create deployment config
$config = @"
# AWS Configuration
AWS_REGION=us-west-2
CLUSTER_NAME=dps-cluster-$(Get-Date -Format 'yyyyMMdd')
INSTANCE_TYPE=g5.2xlarge
MIN_NODES=1
MAX_NODES=3
S3_BUCKET=dps-models-$(Get-Date -Format 'yyyyMMddHHmmss')
USE_SPOT_INSTANCES=true
ENVIRONMENT=production
"@

$configPath = "deploy\aws-config.env"
$config | Out-File -FilePath $configPath -Encoding UTF8
Write-Host "[✓] Configuration saved to $configPath" -ForegroundColor Green

Write-Host ""
Write-Host "Checking AWS pricing for GPU instances..." -ForegroundColor Yellow

try {
    $pricing = aws ec2 describe-spot-price-history `
        --instance-types "p3.2xlarge" "g5.2xlarge" `
        --max-results 5 `
        --product-descriptions "Linux/UNIX" `
        --region us-west-2 `
        --output json | ConvertFrom-Json
    
    Write-Host ""
    Write-Host "Current Spot Prices (us-west-2):" -ForegroundColor Cyan
    foreach ($price in $pricing.SpotPriceHistory) {
        Write-Host "  • $($price.InstanceType): $$($price.SpotPrice)/hour" -ForegroundColor White
    }
} catch {
    Write-Host "[⚠] Could not fetch pricing information" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Estimated Monthly Costs:" -ForegroundColor Cyan
Write-Host "  • Development (1 node):  ~$500-800/month" -ForegroundColor White
Write-Host "  • Production (2 nodes):  ~$1,500-2,300/month" -ForegroundColor White
Write-Host "  • Scale (auto-scaling):  ~$3,000-5,000/month" -ForegroundColor White

Write-Host ""
Write-Host "=== AWS Preparation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Verify local deployment is working" -ForegroundColor White
Write-Host "2. Review costs and select instance type" -ForegroundColor White
Write-Host "3. Run AWS deployment:" -ForegroundColor White
Write-Host "   .\deploy\aws-deploy.sh full" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to continue"