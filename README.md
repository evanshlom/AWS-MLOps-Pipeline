# Simple ML SageMaker Deployment

## Project Structure
```
simple-ml-sagemaker/
├── Dockerfile
├── README.md
├── requirements.txt
├── .github/
│   └── workflows/
│       └── deploy.yml
├── src/
│   ├── train.py
│   └── inference.py
└── scripts/
    └── test_endpoint.py
```

## Setup Steps

### 1. Create ECR Repository
```bash
aws ecr create-repository --repository-name simple-ml --region us-east-1
```

Result:
```bash
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:429099991557:repository/simple-ml",
        "registryId": "429099991557",
        "repositoryName": "simple-ml",
        "repositoryUri": "429099991557.dkr.ecr.us-east-1.amazonaws.com/simple-ml",
        "createdAt": "2025-06-07T20:07:22.366000-07:00",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}
```

### 2. Create SageMaker Execution Role
Create an IAM role named `SageMakerExecutionRole` with:
- **Policies**: AmazonSageMakerFullAccess
- **Trust Policy**: Allow sagemaker.amazonaws.com to assume the role

### 3. GitHub Secrets
Add these secrets to your GitHub repository:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY` 
- `AWS_REGION` (e.g., us-east-1)
- `AWS_ACCOUNT_ID`

## Local Testing
```bash
# Build and run locally
docker build -t simple-ml .
docker run -p 8080:8080 simple-ml

# Test locally
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.0, 2.0, 3.0, 4.0]]}'
```

## Deployment
1. Push code to main branch
2. GitHub Actions automatically builds, pushes to ECR, and deploys to SageMaker
3. Test the endpoint: `python scripts/test_endpoint.py`

## How It Works
- **Training**: Creates a simple classification dataset and trains a RandomForest
- **Inference**: Flask server with `/ping` and `/invocations` endpoints
- **Deployment**: Containerized model deployed to SageMaker endpoint