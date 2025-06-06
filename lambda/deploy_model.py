import json
import boto3
import os
from datetime import datetime

sagemaker = boto3.client('sagemaker')
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """Deploy trained model to SageMaker endpoint"""
    
    # Configuration
    model_name = f"ml-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    endpoint_config_name = f"ml-endpoint-config-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    endpoint_name = os.environ.get('ENDPOINT_NAME', 'ml-model-endpoint')
    role_arn = os.environ['SAGEMAKER_ROLE_ARN']
    image_uri = os.environ['ECR_IMAGE_URI']
    model_artifacts = os.environ['MODEL_ARTIFACTS_S3_PATH']
    
    try:
        # Create model
        print(f"Creating model: {model_name}")
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_artifacts,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': model_artifacts
                }
            },
            ExecutionRoleArn=role_arn
        )
        
        # Create endpoint configuration
        print(f"Creating endpoint configuration: {endpoint_config_name}")
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'primary',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.t2.medium',
                'InitialVariantWeight': 1
            }]
        )
        
        # Check if endpoint exists
        try:
            sagemaker.describe_endpoint(EndpointName=endpoint_name)
            # Update existing endpoint
            print(f"Updating endpoint: {endpoint_name}")
            sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
        except sagemaker.exceptions.ClientError:
            # Create new endpoint
            print(f"Creating endpoint: {endpoint_name}")
            sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Model deployment initiated successfully',
                'endpoint_name': endpoint_name,
                'model_name': model_name
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }