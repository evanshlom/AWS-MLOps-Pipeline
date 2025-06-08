import boto3
import json
import numpy as np

def test_endpoint():
    # Create SageMaker runtime client
    client = boto3.client('sagemaker-runtime', region_name='us-east-1')
    
    # Test data - 4 features since the model was trained on 4-feature data
    test_data = {
        'instances': [
            [1.0, 2.0, 3.0, 4.0],        # First test sample
            [0.5, 1.5, 2.5, 3.5],        # Second test sample  
            [-1.0, 0.0, 1.0, 2.0],       # Third test sample
            [2.5, 3.0, 1.5, 0.5]         # Fourth test sample
        ]
    }
    
    try:
        print("Sending prediction request to SageMaker endpoint...")
        print(f"Test data: {test_data}")
        
        response = client.invoke_endpoint(
            EndpointName='simple-ml-endpoint',
            ContentType='application/json',
            Body=json.dumps(test_data)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        print(f"\n✅ Success!")
        print(f"Predictions: {result['predictions']}")
        print(f"Classes predicted: {result['predictions']} (0 or 1)")
        
        # Show input vs output
        print(f"\nInput → Output:")
        for i, (input_features, prediction) in enumerate(zip(test_data['instances'], result['predictions'])):
            print(f"  Sample {i+1}: {input_features} → Class {prediction}")
            
    except Exception as e:
        print(f"❌ Error calling endpoint: {e}")
        print("\nTroubleshooting:")
        print("1. Check if endpoint is InService: aws sagemaker describe-endpoint --endpoint-name simple-ml-endpoint")
        print("2. Verify your AWS credentials are configured")
        print("3. Make sure you're in the right region (us-east-1)")

if __name__ == "__main__":
    test_endpoint()