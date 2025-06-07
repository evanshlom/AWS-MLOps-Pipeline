import boto3
import json
import argparse
import time

def test_endpoint(endpoint_name, region='us-east-1'):
    """Test SageMaker endpoint with sample data"""
    
    # Initialize SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Sample test data (4 samples from iris dataset)
    test_data = {
        "instances": [
            [5.1, 3.5, 1.4, 0.2],  # Setosa
            [7.0, 3.2, 4.7, 1.4],  # Versicolor
            [6.3, 3.3, 6.0, 2.5],  # Virginica
            [5.9, 3.0, 4.2, 1.5]   # Versicolor
        ]
    }
    
    print(f"Testing endpoint: {endpoint_name}")
    print(f"Input data: {json.dumps(test_data, indent=2)}")
    
    try:
        # Invoke endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(test_data)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        print("\nEndpoint Response:")
        print(f"Predictions: {result['predictions']}")
        print(f"Probabilities: {result['probabilities']}")
        
        # Map predictions to class names
        class_names = ['setosa', 'versicolor', 'virginica']
        for i, pred in enumerate(result['predictions']):
            print(f"\nSample {i+1}: Predicted class = {class_names[pred]}")
            probs = result['probabilities'][i]
            for j, prob in enumerate(probs):
                print(f"  {class_names[j]}: {prob:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing endpoint: {str(e)}") 
        return False

def wait_for_endpoint(endpoint_name, region='us-east-1', max_wait=600):
    """Wait for endpoint to be in service"""
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    print(f"Waiting for endpoint {endpoint_name} to be in service...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            print(f"Endpoint status: {status}")
            
            if status == 'InService':
                print("Endpoint is ready!")
                return True
            elif status in ['Failed', 'RollingBack']:
                print(f"Endpoint deployment failed with status: {status}")
                return False
            
        except Exception as e:
            print(f"Error checking endpoint status: {str(e)}")
        
        time.sleep(30)
    
    print("Timeout waiting for endpoint")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SageMaker endpoint')
    parser.add_argument('--endpoint-name', default='ml-model-endpoint', help='Endpoint name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--wait', action='store_true', help='Wait for endpoint to be ready')
    
    args = parser.parse_args()
    
    if args.wait:
        if wait_for_endpoint(args.endpoint_name, args.region):
            test_endpoint(args.endpoint_name, args.region)
    else:
        test_endpoint(args.endpoint_name, args.region)