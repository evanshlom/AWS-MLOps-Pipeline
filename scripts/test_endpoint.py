import boto3
import json
import numpy as np

def test_endpoint():
    client = boto3.client('sagemaker-runtime')
    
    # Test data
    test_data = {
        'instances': [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]
    }
    
    try:
        response = client.invoke_endpoint(
            EndpointName='simple-ml-endpoint',
            ContentType='application/json',
            Body=json.dumps(test_data)
        )
        
        result = json.loads(response['Body'].read().decode())
        print(f"Predictions: {result}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_endpoint()