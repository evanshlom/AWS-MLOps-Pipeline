import joblib
import json
import numpy as np
import os

def model_fn(model_dir):
    """Load model for inference""" 
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    return model

def input_fn(request_body, content_type="application/json"):
    """Parse input data"""
    if content_type == "application/json":
        input_data = json.loads(request_body)
        return np.array(input_data["instances"])
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    }

def output_fn(prediction, content_type="application/json"):
    """Format output"""
    if content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")