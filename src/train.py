import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import boto3
import tarfile
import json

def train():
    # Load sample data (replace with your data)
    print("Loading data...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy}")
    
    # Save model temporarily to /tmp (writable)
    local_model_path = "/tmp/model.pkl"
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
    joblib.dump(model, local_model_path)
    print("Model saved temporarily")
    
    # Save model metadata temporarily
    metadata = {
        "accuracy": float(accuracy),
        "features": list(iris.feature_names),
        "classes": list(iris.target_names)
    }
    local_metadata_path = "/tmp/metadata.json"
    with open(local_metadata_path, "w") as f:
        json.dump(metadata, f)
    print("Metadata saved temporarily")
    
    # Upload to S3 output path
    s3 = boto3.client('s3')
    bucket = os.environ['SM_OUTPUT_DATA_DIR'].split('/')[2]  # Extract bucket name
    s3_prefix = os.environ['SM_OUTPUT_DATA_DIR'].replace(f"s3://{bucket}/", "")
    s3.upload_file(local_model_path, bucket, f"{s3_prefix}model.pkl")
    s3.upload_file(local_metadata_path, bucket, f"{s3_prefix}metadata.json")
    print("Model and metadata uploaded to S3")

if __name__ == "__main__":
    train()