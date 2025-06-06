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
    
    # Save model
    model_path = "/opt/ml/model"
    os.makedirs(model_path, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_path, "model.pkl"))
    print("Model saved successfully")
    
    # Save model metadata
    metadata = {
        "accuracy": float(accuracy),
        "features": list(iris.feature_names),
        "classes": list(iris.target_names)
    }
    
    with open(os.path.join(model_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    train()