from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train():
    # Generate simple dataset
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('/opt/ml/model', exist_ok=True)
    joblib.dump(model, '/opt/ml/model/model.joblib')
    
    print(f"Model trained. Accuracy: {model.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    train()