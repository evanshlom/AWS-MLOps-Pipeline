from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model at startup
model_path = '/opt/ml/model/model.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    # Train model if not found (for local testing)
    from train import train
    train()
    model = joblib.load(model_path)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'healthy'})

@app.route('/invocations', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['instances'])
        predictions = model.predict(features).tolist()
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)