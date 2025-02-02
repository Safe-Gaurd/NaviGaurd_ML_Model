from flask import Flask, request, jsonify
import numpy as np
import joblib  # Ensure you have joblib installed for loading ML models

app = Flask(__name__)

# Load all models, scaler, and encoder from the saved file
model_data = joblib.load("models.pkl")

slow_region_model = model_data['slow_region_model']
group_model = model_data['group_model']
scaler = model_data['scaler']
label_encoder = model_data['label_encoder']  # Label encoder for decoding group

@app.route("/")
def home():
    return "Welcome to my model!"
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get the JSON payload
        
        # Validate input
        if 'latitude' not in data or 'longitude' not in data:
            return jsonify({'error': 'Missing latitude or longitude'}), 400

        # Prepare input data
        input_data = np.array([[data['latitude'], data['longitude']]])  # Keep it 2D

        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(input_data)

        # Make predictions
        slow_region_pred = slow_region_model.predict(scaled_data)[0]
        group_pred_encoded = group_model.predict(scaled_data)[0]

        # Decode the group prediction
        group_pred = label_encoder.inverse_transform([group_pred_encoded])[0]

        return jsonify({'slow_region': str(slow_region_pred), 'group': group_pred})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
