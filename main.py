from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and encoder at startup
# This is much faster than training!
try:
    model = joblib.load('model.joblib')
    le = joblib.load('encoder.joblib')
    print("Model and Encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    le = None

TARGET_FAMILIES = ['BREAD/BAKERY', 'DELI', 'DAIRY', 'EGGS', 'PRODUCE', 'GROCERY I']

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model or not le:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    try:
        data = request.json
        input_date = pd.to_datetime(data['date'])
        store_nbr = int(data['store_nbr'])
        is_promo = 1 if data.get('promo', False) else 0
        
        results = []
        
        for family_name in TARGET_FAMILIES:
            if family_name not in le.classes_:
                continue
                
            family_code = le.transform([family_name])[0]
            
            # Create feature vector
            input_features = pd.DataFrame({
                'store_nbr': [store_nbr],
                'family_code': [family_code],
                'onpromotion': [is_promo],
                'year': [input_date.year],
                'month': [input_date.month],
                'day': [input_date.day],
                'dayofweek': [input_date.dayofweek],
                'is_weekend': [1 if input_date.dayofweek >= 5 else 0]
            })
            
            pred_sales = model.predict(input_features)[0]
            results.append({
                "family": family_name,
                "prediction": round(pred_sales, 1)
            })
            
        return jsonify({"status": "success", "predictions": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 for Docker/Cloud compatibility
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))