from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import joblib
import os

app = Flask(__name__)

# 1. ALLOW ALL ORIGINS (Fixes your CORS error permanently)
CORS(app, resources={r"/*": {"origins": "*"}})

# 2. LOAD LOCAL WEIGHTS
# Since you pushed them to GitHub, they are just local files now.
try:
    print("Loading model files from local disk...")
    model = joblib.load('model.joblib')
    le = joblib.load('encoder.joblib')
    print("✅ Model and Encoder loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # It is better to crash now than to run a broken server
    model = None
    le = None

TARGET_FAMILIES = ['BREAD/BAKERY', 'DELI', 'DAIRY', 'EGGS', 'PRODUCE', 'GROCERY I']

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
@cross_origin() # Ensures the browser's pre-check passes
def predict():
    # Handle Preflight OPTIONS request manually if needed
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    if not model or not le:
        return jsonify({"status": "error", "message": "Model not loaded properly"}), 500

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
                "prediction": round(float(pred_sales), 1)
            })
            
        return jsonify({"status": "success", "predictions": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def _build_cors_preflight_response():
    """Helper to send the correct headers for browser checks"""
    response = jsonify({})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response

if __name__ == '__main__':
    # 3. SET DEBUG=FALSE for Production
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
