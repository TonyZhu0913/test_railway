# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder
# import os

# app = Flask(__name__)
# CORS(app)  # 允許跨網域請求，讓前端 HTML 可以呼叫此 API

# # ==========================================
# # 1. 初始化與模型訓練 (伺服器啟動時執行)
# # ==========================================
# print("正在初始化模型...")

# # 定義全域變數
# model = None
# le = None
# TARGET_FAMILIES = ['BREAD/BAKERY', 'DELI', 'DAIRY', 'EGGS', 'PRODUCE', 'GROCERY I']

# # 檢查資料是否存在
# if not os.path.exists('small_data.csv'):
#     print("錯誤：找不到 small_data.csv，請確保檔案在同一目錄下。")
# else:
#     # 讀取資料
#     df = pd.read_csv('small_data.csv')
    
#     # 資料前處理 (Feature Engineering) - 必須與訓練時一致
#     df['date'] = pd.to_datetime(df['date'])
#     df['year'] = df['date'].dt.year
#     df['month'] = df['date'].dt.month
#     df['day'] = df['date'].dt.day
#     df['dayofweek'] = df['date'].dt.dayofweek
#     # 特徵：是否為週末 (週六=5, 週日=6)
#     df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

#     # 編碼商品類別 (Family)
#     le = LabelEncoder()
#     df['family_code'] = le.fit_transform(df['family'])

#     # 選定特徵 (X) 與 目標 (y)
#     features = ['store_nbr', 'family_code', 'onpromotion', 'year', 'month', 'day', 'dayofweek', 'is_weekend']
#     X = df[features]
#     y = df['sales']

#     # 訓練隨機森林模型
#     print("正在訓練 Random Forest 模型，請稍候...")
#     model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#     model.fit(X, y)
#     print("模型初始化完成！API 準備就緒。")

# # ==========================================
# # 2. API 端點
# # ==========================================
# @app.route('/api/predict', methods=['POST'])
# def predict():
#     try:
#         # 檢查模型是否已訓練
#         if model is None:
#             return jsonify({"status": "error", "message": "Model not trained (small_data.csv missing?)"}), 500

#         # 接收前端傳來的 JSON 資料
#         data = request.json
#         # 格式範例: { "date": "2013-01-02", "store_nbr": "1", "promo": true }
        
#         input_date = pd.to_datetime(data['date'])
#         store_nbr = int(data['store_nbr'])
#         is_promo = 1 if data.get('promo', False) else 0
        
#         results = []
        
#         # 針對我們關注的每一個商品類別進行預測
#         for family_name in TARGET_FAMILIES:
#             # 安全檢查：如果 CSV 中沒有這個類別，LabelEncoder 會報錯，所以要先檢查
#             if family_name not in le.classes_:
#                 continue
                
#             family_code = le.transform([family_name])[0]
            
#             # 建構特徵向量 (DataFrame 欄位順序必須與訓練時完全一致)
#             input_features = pd.DataFrame({
#                 'store_nbr': [store_nbr],
#                 'family_code': [family_code],
#                 'onpromotion': [is_promo],
#                 'year': [input_date.year],
#                 'month': [input_date.month],
#                 'day': [input_date.day],
#                 'dayofweek': [input_date.dayofweek],
#                 'is_weekend': [1 if input_date.dayofweek >= 5 else 0]
#             })
            
#             # 進行預測
#             pred_sales = model.predict(input_features)[0]
            
#             # 將結果加入回傳列表
#             results.append({
#                 "family": family_name,
#                 "prediction": round(pred_sales, 1) # 四捨五入到小數點第一位
#             })
            
#         return jsonify({"status": "success", "predictions": results})

#     except Exception as e:
#         print(f"API Error: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

# if __name__ == '__main__':
#     print("-" * 50)
#     print("伺服器啟動中...")
#     print("請打開 final_demo_real_data.html")
#     print("API 地址: http://127.0.0.1:5000/api/predict")
#     print("-" * 50)
#     app.run(debug=True, port=5000)
