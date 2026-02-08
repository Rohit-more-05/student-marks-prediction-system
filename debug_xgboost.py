import joblib
import sys
import os

try:
    print("Attempting to load XGBoost model...")
    model = joblib.load('models/xgboost_model.pkl')
    print("Successfully loaded XGBoost model.")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    import traceback
    traceback.print_exc()
