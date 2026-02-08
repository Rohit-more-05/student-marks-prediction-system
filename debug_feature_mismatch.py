import joblib
import pandas as pd
import numpy as np
from logger_system import log_wrapper

@log_wrapper
def run_debug():
    print("[DEBUG] Analyzing Feature Mismatch...")
    try:
        # Load Artifacts
        feature_cols = joblib.load('models/feature_columns.pkl')
        model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')

        print(f"Feature Columns in PKL: {len(feature_cols)}")
        print(f"Model Expects: {getattr(model, 'n_features_in_', 'Unknown')}")
        
        if hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)
            print(f"Model Feature Names Available: Yes ({len(model_features)})")
            
            # Find difference
            extra_in_pkl = set(feature_cols) - set(model_features)
            missing_in_pkl = set(model_features) - set(feature_cols)
            
            print(f"\n[CRITICAL] Extra columns in 'feature_columns.pkl' that model doesn't know: {extra_in_pkl}")
            print(f"[CRITICAL] Columns model needs but missing in pkl: {missing_in_pkl}")
            
        else:
            print("Model does not have 'feature_names_in_'. Cannot compare by name, only count.")

    except Exception as e:
        print(f"[ERROR] {str(e)}")

if __name__ == "__main__":
    run_debug()
