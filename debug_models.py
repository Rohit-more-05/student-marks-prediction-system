import joblib
import pandas as pd
import numpy as np

print("[DEBUG] Inspecting artifacts...")

try:
    scaler = joblib.load('models/scaler.pkl')
    feature_cols = joblib.load('models/feature_columns.pkl')
    
    print(f"Feature Columns (Count: {len(feature_cols)})")
    
    has_G1 = 'G1' in feature_cols
    has_G2 = 'G2' in feature_cols
    print(f"feature_cols has G1: {has_G1}")
    print(f"feature_cols has G2: {has_G2}")

    if hasattr(scaler, 'feature_names_in_'):
        print(f"\nScaler Feature Names In ({len(scaler.feature_names_in_)})")
        scaler_feats = set(scaler.feature_names_in_)
        
        print(f"Scaler has G1: {'G1' in scaler_feats}")
        print(f"Scaler has G2: {'G2' in scaler_feats}")
        
        missing_in_scaler = [c for c in feature_cols if c not in scaler_feats]
        print(f"Present in feature_cols BUT MISSING in Scaler: {missing_in_scaler}")
        
    else:
        print(f"\nScaler does not have 'feature_names_in_'.")

except Exception as e:
    print(f"[ERROR] {str(e)}")
