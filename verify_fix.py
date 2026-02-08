import joblib
import pandas as pd
import numpy as np
import sys
from logger_system import log_wrapper

@log_wrapper
def run_verification():
    print(f"[TEST] Python Executable: {sys.executable}")
    print("[TEST] Verifying Dependencies and Model Logic...")

    # 1. Verify Plotly
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("[SUCCESS] Plotly imported successfully.")
    except ImportError as e:
        print(f"[ERROR] Plotly import failed: {e}")
        return

    # 2. Verify Prediction Logic
    try:
        # Load Models
        scaler = joblib.load('models/scaler.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        try:
            model = joblib.load('models/random_forest_model.pkl')
        except:
            model = joblib.load('models/xgboost_model.pkl')

        print(f"Model expects: {getattr(model, 'n_features_in_', 'Unknown')} features.")
        
        # Construct Mock Input
        input_vector = pd.DataFrame(0, index=[0], columns=feature_cols)
        input_vector['G1'] = 15
        input_vector['G2'] = 16
        
        # --- APP LOGIC REPLICATION ---
        final_input = input_vector.copy()
        
        # Force Alignment based on Scaler
        if hasattr(scaler, 'feature_names_in_'):
            model_cols = list(scaler.feature_names_in_)
            print(f"Aligning to scaler's {len(model_cols)} features...")
            final_input = final_input[model_cols]
        else:
            # Fallback
            cols_to_drop = ['G1', 'G2']
            final_input = final_input.drop(columns=[c for c in cols_to_drop if c in final_input.columns], errors='ignore')
        
        print(f"Input shape after alignment: {final_input.shape}")

        if hasattr(scaler, 'feature_names_in_'):
            final_input = pd.DataFrame(scaler.transform(final_input), columns=final_input.columns)
        else:
            final_input = scaler.transform(final_input)
            
        # Predict
        proba = model.predict_proba(final_input)[0]
        print(f"[SUCCESS] Prediction Probabilities: {proba}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Logic Failed: {str(e)}")

if __name__ == "__main__":
    run_verification()
