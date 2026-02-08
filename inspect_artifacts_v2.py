import joblib
import pandas as pd
from logger_system import log_wrapper

@log_wrapper
def run_inspection():
    print("--- ARTIFACT INSPECTION ---")
    try:
        feature_cols = joblib.load('models/feature_columns.pkl')
        print(f"ALL Columns in feature_columns.pkl ({len(feature_cols)}):")
        print(feature_cols)
        
        scaler = joblib.load('models/scaler.pkl')
        if hasattr(scaler, 'feature_names_in_'):
            print(f"\nALL Columns in Scaler ({len(scaler.feature_names_in_)}):")
            print(list(scaler.feature_names_in_))
            
        model = joblib.load('models/random_forest_model.pkl')
        print(f"\nModel Expects (Count): {getattr(model, 'n_features_in_', 'Unknown')}")
        if hasattr(model, 'feature_names_in_'):
            print(f"Model feature names: Yes ({len(model.feature_names_in_)})")
            print(f"G1 in model: {'G1' in model.feature_names_in_}")
        else:
            print("Model feature names: No")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_inspection()
