import joblib
import os
import pandas as pd
import numpy as np

def test_load():
    print(f"Current Working Directory: {os.getcwd()}")
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print(f"[ERROR] Models directory '{models_dir}' not found!")
        return

    files = {
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'Support Vector Machine': 'support_vector_machine_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl'
    }

    for name, f in files.items():
        path = os.path.join(models_dir, f)
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                print(f"[SUCCESS] {name} loaded. Classes: {model.classes_}")
            except Exception as e:
                print(f"[ERROR] {name} load failed: {e}")
        else:
            print(f"[ERROR] {name} file NOT found at {path}")

if __name__ == "__main__":
    test_load()
