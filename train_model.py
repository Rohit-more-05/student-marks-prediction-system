"""
Student Intelligence Platform — Retraining Pipeline
Merges baseline data with new collected data and retrains the top model.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ===== PATHS =====
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== CONFIG =====
FEATURES = [
    "previous_sem_cgpa",
    "previous_to_previous_sem_cgpa",
    "number_of_backlogs",
    "attendance_percentage",
    "studytime",
    "goout",
]

def load_and_merge():
    print("📊 Loading baseline data...")
    # Load original datasets
    math_path = os.path.join(DATA_DIR, "student-mat.csv")
    por_path  = os.path.join(DATA_DIR, "student-por.csv")
    
    df_list = []
    if os.path.exists(math_path):
        m_df = pd.read_csv(math_path, sep=";")
        # Map to our 6 features
        m_df["previous_sem_cgpa"] = m_df["G2"] / 2.0
        m_df["previous_to_previous_sem_cgpa"] = m_df["G1"] / 2.0
        m_df["number_of_backlogs"] = m_df["failures"]
        m_df["attendance_percentage"] = (100 - m_df["absences"] * 1.5).clip(0, 100)
        m_df["target"] = (m_df["G3"] >= 10).astype(int)
        df_list.append(m_df[FEATURES + ["target"]])
        
    if os.path.exists(por_path):
        p_df = pd.read_csv(por_path, sep=";")
        p_df["previous_sem_cgpa"] = p_df["G2"] / 2.0
        p_df["previous_to_previous_sem_cgpa"] = p_df["G1"] / 2.0
        p_df["number_of_backlogs"] = p_df["failures"]
        p_df["attendance_percentage"] = (100 - p_df["absences"] * 1.5).clip(0, 100)
        p_df["target"] = (p_df["G3"] >= 10).astype(int)
        df_list.append(p_df[FEATURES + ["target"]])

    baseline_df = pd.concat(df_list, ignore_index=True)
    
    # Load collected data
    collected_path = os.path.join(DATA_DIR, "collected_data.csv")
    if os.path.exists(collected_path):
        print("📥 Merging new collected data...")
        collected_df = pd.read_csv(collected_path)
        # Ensure 'target' column exists in collected data
        if "target" in collected_df.columns:
            combined_df = pd.concat([baseline_df, collected_df], ignore_index=True)
            return combined_df
    
    return baseline_df

def train():
    df = load_and_merge()
    print(f"✅ Dataset ready: {len(df)} samples total.")
    
    X = df[FEATURES]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    # We use Logistic Regression as the baseline 'best' but you can plug others
    # In a real system, we might re-evaluate all 9 if needed.
    print("🏋️ Retraining best model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"📈 New Accuracy: {acc*100:.2f}%")
    
    # Save artifacts
    joblib.dump(model,   os.path.join(OUT_DIR, "best_model.pkl"))
    joblib.dump(model,   os.path.join(OUT_DIR, "logistic_regression_model.pkl"))
    joblib.dump(scaler,  os.path.join(OUT_DIR, "scaler.pkl"))
    joblib.dump(FEATURES, os.path.join(OUT_DIR, "selected_features.pkl"))
    
    # Update comparison placeholder for the UI
    results = [{"Model": "Logistic Regression", "Accuracy": acc}]
    joblib.dump(pd.DataFrame(results), os.path.join(OUT_DIR, "model_comparison.pkl"))
    
    print(f"🚀 Model updated in {OUT_DIR}")

if __name__ == "__main__":
    train()
