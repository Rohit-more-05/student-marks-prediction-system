"""
Student Intelligence Platform — Retraining Pipeline (v2.2)
Merges baseline data with new collected data and retrains ALL 9 models.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
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
    print("Loading baseline data...")
    math_path = os.path.join(DATA_DIR, "student-mat.csv")
    por_path  = os.path.join(DATA_DIR, "student-por.csv")
    
    df_list = []
    if os.path.exists(math_path):
        m_df = pd.read_csv(math_path, sep=";")
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
    
    collected_path = os.path.join(DATA_DIR, "collected_data.csv")
    if os.path.exists(collected_path):
        print("Merging existing collected data...")
        collected_df = pd.read_csv(collected_path)
        if "target" in collected_df.columns:
            combined_df = pd.concat([baseline_df, collected_df], ignore_index=True)
            return combined_df
    
    return baseline_df

def train():
    df = load_and_merge()
    print(f"Dataset ready: {len(df)} samples total.")
    X = df[FEATURES]; y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes":         GaussianNB(),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42, verbosity=0),
        "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }

    results = []; best_acc = 0; best_name = ""

    print("Training 9 models...")
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        cv = cross_val_score(model, X_train_s, y_train, cv=5)
        
        results.append({"Model": name, "Accuracy": acc, "CV_Mean": cv.mean()})
        fname = f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, os.path.join(OUT_DIR, fname))
        
        if acc > best_acc:
            best_acc = acc; best_name = name
        print(f"  - {name}: {acc*100:.2f}%")

    joblib.dump(models[best_name], os.path.join(OUT_DIR, "best_model.pkl"))
    joblib.dump(scaler,            os.path.join(OUT_DIR, "scaler.pkl"))
    joblib.dump(FEATURES,          os.path.join(OUT_DIR, "selected_features.pkl"))
    joblib.dump(pd.DataFrame(results), os.path.join(OUT_DIR, "model_comparison.pkl"))
    
    print(f"Update complete. Best: {best_name} ({best_acc*100:.2f}%)")

if __name__ == "__main__":
    train()
