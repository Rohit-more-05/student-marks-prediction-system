"""
Student Intelligence Platform — Retraining Pipeline (v3.0)
Fixes applied:
  - Honest feature names (period_2_score, period_1_score, not 'CGPA')
  - Deduplication of ~382 students shared across Math+Portuguese datasets
  - Stratified train/test split (consistent with notebook)
  - Full evaluation: Accuracy, F1-macro, Precision/Recall (fail class), AUC-ROC
  - Dummy classifier baseline added to leaderboard
  - Removed deprecated use_label_encoder from XGBoost (xgboost>=2.0)
  - importlib-safe: train() is callable from app.py without subprocess
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost not found. Skipping XGBoost.")

# ===== PATHS =====
BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== FEATURE CONFIG =====
# Honest feature names — these map directly to G2/2, G1/2, failures, absences-derived, studytime, goout
FEATURES = [
    "period_2_score",       # G2/2.0 — second period score scaled 0-10
    "period_1_score",       # G1/2.0 — first period score scaled 0-10
    "number_of_backlogs",   # failures {0,1,2,3}
    "absences_inverse",     # (75 - absences) / 75 * 100 — higher=better attendance proxy
    "studytime",            # 1-4 ordinal scale
    "goout",                # 1-5 ordinal scale
]

# UCI merge key — 382 students appear in BOTH math and Portuguese datasets
# Source: https://archive.ics.uci.edu/dataset/320/student+performance
_UCI_MERGE_COLS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup",
    "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences",
]

def _engineer(df_raw):
    """Apply honest feature engineering to a raw UCI dataframe."""
    df = df_raw.copy()
    df["period_2_score"]      = df["G2"] / 2.0
    df["period_1_score"]      = df["G1"] / 2.0
    df["number_of_backlogs"]  = df["failures"].clip(0, 3)
    # absences_inverse: 0 absences=100, 75 absences=0 (max observed in dataset is 75)
    df["absences_inverse"]    = ((75 - df["absences"]) / 75 * 100).clip(0, 100)
    df["target"]              = (df["G3"] >= 10).astype(int)
    return df


def load_and_merge():
    """Load, deduplicate, and engineer features from the UCI Student Performance data."""
    math_path = os.path.join(DATA_DIR, "student-mat.csv")
    por_path  = os.path.join(DATA_DIR, "student-por.csv")

    math_df = pd.read_csv(math_path, sep=";") if os.path.exists(math_path) else None
    por_df  = pd.read_csv(por_path,  sep=";") if os.path.exists(por_path)  else None

    if math_df is None and por_df is None:
        raise FileNotFoundError("Neither student-mat.csv nor student-por.csv found in data/")

    # --- Deduplication ---
    # UCI documents ~382 students appear in both datasets.
    # Strategy: keep math record for shared students; use Portuguese-only for the rest.
    if math_df is not None and por_df is not None:
        valid_merge_cols = [c for c in _UCI_MERGE_COLS if c in math_df.columns and c in por_df.columns]
        shared = pd.merge(math_df, por_df, on=valid_merge_cols, how="inner", suffixes=("_mat", "_por"))
        n_shared = len(shared)
        print(f"  Deduplication: Math={len(math_df)}, Portuguese={len(por_df)}, Shared~={n_shared}")

        # Mark shared students in por_df by merging on those cols; drop their rows
        por_df["_shared"] = por_df[valid_merge_cols].apply(tuple, axis=1).isin(
            math_df[valid_merge_cols].apply(tuple, axis=1)
        )
        por_unique = por_df[~por_df["_shared"]].drop(columns=["_shared"])
        math_eng  = _engineer(math_df)
        por_eng   = _engineer(por_unique)
        baseline_df = pd.concat([math_eng, por_eng], ignore_index=True)
        print(f"  Final dataset: {len(baseline_df)} unique student records")
    elif math_df is not None:
        baseline_df = _engineer(math_df)
    else:
        baseline_df = _engineer(por_df)

    # --- Merge with collected data ---
    collected_path = os.path.join(DATA_DIR, "collected_data.csv")
    if os.path.exists(collected_path):
        collected_df = pd.read_csv(collected_path)
        # Update old column names if coming from old collected_data format
        rename_map = {
            "previous_sem_cgpa": "period_2_score",
            "previous_to_previous_sem_cgpa": "period_1_score",
            "attendance_percentage": "absences_inverse",
        }
        collected_df = collected_df.rename(columns=rename_map)
        if "target" in collected_df.columns and set(FEATURES).issubset(collected_df.columns):
            combined = pd.concat([baseline_df, collected_df[FEATURES + ["target"]]], ignore_index=True)
            print(f"  Added {len(collected_df)} collected records. Total: {len(combined)}")
            return combined

    return baseline_df


def _eval_model(model, X_train_s, X_test_s, y_train, y_test, name):
    """Return a dict of metrics for a fitted model."""
    y_pred = model.predict(X_test_s)
    try:
        y_prob = model.predict_proba(X_test_s)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    cv = cross_val_score(model, X_train_s, y_train,
                         cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                         scoring="f1_macro")
    return {
        "Model":          name,
        "Accuracy":       round(accuracy_score(y_test, y_pred), 4),
        "F1_Macro":       round(f1_score(y_test, y_pred, average="macro"), 4),
        "F1_Fail":        round(f1_score(y_test, y_pred, pos_label=0, average="binary"), 4),
        "Recall_Fail":    round(recall_score(y_test, y_pred, pos_label=0, zero_division=0), 4),
        "Precision_Fail": round(precision_score(y_test, y_pred, pos_label=0, zero_division=0), 4),
        "AUC_ROC":        round(auc, 4) if not np.isnan(auc) else "N/A",
        "CV_F1_Mean":     round(cv.mean(), 4),
    }


def train():
    print("=" * 60)
    print("Student Intelligence Platform — Retraining Pipeline v3.0")
    print("=" * 60)

    df  = load_and_merge()
    X   = df[FEATURES]
    y   = df["target"]

    print(f"\nClass distribution: PASS={y.sum()} ({y.mean()*100:.1f}%)  FAIL={(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")

    # Stratified split — consistent with notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train)
    X_test_s    = scaler.transform(X_test)

    # --- Model definitions ---
    models = {
        "Dummy (Majority)":   DummyClassifier(strategy="most_frequent", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes":         GaussianNB(),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4,
            random_state=42, eval_metric="logloss", verbosity=0
        )

    # --- Train, evaluate, save ---
    results     = []
    best_acc    = 0.0
    best_name   = ""

    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        metrics = _eval_model(model, X_train_s, X_test_s, y_train, y_test, name)
        results.append(metrics)

        fname = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model.pkl"
        joblib.dump(model, os.path.join(OUT_DIR, fname))

        print(f"  {name:<22} Acc={metrics['Accuracy']:.4f}  F1={metrics['F1_Macro']:.4f}  "
              f"RecallFail={metrics['Recall_Fail']:.4f}  AUC={metrics['AUC_ROC']}")

        if name != "Dummy (Majority)" and metrics["Accuracy"] > best_acc:
            best_acc  = metrics["Accuracy"]
            best_name = name

    # --- Save artifacts ---
    metrics_df = pd.DataFrame(results)
    joblib.dump(models[best_name], os.path.join(OUT_DIR, "best_model.pkl"))
    joblib.dump(scaler,            os.path.join(OUT_DIR, "scaler.pkl"))
    joblib.dump(FEATURES,          os.path.join(OUT_DIR, "selected_features.pkl"))
    joblib.dump(metrics_df,        os.path.join(OUT_DIR, "model_comparison.pkl"))

    print(f"\nBest model (excluding Dummy): {best_name} — Accuracy={best_acc:.4f}")
    print("All artifacts saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    train()
