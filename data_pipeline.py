"""
Student Performance Prediction - Data Pipeline (Multi-Model Edition)
Trains and compares 5 models: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy import stats
from logger_system import log_wrapper

# ============= CREATE DIRECTORIES =============
print("[SETUP] Creating project directories...")
os.makedirs('models', exist_ok=True)
print("[SUCCESS] Directory structure ready")

# ============= DATA LOADING =============
print("\n[PHASE] Starting Data Loading...")

try:
    math_df = pd.read_csv('data/student-mat.csv', sep=';')
    por_df = pd.read_csv('data/student-por.csv', sep=';')
    print(f"[SUCCESS] Loaded Math ({len(math_df)}) & Portuguese ({len(por_df)}) datasets")
except FileNotFoundError:
    print("[ERROR] CSV files not found in data/ folder")
    exit(1)

# Combine datasets
df = pd.concat([math_df, por_df], ignore_index=True)
print(f"[SUCCESS] Combined dataset: {len(df)} students")

# ============= CREATE TARGET VARIABLE =============
print("\n[PHASE] Creating Target Variable...")

@log_wrapper
def create_target(row):
    """3-Level Risk Classification based on final grade G3"""

    G3 = row['G3']
    if G3 < 10:
        return 0  # High Risk (FAIL)
    elif G3 < 14:
        return 1  # Medium Risk
    else:
        return 2  # Low Risk (PASS)

df['target'] = df.apply(create_target, axis=1)

# ============= DATA CLEANING =============
print("\n[PHASE] Starting Data Cleaning...")

# Fill missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Remove outliers (Z-score)
numeric_cols_no_target = [col for col in numeric_cols if col != 'target']
z_scores = np.abs(stats.zscore(df[numeric_cols_no_target]))
df_clean = df[(z_scores < 3).all(axis=1)]
print(f"[SUCCESS] Cleaned data: {len(df_clean)} students (removed {len(df) - len(df_clean)} outliers)")

# ============= FEATURE ENGINEERING =============
print("\n[PHASE] Starting Feature Engineering...")

df_clean['academic_risk'] = (df_clean['failures'] * 3 + (5 - df_clean['studytime']) * 2)
df_clean['study_efficiency'] = df_clean['G1'] / (df_clean['studytime'] + 1)
df_clean['parent_edu_avg'] = (df_clean['Medu'] + df_clean['Fedu']) / 2
schools_sup = df_clean['schoolsup'].map({'yes': 1, 'no': 0})
fam_sup = df_clean['famsup'].map({'yes': 1, 'no': 0})
df_clean['support_index'] = (schools_sup + fam_sup) / 2
df_clean['grade_improvement'] = df_clean['G2'] - df_clean['G1']

print("[SUCCESS] Engineered 5 new features")

# ============= CATEGORICAL ENCODING =============
print("\n[PHASE] Starting Categorical Encoding...")

categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                        'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                        'nursery', 'higher', 'internet', 'romantic']

# One-hot encode
df_encoded = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True)

# Important: Save columns used for training to ensure matching during inference
feature_columns = df_encoded.drop(['G1', 'G2', 'G3', 'target'], axis=1).columns.tolist()
joblib.dump(feature_columns, 'models/feature_columns.pkl')

print(f"[SUCCESS] Encoded features. Total inputs: {len(feature_columns)}")

# ============= TRAIN-TEST SPLIT =============
X = df_encoded[feature_columns]
y = df_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'models/scaler.pkl')
print("[SUCCESS] Data split and scaled")

# ============= MULTI-MODEL TRAINING =============
print("\n[PHASE] Starting Multi-Model Training & Comparison...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = []

for name, model in models.items():
    print(f"   Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Save model
    filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, filename)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })

# Save comparison results
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
joblib.dump(results_df, 'models/model_comparison.pkl')

print("\n" + "="*60)
print("🏆 MODEL LEADERBOARD (Ranked by Accuracy)")
print("="*60)
print(results_df.to_string(index=False, float_format="%.4f"))
print("="*60)

print(f"\n[SUMMARY] Best Model: {results_df.iloc[0]['Model']} ({results_df.iloc[0]['Accuracy']*100:.2f}%)")
print("[SUCCESS] All models saved to models/ folder")
