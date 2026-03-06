import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from scipy import stats
from logger_system import log_wrapper

@log_wrapper
def create_target(row):
    G3 = row['G3']
    if G3 < 10: return 0
    elif G3 < 14: return 1
    else: return 2

@log_wrapper
def run_generation():
    print("[DATA-GEN] Starting test data generation...")
    # 1. Load Data
    try:
        math_df = pd.read_csv('data/student-mat.csv', sep=';')
        por_df = pd.read_csv('data/student-por.csv', sep=';')
        df = pd.concat([math_df, por_df], ignore_index=True)
        print(f"[SUCCESS] Loaded {len(df)} records")
    except FileNotFoundError:
        print("[ERROR] Data files not found!")
        return

    # 2. Target Variable
    df['target'] = df.apply(create_target, axis=1)

    # 3. Clean Data (Exact replica of pipeline)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    numeric_cols_no_target = [col for col in numeric_cols if col != 'target']
    z_scores = np.abs(stats.zscore(df[numeric_cols_no_target]))
    df_clean = df[(z_scores < 3).all(axis=1)].copy()

    # 4. Feature Engineering
    df_clean['academic_risk'] = (df_clean['failures'] * 3 + (5 - df_clean['studytime']) * 2)
    df_clean['study_efficiency'] = df_clean['G1'] / (df_clean['studytime'] + 1)
    df_clean['parent_edu_avg'] = (df_clean['Medu'] + df_clean['Fedu']) / 2
    schoolsup_mapped = df_clean['schoolsup'].map({'yes': 1, 'no': 0})
    famsup_mapped = df_clean['famsup'].map({'yes': 1, 'no': 0})
    df_clean['support_index'] = (schoolsup_mapped + famsup_mapped) / 2
    df_clean['grade_improvement'] = df_clean['G2'] - df_clean['G1']

    # 5. Encoding
    categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                            'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                            'nursery', 'higher', 'internet', 'romantic']
    df_encoded = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True)

    # 6. Alignment
    feature_columns = joblib.load('models/feature_columns.pkl')
    X = df_encoded[feature_columns]
    y = df_encoded['target']

    # 7. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 8. Save
    os.makedirs('data', exist_ok=True)
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    print(f"[SUCCESS] Saved X_test.csv ({len(X_test)} rows) and y_test.csv to data/")

if __name__ == "__main__":
    run_generation()
