"""
Helper: Quick predictions using the trained 6-feature model.
Run AFTER executing the notebook to generate outputs/.

Usage:
    python predict.py
"""

import joblib
import pandas as pd
import os

FEATURES = [
    'previous_sem_cgpa',
    'previous_to_previous_sem_cgpa',
    'number_of_backlogs',
    'attendance_percentage',
    'studytime',
    'goout'
]


def load_model():
    """Load saved model, scaler, features from outputs/."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    model    = joblib.load(os.path.join(base, 'best_model.pkl'))
    scaler   = joblib.load(os.path.join(base, 'scaler.pkl'))
    features = joblib.load(os.path.join(base, 'selected_features.pkl'))
    return model, scaler, features


def predict(cgpa_prev, cgpa_prev_prev, backlogs, attendance, studytime, goout,
            model=None, scaler=None):
    """Predict Pass/Fail for one student."""
    if model is None or scaler is None:
        model, scaler, _ = load_model()

    inp = pd.DataFrame(
        [[cgpa_prev, cgpa_prev_prev, backlogs, attendance, studytime, goout]],
        columns=FEATURES
    )
    inp_scaled = scaler.transform(inp)
    pred  = model.predict(inp_scaled)[0]
    proba = model.predict_proba(inp_scaled)[0]
    return pred, proba


if __name__ == '__main__':
    print("=" * 50)
    print("Student Pass/Fail Predictor (6 Features)")
    print("=" * 50)

    try:
        model, scaler, features = load_model()
        print(f"\n✅ Model loaded! Features: {features}")

        examples = [
            ("Strong student",  8.5, 8.0, 0, 92, 3, 2),
            ("Average student", 6.0, 5.5, 1, 75, 2, 3),
            ("At-risk student", 3.5, 4.0, 3, 55, 1, 4),
        ]

        for label, *args in examples:
            pred, proba = predict(*args, model=model, scaler=scaler)
            status = "✅ PASS" if pred == 1 else "❌ FAIL"
            print(f"\n{label}: {status}  (Fail={proba[0]*100:.1f}%, Pass={proba[1]*100:.1f}%)")

    except FileNotFoundError:
        print("\n⚠️ Model files not found. Run the notebook first!")
