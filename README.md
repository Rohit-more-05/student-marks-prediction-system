




























# 🧠 Student Intelligence Platform (v3.0)

A Streamlit ML dashboard that predicts student academic outcomes (**PASS/FAIL**) using the UCI Student Performance dataset (Math + Portuguese courses).

👉 **Live App**: [rohit-student-marks.streamlit.app](https://rohit-student-marks.streamlit.app/)

---

## ⚠️ Data Transparency

This system uses the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) from two secondary schools in Portugal. The UCI documentation explicitly warns:

> *"G3 has a strong correlation with G2 and G1. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful."*

**The two period score features (G1/G2) come from the same academic course and year as the final grade (G3).** This makes them strong predictors but not indicators of independent prior academic history. The system is transparent about this limitation and labels features accordingly.

**This dataset is NOT from Indian college students.** It reflects Portuguese secondary school students (ages 15–22, 0–20 grading scale). Use results as exploratory tools, not academic verdicts.

---

## 🚀 What Was Fixed in v3.0

| Issue | Fix Applied |
|---|---|
| Hardcoded password in source code | Password moved to `.streamlit/secrets.toml` (git-ignored) |
| Backlogs input allowed 0–10 (training max is 3) | Capped to 0–3 matching UCI data definition |
| Consensus voting used `np.mean` rounding | Replaced with `scipy.stats.mode` majority vote |
| Only winning class probability shown | Both P(PASS) and P(FAIL) now displayed |
| Duplicate students (Math + Portuguese) | Deduplicated using UCI-recommended merge key |
| Only Accuracy in leaderboard | Added F1_Macro, Recall_Fail, AUC_ROC, Dummy baseline |
| Stratify disabled in train_model.py | Stratified split re-enabled |
| Deprecated XGBoost `use_label_encoder` param | Removed (dropped in xgboost≥2.0) |
| File upload had no value validation | Added per-column range + type + null checks |
| `subprocess.run` for retraining (RCE risk) | Replaced with `importlib`-based call |
| Feedback link pointed to same app | Replaced with correct placeholder |
| Notebook never executed, missing EDA | Full rewrite: EDA, heatmap, ablation, ROC curves |
| Feature names labelled as "CGPA" | Honestly labelled as period scores |
| Radar chart missing backlogs axis | All 6 features now shown with normalized scaling |

---

## 📊 ML Model Suite

The leaderboard evaluates all models including a **Dummy classifier baseline** — any real model must beat it to be useful.

1. Logistic Regression
2. Naive Bayes
3. SVM
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. AdaBoost
9. KNN
10. **Dummy (Majority)** — baseline floor

---

## 🛠️ Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rohit-more-05/student-marks-prediction-system.git
   cd student-marks-prediction-system-v2
   ```

2. **Setup virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   # source .venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure admin password**:
   Create `.streamlit/secrets.toml` (already git-ignored):
   ```toml
   admin_password = "your-secure-password-here"
   ```

5. **Train models** (required before running the app):
   ```bash
   python train_model.py
   ```

6. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

## 🔁 Data Collection & Retraining

### Uploading New Data
Use the **ADMIN PANEL** tab with your configured password. Upload CSV/XLSX files with these columns:

| Column | Type | Range |
|---|---|---|
| `period_2_score` | float | 0.0 – 10.0 |
| `period_1_score` | float | 0.0 – 10.0 |
| `number_of_backlogs` | int | 0 – 3 |
| `absences_inverse` | float | 0.0 – 100.0 |
| `studytime` | int | 1 – 4 |
| `goout` | int | 1 – 5 |
| `target` | int | 0 (Fail) or 1 (Pass) |

### Retraining
From the Admin Panel or terminal:
```bash
python train_model.py
```

---

## 📂 Project Structure
```
├── app.py                          # Main Streamlit Dashboard (v3.0)
├── train_model.py                  # Retraining Pipeline (v3.0)
├── requirements.txt
├── notebooks/
│   └── feature_selection_experiment.ipynb   # Full research notebook
├── data/
│   ├── student-mat.csv             # UCI Math dataset (395 students)
│   ├── student-por.csv             # UCI Portuguese dataset (649 students)
│   └── collected_data.csv          # Collected data (appended by admin uploads)
├── outputs/                        # Saved model artifacts (git-ignored)
└── .streamlit/
    └── secrets.toml                # Admin password (git-ignored — create manually)
```

---

## 📋 Feature Descriptions

| Feature | Source Column | Description |
|---|---|---|
| `period_2_score` | G2 / 2.0 | 2nd period in-term score, scaled 0–10 |
| `period_1_score` | G1 / 2.0 | 1st period in-term score, scaled 0–10 |
| `number_of_backlogs` | failures | Prior course failures, clipped to 0–3 per UCI definition |
| `absences_inverse` | absences | (75 − absences) / 75 × 100 — higher means fewer absences |
| `studytime` | studytime | 1=<2h, 2=2–5h, 3=5–10h, 4=>10h weekly study |
| `goout` | goout | Going out frequency: 1 (very low) to 5 (very high) |

---

**Developed by Rohit More** | v3.0 — Honest & Hardened Edition
