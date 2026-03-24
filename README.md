# 🧠 Student Intelligence Platform (v2.1)

A complete ML-powered system to predict student academic outcomes (**PASS/FAIL**) tailored for Indian college students.

👉 **Live App**: [rohit-student-marks.streamlit.app](https://rohit-student-marks.streamlit.app/)

---

## 🚀 Overview
This system uses a streamlined **6-feature** pipeline and a comparison suite of **9 Machine Learning models** to provide highly accurate predictions. It features a full ML lifecycle, including real-time predictions, data collection via file uploads/Google Forms, and a manual retraining pipeline.

### 🎯 Key Features
- **Smart Prediction**: Real-time Pass/Fail analysis with confidence scoring.
- **Dynamic Inputs**: Adapts to your current semester (Sem 2–8).
- **Consensus Voting**: Aggregates predictions from 9 different algorithms for maximum reliability.
- **Data Collection**: Integrated file upload (.csv, .xlsx) for appending new student records.
- **Retraining Pipeline**: One-click manual retraining to update models with new data.

---

## 📊 ML Model Suite
The system evaluates student performance using:
1. Logistic Regression
2. Naive Bayes
3. SVM
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. AdaBoost
9. KNN

---

## 🛠️ Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rohit-more-05/student-marks-prediction-system.git
   cd student-marks-prediction-system
   ```

2. **Setup virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

## 🔁 Data Collection & Retraining

### 📥 Collecting New Data
- **Via App**: Use the **"📥 DATA COLLECTION"** tab in the Streamlit app to upload student records.
- **Format**: Uploaded files must contain the following columns:
  - `previous_sem_cgpa`, `previous_to_previous_sem_cgpa`, `number_of_backlogs`, `attendance_percentage`, `studytime`, `goout`, `target` (1 for Pass, 0 for Fail).

### ⚙️ Retraining
Manual retraining can be triggered from the **"🔁 RETRAINING"** tab or via terminal:
```bash
python train_model.py
```
This script will merge the original baseline data with any new data in `data/collected_data.csv` and save the updated models to `outputs/`.

---

## 📂 Project Structure
```
├── app.py                # Main Streamlit Dashboard
├── train_model.py        # Retraining Pipeline
├── predict.py            # Terminal Prediction Helper
├── notebooks/            # Original Experiment Notebooks
├── data/                 # Baseline & Collected Datasets
│   ├── student-mat.csv
│   └── student-por.csv
└── outputs/              # Saved Models & Scalers
```

---
**Developed by Rohit More**
