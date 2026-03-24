"""
Student Intelligence Platform — v2.1 (Production Ready)
Refactored: 6 features, 9 models, Data Collection, File Upload, Retraining.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import time
import subprocess

try:
    import xgboost  # noqa: F401
except ImportError:
    pass

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="Student Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============= CONSTANTS =============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
COLLECTED_DATA_PATH = os.path.join(DATA_DIR, "collected_data.csv")

FEATURES = [
    "previous_sem_cgpa",
    "previous_to_previous_sem_cgpa",
    "number_of_backlogs",
    "attendance_percentage",
    "studytime",
    "goout",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes":         "naive_bayes_model.pkl",
    "SVM":                 "svm_model.pkl",
    "Decision Tree":       "decision_tree_model.pkl",
    "Random Forest":       "random_forest_model.pkl",
    "Gradient Boosting":   "gradient_boosting_model.pkl",
    "XGBoost":             "xgboost_model.pkl",
    "AdaBoost":            "adaboost_model.pkl",
    "KNN":                 "knn_model.pkl",
}

# ============= THEME =============
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

def reset_prediction():
    st.session_state.prediction_made = False

if st.session_state.theme == "dark":
    COLORS = {
        "bg": "#0B0E14", "card": "#1A1F2E", "text": "#00D9FF",
        "text_sec": "#A0A9B8", "accent": "#00D9FF",
        "success": "#39FF14", "warning": "#FFD700", "danger": "#FF006E",
        "grid_border": "rgba(0, 217, 255, 0.1)",
    }
    PLOT_TEMPLATE = "plotly_dark"
    BTN_TEXT = "#000000"
else:
    COLORS = {
        "bg": "#FFFFFF", "card": "#F8FAFC", "text": "#0F172A",
        "text_sec": "#64748B", "accent": "#0066CC",
        "success": "#22C55E", "warning": "#EAB308", "danger": "#EF4444",
        "grid_border": "rgba(15, 23, 42, 0.1)",
    }
    PLOT_TEMPLATE = "plotly_white"
    BTN_TEXT = "#FFFFFF"

# ============= CSS =============
st.markdown(f"""
<style>
.stApp {{ background-color: {COLORS['bg']}; transition: background-color 0.4s; }}
* {{ font-family: 'Inter', system-ui, sans-serif; }}
h1, h2, h3, h4 {{ color: {COLORS['text']} !important; font-weight: 800; letter-spacing: -0.02em; }}
p, label, .stMarkdown {{ color: {COLORS['text_sec']} !important; }}
.bento-card {{
    background: {COLORS['card']}; border: 1px solid {COLORS['grid_border']};
    border-radius: 20px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}}
.stSelectbox > div > div {{
    background-color: {COLORS['card']} !important; border: 1px solid {COLORS['accent']} !important;
    color: {COLORS['text']} !important; border-radius: 12px;
}}
.stButton > button {{
    background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['accent']}dd);
    color: {BTN_TEXT} !important; border: none; border-radius: 12px;
    padding: 0.8rem 2rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
}}
.stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 10px 30px rgba(0,217,255,0.3); }}
[data-testid="stMetricValue"] {{ color: {COLORS['text']} !important; font-size: 2.5rem !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; background: {COLORS['card']}; padding: 8px; border-radius: 16px; }}
.stTabs [aria-selected="true"] {{ background-color: {COLORS['accent']} !important; color: {BTN_TEXT} !important; font-weight: 600; }}
#MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}} header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ============= LOAD MODELS =============
@st.cache_resource
def load_all():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "selected_features.pkl"))
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    try:
        metrics_df = joblib.load(os.path.join(MODEL_DIR, "model_comparison.pkl"))
    except Exception:
        metrics_df = pd.DataFrame()
    return models, scaler, features, metrics_df

try:
    models_dict, scaler, feature_cols, metrics_df = load_all()
    if not models_dict:
        st.error(f"❌ No model files found in: `{MODEL_DIR}`")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ============= HELPERS =============
def predict_student(input_dict, model):
    inp = pd.DataFrame(0, index=[0], columns=feature_cols)
    for col in feature_cols:
        if col in input_dict:
            inp[col] = input_dict[col]
    inp_scaled = scaler.transform(inp)
    pred = model.predict(inp_scaled)[0]
    proba = model.predict_proba(inp_scaled)[0]
    return pred, proba

def save_new_data(df):
    if os.path.exists(COLLECTED_DATA_PATH):
        existing_df = pd.read_csv(COLLECTED_DATA_PATH)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(COLLECTED_DATA_PATH, index=False)

# ============= HEADER =============
col_h1, col_h2 = st.columns([8, 1])
with col_h1:
    st.markdown("<h1>🧠 Student Intelligence Platform</h1>", unsafe_allow_html=True)
with col_h2:
    btn_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
    if st.button(btn_label):
        toggle_theme()
        st.rerun()

st.markdown("---")

# ============= TABS =============
tab_predict, tab_leaderboard, tab_collection, tab_retrain = st.tabs([
    "🔮 PREDICT", "🏆 LEADERBOARD", "📥 DATA COLLECTION", "🔁 RETRAINING"
])

# ============= TAB 1: PREDICT =============
with tab_predict:
    col_form, col_result = st.columns([4, 6], gap="large")
    with col_form:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Student Input Form")
        semester = st.selectbox("Current Semester", [2,3,4,5,6,7,8], index=2, on_change=reset_prediction)
        
        c1, c2 = st.columns(2)
        with c1: prev_cgpa = st.number_input(f"Sem {semester-1} CGPA", 0.0, 10.0, 6.5, 0.1, on_change=reset_prediction)
        with c2: prev_prev_cgpa = st.number_input(f"Sem {semester-2} CGPA", 0.0, 10.0, 6.0, 0.1, on_change=reset_prediction)
        
        ac1, ac2 = st.columns(2)
        with ac1: backlogs = st.number_input("Backlogs/KTs", 0, 10, 0, on_change=reset_prediction)
        with ac2: attendance = st.slider("Attendance %", 0, 100, 75, on_change=reset_prediction)
        
        lf1, lf2 = st.columns(2)
        with lf1: studytime = st.select_slider("Study/Week", [1,2,3,4], 2, format_func=lambda x: {1:"<2h", 2:"2-5h", 3:"5-10h", 4:">10h"}[x], on_change=reset_prediction)
        with lf2: goout = st.select_slider("Socializing", [1,2,3,4,5], 3, format_func=lambda x: {1:"Rarely", 2:"Low", 3:"Medium", 4:"Often", 5:"Very Often"}[x], on_change=reset_prediction)

        selected_model = st.selectbox("Prediction Engine", ["All Models (Consensus)"] + list(models_dict.keys()), on_change=reset_prediction)

        if st.button("🚀 RUN PREDICTION", use_container_width=True):
            input_data = {"previous_sem_cgpa": prev_cgpa, "previous_to_previous_sem_cgpa": prev_prev_cgpa, "number_of_backlogs": backlogs, "attendance_percentage": attendance, "studytime": studytime, "goout": goout}
            if selected_model == "All Models (Consensus)":
                votes, probas = [], []
                for m_name, mdl in models_dict.items():
                    p, pb = predict_student(input_data, mdl)
                    votes.append(p); probas.append(pb)
                st.session_state.update({"prediction_made":True, "pred":int(np.round(np.mean(votes))), "proba":np.mean(probas, axis=0), "model_name":"Consensus (9 Models)", "input_data":input_data, "votes":votes})
            else:
                p, pb = predict_student(input_data, models_dict[selected_model])
                st.session_state.update({"prediction_made":True, "pred":p, "proba":pb, "model_name":selected_model, "input_data":input_data, "votes":None})
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if st.session_state.get("prediction_made"):
            pred, proba, m_name = st.session_state["pred"], st.session_state["proba"], st.session_state["model_name"]
            is_pass = pred == 1
            res_col = COLORS["success"] if is_pass else COLORS["danger"]
            val_txt = "✅ PASS" if is_pass else "❌ FAIL"
            conf = (proba[1] if is_pass else proba[0]) * 100

            st.markdown(f'<div class="bento-card" style="text-align:center; border: 2px solid {res_col};">', unsafe_allow_html=True)
            st.markdown(f'<h1 style="color:{res_col}; font-size:3rem;">{val_txt}</h1>', unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = conf, domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence", 'font': {'size': 16, 'color': COLORS['text_sec']}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS['text_sec']},
                    'bar': {'color': res_col},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2, 'bordercolor': COLORS['grid_border'],
                    'steps': [{'range': [0, 50], 'color': 'rgba(255,0,0,0.1)'}, {'range': [50, 100], 'color': 'rgba(0,255,0,0.1)'}],
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': COLORS['text'], 'family': "Inter"}, height=250, margin=dict(l=40, r=40, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f'<p style="color:{COLORS["text_sec"]}; font-size:0.8rem;">Engine: {m_name}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="bento-card" style="text-align:center; padding:100px 20px;"><h3>🔮 Ready to Predict</h3><p>Fill in the form and click <b>🚀 RUN PREDICTION</b></p></div>', unsafe_allow_html=True)

# ============= TAB 2: LEADERBOARD =============
with tab_leaderboard:
    st.markdown("### 🏆 Model Comparison")
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True)
        fig_lb = px.bar(metrics_df, x="Accuracy", y="Model", orientation='h', title="Model Accuracy", color="Accuracy", color_continuous_scale="Viridis", template=PLOT_TEMPLATE)
        st.plotly_chart(fig_lb, use_container_width=True)
    else: st.warning("No metrics found. Run retraining first.")

# ============= TAB 3: DATA COLLECTION =============
with tab_collection:
    st.markdown("### 📥 Expand Our Intelligence")
    st.info("You can contribute new student data to make the system smarter. Data is stored locally for future retraining.")
    
    col_g, col_f = st.columns(2)
    with col_g:
        st.markdown(f'<div class="bento-card"><h4>🔗 Google Form</h4><p>Use our official form for manual collection:</p><a href="https://rohit-student-marks.streamlit.app/" target="_blank"><button style="width:100%; height:40px; background:{COLORS["accent"]}; border:none; border-radius:10px; cursor:pointer; color:{BTN_TEXT}; font-weight:bold;">OPEN GOOGLE FORM</button></a></div>', unsafe_allow_html=True)
    
    with col_f:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("#### 📁 File Upload")
        st.info("Upload CSV/Excel with exactly these columns: `previous_sem_cgpa`, `previous_to_previous_sem_cgpa`, `number_of_backlogs`, `attendance_percentage`, `studytime`, `goout`, `target` (0 or 1).")
        uploaded_file = st.file_uploader("Contribute Data", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'): df_up = pd.read_csv(uploaded_file)
                else: df_up = pd.read_excel(uploaded_file)
                
                # Validation
                missing = [c for c in FEATURES + ["target"] if c not in df_up.columns]
                if missing: st.error(f"Missing columns: {missing}")
                else:
                    save_new_data(df_up[FEATURES + ["target"]])
                    st.success(f"✅ Successfully added {len(df_up)} new student records!")
            except Exception as e: st.error(f"Upload error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# ============= TAB 4: RETRAINING =============
with tab_retrain:
    st.markdown("### 🔁 Manual Retraining Pipeline")
    st.warning("⚠️ Retraining will merge the baseline dataset with all collected data. This process is irreversible once started.")
    
    if os.path.exists(COLLECTED_DATA_PATH):
        c_df = pd.read_csv(COLLECTED_DATA_PATH)
        st.metric("New Records Found", len(c_df))
    else: st.info("No new records found yet.")

    if st.button("🏗️ TRIGGER RETRAINING"):
        with st.status("🛠️ Retraining pipeline started...", expanded=True) as status:
            try:
                result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    status.update(label="✅ Retraining complete!", state="complete")
                    st.success("The system has been updated with the latest data. Please refresh the page to load the new models.")
                    st.toast("Retraining Successful!")
                else:
                    status.update(label="❌ Retraining failed", state="error")
                    st.error(result.stderr)
            except Exception as e: st.error(f"Error: {e}")

st.markdown(f'<p style="text-align:center; color:{COLORS["text_sec"]}; font-size:0.8rem; margin-top:20px;">Hosted at: rohit-student-marks.streamlit.app</p>', unsafe_allow_html=True)
