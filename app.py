"""
Student Intelligence Platform — v2.3 (Action-Hardened Security)
Refactored: 6 features, 9 models, Radar Chart, Action-based Password Protection.
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

# ============= XGBOOST STABILIZATION =============
XGB_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
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
ADMIN_PASSWORD = "qwertyui"

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

# ============= SESSION STATE =============
if "theme" not in st.session_state: st.session_state.theme = "dark"
if "prediction_made" not in st.session_state: st.session_state.prediction_made = False

def toggle_theme(): st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
def reset_prediction(): st.session_state.prediction_made = False

if st.session_state.theme == "dark":
    COLORS = {"bg": "#0B0E14", "card": "#1A1F2E", "text": "#00D9FF", "text_sec": "#A0A9B8", "accent": "#00D9FF", "success": "#39FF14", "warning": "#FFD700", "danger": "#FF006E", "grid_border": "rgba(0, 217, 255, 0.1)"}
    PLOT_TEMPLATE = "plotly_dark"; BTN_TEXT = "#000000"
else:
    COLORS = {"bg": "#FFFFFF", "card": "#F8FAFC", "text": "#0F172A", "text_sec": "#64748B", "accent": "#0066CC", "success": "#22C55E", "warning": "#EAB308", "danger": "#EF4444", "grid_border": "rgba(15, 23, 42, 0.1)"}
    PLOT_TEMPLATE = "plotly_white"; BTN_TEXT = "#FFFFFF"

# ============= CSS =============
st.markdown(f"""
<style>
.stApp {{ background-color: {COLORS['bg']}; transition: background-color 0.4s; }}
h1, h2, h3, h4 {{ color: {COLORS['text']} !important; font-weight: 800; }}
p, label, .stMarkdown {{ color: {COLORS['text_sec']} !important; }}
.bento-card {{ background: {COLORS['card']}; border: 1px solid {COLORS['grid_border']}; border-radius: 20px; padding: 24px; }}
.stButton > button {{ background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['accent']}dd); color: {BTN_TEXT} !important; border-radius: 12px; font-weight: 700; }}
.stTabs [aria-selected="true"] {{ background-color: {COLORS['accent']} !important; color: {BTN_TEXT} !important; }}
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
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Failed to load {name}: {e}")
    try: metrics_df = joblib.load(os.path.join(MODEL_DIR, "model_comparison.pkl"))
    except: metrics_df = pd.DataFrame()
    return models, scaler, features, metrics_df

try:
    models_dict, scaler, feature_cols, metrics_df = load_all()
    if not models_dict: st.error("❌ Models not found. Run training first."); st.stop()
except Exception as e: st.error(f"❌ Initialization Error: {e}"); st.stop()

# ============= HELPERS =============
def predict_student(input_dict, model):
    inp = pd.DataFrame(0, index=[0], columns=feature_cols)
    for col in feature_cols:
        if col in input_dict: inp[col] = input_dict[col]
    inp_scaled = scaler.transform(inp)
    return model.predict(inp_scaled)[0], model.predict_proba(inp_scaled)[0]

def save_new_data(df):
    if os.path.exists(COLLECTED_DATA_PATH):
        existing_df = pd.read_csv(COLLECTED_DATA_PATH)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(COLLECTED_DATA_PATH, index=False)

# ============= SIDEBAR =============
with st.sidebar:
    st.markdown("### 🎨 UI Customization")
    if st.button("☀️" if st.session_state.theme == "dark" else "🌙"):
        toggle_theme(); st.rerun()
    st.markdown("---")
    st.info("Admin features are now gated by password in the 'ADMIN PANEL' tab.")

# ============= HEADER =============
st.markdown("<h1>🧠 Student Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown("---")

# ============= TABS =============
tabs = st.tabs(["🔮 PREDICT", "🏆 LEADERBOARD", "📊 ANALYTICS", "🔐 ADMIN PANEL"])

# ============= TAB 1: PREDICT =============
with tabs[0]:
    col_form, col_result = st.columns([4, 6], gap="large")
    with col_form:
        st.markdown('<div class="bento-card">### 🎛️ Input Parameters', unsafe_allow_html=True)
        semester = st.selectbox("Current Semester", [2,3,4,5,6,7,8], index=2, on_change=reset_prediction)
        c1, c2 = st.columns(2)
        with c1: p_cgpa = st.number_input(f"Sem {semester-1} CGPA", 0.0, 10.0, 6.5, 0.1, on_change=reset_prediction)
        with c2: pp_cgpa = st.number_input(f"Sem {semester-2} CGPA", 0.0, 10.0, 6.0, 0.1, on_change=reset_prediction)
        ac1, ac2 = st.columns(2)
        with ac1: b_logs = st.number_input("Backlogs/KTs", 0, 10, 0, on_change=reset_prediction)
        with ac2: attend = st.slider("Attendance %", 0, 100, 75, on_change=reset_prediction)
        lf1, lf2 = st.columns(2)
        with lf1: st_time = st.select_slider("Study Intensity", [1,2,3,4], 2, format_func=lambda x: {1:"Low (<2h)", 2:"Medium (2-5h)", 3:"High (5-10h)", 4:"Insane (>10h)"}[x], on_change=reset_prediction)
        with lf2: g_out = st.select_slider("Socializing", [1,2,3,4,5], 3, format_func=lambda x: {1:"None", 2:"Low", 3:"Mid", 4:"High", 5:"Party Animal"}[x], on_change=reset_prediction)
        
        # XGBoost Warning if library missing
        mdl_list = list(models_dict.keys())
        if "XGBoost" in mdl_list and not XGB_AVAILABLE:
            st.warning("⚠️ XGBoost library not installed in this environment. Using other models.")
            mdl_list.remove("XGBoost")
            
        sel_mdl = st.selectbox("Prediction Engine", ["All Models (Consensus)"] + mdl_list, on_change=reset_prediction)
        
        if st.button("🚀 RUN ANALYSIS", use_container_width=True):
            in_data = {"previous_sem_cgpa": p_cgpa, "previous_to_previous_sem_cgpa": pp_cgpa, "number_of_backlogs": b_logs, "attendance_percentage": attend, "studytime": st_time, "goout": g_out}
            if sel_mdl == "All Models (Consensus)":
                votes, probas = [], []
                for m_name, mdl in models_dict.items():
                    if m_name == "XGBoost" and not XGB_AVAILABLE: continue
                    p, pb = predict_student(in_data, mdl); votes.append(p); probas.append(pb)
                st.session_state.update({"prediction_made":True, "pred":int(np.round(np.mean(votes))), "proba":np.mean(probas, axis=0), "model_name":"Consensus", "input_data":in_data})
            else:
                p, pb = predict_student(in_data, models_dict[sel_mdl])
                st.session_state.update({"prediction_made":True, "pred":p, "proba":pb, "model_name":sel_mdl, "input_data":in_data})
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if st.session_state.prediction_made:
            p, pb, m_n = st.session_state["pred"], st.session_state["proba"], st.session_state["model_name"]
            res_c = COLORS["success"] if p == 1 else COLORS["danger"]
            st.markdown(f'<div class="bento-card" style="text-align:center; border:2px solid {res_c};"><h1 style="color:{res_c}; font-size:3.5rem; margin-bottom:0;">{"PASS" if p==1 else "FAIL"}</h1>', unsafe_allow_html=True)
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=pb[1 if p==1 else 0]*100, number={'suffix': '%'}, title={'text': "Prediction Confidence"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': res_c}, 'bgcolor': COLORS['card'], 'borderwidth': 2, 'bordercolor': COLORS['grid_border']}))
            gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': COLORS['text']}, height=280, margin=dict(l=40, r=40, t=60, b=0), template=PLOT_TEMPLATE)
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown(f'<p style="color:{COLORS["text_sec"]}; margin-top:-20px;">Validated by {m_n} Engine</p></div>', unsafe_allow_html=True)
        else: st.markdown('<div class="bento-card" style="text-align:center; padding:120px 20px;"><h3>🔮 System Ready</h3><p>Adjust parameters and trigger analysis.</p></div>', unsafe_allow_html=True)

# ============= TAB 2: LEADERBOARD =============
with tabs[1]:
    st.markdown("### 🏆 Performance Leaderboard")
    if not metrics_df.empty:
        st.dataframe(metrics_df.sort_values("Accuracy", ascending=False).reset_index(drop=True), use_container_width=True)
        fig_lb = px.bar(metrics_df, x="Accuracy", y="Model", orientation='h', color="Accuracy", color_continuous_scale="Viridis", template=PLOT_TEMPLATE)
        st.plotly_chart(fig_lb, use_container_width=True)
    else: st.warning("No performance metrics available.")

# ============= TAB 3: ANALYTICS =============
with tabs[2]:
    st.markdown("### 📊 Student Analytics")
    if st.session_state.prediction_made:
        c_p, c_r = st.columns(2, gap="large")
        with c_p:
            st.markdown('<div class="bento-card"><h4>🕸️ Behavioral Radar Chart</h4>', unsafe_allow_html=True)
            in_d = st.session_state["input_data"]
            cats = ["CGPA (Prev)", "CGPA (P-Prev)", "Attendance", "Study Habit", "Social Life"]
            vals = [in_d["previous_sem_cgpa"]*10, in_d["previous_to_previous_sem_cgpa"]*10, in_d["attendance_percentage"], in_d["studytime"]*25, in_d["goout"]*20]
            radar = go.Figure(data=go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself', line=dict(color=COLORS['accent'], width=3), fillcolor="rgba(0, 217, 255, 0.2)"))
            radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor=COLORS['grid_border']), angularaxis=dict(gridcolor=COLORS['grid_border'])), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': COLORS['text']}, height=400, margin=dict(l=60, r=60, t=30, b=30), template=PLOT_TEMPLATE)
            st.plotly_chart(radar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c_r:
            st.markdown(f'<div class="bento-card"><h4>🔗 Data Contribution</h4><p>Users can contribute their academic data below.</p><a href="https://rohit-student-marks.streamlit.app/" target="_blank"><button style="width:100%; height:45px; background:{COLORS["accent"]}; border:none; border-radius:12px; cursor:pointer; font-weight:800; color:{BTN_TEXT}; font-size:1rem;">OPEN FEEDBACK FORM</button></a><p style="margin-top:20px; font-size:0.8rem;">Manual contributions help improve future model accuracy.</p></div>', unsafe_allow_html=True)
    else: st.info("ℹ️ Run a prediction to see the behavioral radar analysis.")

# ============= TAB 4: ADMIN PANEL (INSTANT PROTECTION) =============
with tabs[3]:
    st.markdown("### 🛠️ Restricted Administration")
    
    # Instant Password Gate
    admin_auth = st.text_input("Enter Admin Password to Unlock Actions", type="password")
    
    if admin_auth == ADMIN_PASSWORD:
        st.success("✅ Secure Access Granted")
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("#### 📁 Secure Bulk Import")
        up_f = st.file_uploader("Upload CSV/XLSX for Model Retraining", type=["csv", "xlsx"])
        if up_f:
            try:
                df_up = pd.read_csv(up_f) if up_f.name.endswith('.csv') else pd.read_excel(up_f)
                missing = [c for c in FEATURES + ["target"] if c not in df_up.columns]
                if missing: st.error(f"❌ Missing required columns: {missing}")
                else: 
                    save_new_data(df_up[FEATURES + ["target"]])
                    st.success(f"Successfully appended {len(df_up)} records to the dataset.")
            except Exception as e: st.error(f"❌ Processing Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### 🔁 Retraining Engine")
        if st.button("🏗️ INITIATE RETRAINING PIPELINE", use_container_width=True):
            with st.status("🛠️ Retraining all 9 models...") as status:
                res = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
                if res.returncode == 0:
                    status.update(label="✅ Retraining Complete! Refreshing artifacts.", state="complete")
                    st.cache_resource.clear()
                    st.rerun()
                else: 
                    st.error(f"Execution Failed:\n{res.stderr}")
    elif admin_auth == "":
        st.info("Please enter the administrative password to access data management tools.")
    else:
        st.error("❌ Incorrect Password. Access Denied.")

st.markdown(f'<p style="text-align:center; color:{COLORS["text_sec"]}; font-size:0.8rem; margin-top:30px;">Platform Version 2.3 | Secured for Production</p>', unsafe_allow_html=True)
