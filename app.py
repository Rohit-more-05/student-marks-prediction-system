"""
Student Intelligence Platform — v3.0 (Honest & Hardened)
Fixes applied over v2.3:
  - Feature names updated to honest labels (period scores, not 'CGPA')
  - Password moved to st.secrets (no hardcoded credentials)
  - Admin login attempts counter — locks after 5 failures
  - Backlogs capped at 3 (matches training distribution {0,1,2,3})
  - Consensus voting uses scipy.stats.mode (not fragile np.mean rounding)
  - Both P(PASS) and P(FAIL) displayed — not just winning class
  - File upload validates value ranges before saving
  - Retraining uses importlib (not subprocess shell call)
  - Radar chart includes all 6 features with honest normalization
  - Leaderboard shows F1_Macro, Recall_Fail, AUC alongside Accuracy
  - Disclaimer banner added on prediction tab
  - Removed decorative semester selectbox
  - Feedback form link fixed
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import importlib
import sys
from scipy.stats import mode as scipy_mode

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
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR           = os.path.join(BASE_DIR, "outputs")
DATA_DIR            = os.path.join(BASE_DIR, "data")
COLLECTED_DATA_PATH = os.path.join(DATA_DIR, "collected_data.csv")

# Honest feature names — these are period scores from the same academic year, not semester CGPA
FEATURES = [
    "period_2_score",       # G2/2.0 — 2nd period score of the course (0–10 scale)
    "period_1_score",       # G1/2.0 — 1st period score of the course (0–10 scale)
    "number_of_backlogs",   # failures column, capped at 3 per UCI data definition
    "absences_inverse",     # (75 - absences) / 75 * 100 — attendance proxy
    "studytime",            # 1=<2h, 2=2-5h, 3=5-10h, 4=>10h weekly
    "goout",                # 1=very low … 5=very high go-out frequency
]

# Feature value ranges (for upload validation and UI)
FEATURE_RANGES = {
    "period_2_score":      (0.0, 10.0),
    "period_1_score":      (0.0, 10.0),
    "number_of_backlogs":  (0,   3),
    "absences_inverse":    (0.0, 100.0),
    "studytime":           (1,   4),
    "goout":               (1,   5),
    "target":              (0,   1),
}

MODEL_FILES = {
    "Logistic Regression":  "logistic_regression_model.pkl",
    "Naive Bayes":          "naive_bayes_model.pkl",
    "SVM":                  "svm_model.pkl",
    "Decision Tree":        "decision_tree_model.pkl",
    "Random Forest":        "random_forest_model.pkl",
    "Gradient Boosting":    "gradient_boosting_model.pkl",
    "XGBoost":              "xgboost_model.pkl",
    "AdaBoost":             "adaboost_model.pkl",
    "KNN":                  "knn_model.pkl",
    "Dummy (Majority)":     "dummy_(majority)_model.pkl",
}

# ============= SESSION STATE =============
defaults = {
    "theme":           "dark",
    "prediction_made": False,
    "admin_attempts":  0,
    "admin_locked":    False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def toggle_theme():    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
def reset_prediction(): st.session_state.prediction_made = False

# ============= THEME COLORS =============
if st.session_state.theme == "dark":
    COLORS = {
        "bg": "#0B0E14", "card": "#1A1F2E", "text": "#00D9FF",
        "text_sec": "#A0A9B8", "accent": "#00D9FF", "success": "#39FF14",
        "warning": "#FFD700", "danger": "#FF006E", "grid_border": "rgba(0,217,255,0.1)"
    }
    PLOT_TEMPLATE = "plotly_dark"; BTN_TEXT = "#000000"
else:
    COLORS = {
        "bg": "#FFFFFF", "card": "#F8FAFC", "text": "#0F172A",
        "text_sec": "#64748B", "accent": "#0066CC", "success": "#22C55E",
        "warning": "#EAB308", "danger": "#EF4444", "grid_border": "rgba(15,23,42,0.1)"
    }
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
.disclaimer-box {{ background: rgba(255,215,0,0.1); border: 1px solid {COLORS['warning']}; border-radius: 12px; padding: 12px 16px; margin-bottom: 16px; color: {COLORS['warning']} !important; font-size: 0.85rem; }}
#MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}} header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ============= LOAD MODELS =============
@st.cache_resource
def load_all():
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "selected_features.pkl"))
    models   = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Failed to load {name}: {e}")
    try:
        metrics_df = joblib.load(os.path.join(MODEL_DIR, "model_comparison.pkl"))
    except Exception:
        metrics_df = pd.DataFrame()
    return models, scaler, features, metrics_df

try:
    models_dict, scaler, feature_cols, metrics_df = load_all()
    if not models_dict:
        st.error("❌ Models not found. Run: python train_model.py"); st.stop()
except Exception as e:
    st.error(f"❌ Initialization Error: {e}"); st.stop()

# ============= HELPERS =============
def predict_student(input_dict, model):
    inp = pd.DataFrame(0.0, index=[0], columns=feature_cols)
    for col in feature_cols:
        if col in input_dict:
            inp[col] = input_dict[col]
    inp_scaled = scaler.transform(inp)
    pred  = model.predict(inp_scaled)[0]
    try:
        proba = model.predict_proba(inp_scaled)[0]
    except Exception:
        proba = np.array([1 - pred, pred])
    return pred, proba


def save_new_data(df: pd.DataFrame):
    if os.path.exists(COLLECTED_DATA_PATH):
        existing = pd.read_csv(COLLECTED_DATA_PATH)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(COLLECTED_DATA_PATH, index=False)


def validate_uploaded_data(df: pd.DataFrame):
    """Returns (is_valid: bool, error_message: str)."""
    for col in FEATURES + ["target"]:
        if col not in df.columns:
            return False, f"Missing column: '{col}'"
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must be numeric, got {df[col].dtype}"
        lo, hi = FEATURE_RANGES[col]
        if df[col].isnull().any():
            return False, f"Column '{col}' contains null values"
        if df[col].min() < lo or df[col].max() > hi:
            return False, (
                f"Column '{col}' has values outside [{lo}, {hi}]. "
                f"Found range: [{df[col].min():.2f}, {df[col].max():.2f}]"
            )
    return True, "OK"


def get_admin_password():
    """Retrieve admin password from st.secrets. Returns None if not configured."""
    try:
        return str(st.secrets["admin_password"])
    except Exception:
        return None

# ============= SIDEBAR =============
with st.sidebar:
    st.markdown("### 🎨 UI Customization")
    if st.button("☀️" if st.session_state.theme == "dark" else "🌙"):
        toggle_theme(); st.rerun()
    st.markdown("---")
    st.info("Admin features are password-protected in the 'ADMIN PANEL' tab.")

# ============= HEADER =============
st.markdown("<h1>🧠 Student Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown("---")

# ============= TABS =============
tabs = st.tabs(["🔮 PREDICT", "🏆 LEADERBOARD", "📊 ANALYTICS", "🔐 ADMIN PANEL"])

# ============= TAB 1: PREDICT =============
with tabs[0]:
    # Disclaimer banner
    st.markdown(
        '<div class="disclaimer-box">⚠️ <strong>Transparency Notice:</strong> This tool uses Period 1 & Period 2 scores '
        'from the <em>same academic course</em> as the prediction target — they are strong predictors but come from '
        'the same evaluation period as the outcome. Treat results as exploratory, not definitive.</div>',
        unsafe_allow_html=True
    )

    col_form, col_result = st.columns([4, 6], gap="large")
    with col_form:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Input Parameters")

        c1, c2 = st.columns(2)
        with c1:
            p2_score = st.number_input(
                "Period 2 Score (0–10)",
                min_value=0.0, max_value=10.0, value=6.5, step=0.1,
                help="G2/2 — the second in-term score for this course (0–10 scale)",
                on_change=reset_prediction
            )
        with c2:
            p1_score = st.number_input(
                "Period 1 Score (0–10)",
                min_value=0.0, max_value=10.0, value=6.0, step=0.1,
                help="G1/2 — the first in-term score for this course (0–10 scale)",
                on_change=reset_prediction
            )

        ac1, ac2 = st.columns(2)
        with ac1:
            b_logs = st.number_input(
                "Past Course Failures (0–3)",
                min_value=0, max_value=3, value=0,
                help="Number of prior course failures. Dataset range: 0–3 (3 means '3 or more')",
                on_change=reset_prediction
            )
        with ac2:
            attend = st.slider(
                "Attendance Proxy (0–100)",
                min_value=0, max_value=100, value=75,
                help="Higher = fewer absences. Based on school absence count (0–75 absences mapped to 100–0).",
                on_change=reset_prediction
            )

        lf1, lf2 = st.columns(2)
        with lf1:
            st_time = st.select_slider(
                "Study Hours / Week",
                options=[1, 2, 3, 4], value=2,
                format_func=lambda x: {1:"<2h", 2:"2-5h", 3:"5-10h", 4:">10h"}[x],
                on_change=reset_prediction
            )
        with lf2:
            g_out = st.select_slider(
                "Go-Out Frequency",
                options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda x: {1:"Very Low", 2:"Low", 3:"Medium", 4:"High", 5:"Very High"}[x],
                on_change=reset_prediction
            )

        # Model selector (filter out Dummy from user selection)
        available_models = [m for m in models_dict if m != "Dummy (Majority)"]
        if "XGBoost" in available_models and not XGB_AVAILABLE:
            available_models.remove("XGBoost")
            st.warning("⚠️ XGBoost library not available.")

        sel_mdl = st.selectbox(
            "Prediction Engine",
            ["All Models (Consensus)"] + available_models,
            on_change=reset_prediction
        )

        if st.button("🚀 RUN ANALYSIS", use_container_width=True):
            in_data = {
                "period_2_score":     p2_score,
                "period_1_score":     p1_score,
                "number_of_backlogs": b_logs,
                "absences_inverse":   attend,
                "studytime":          st_time,
                "goout":              g_out,
            }
            if sel_mdl == "All Models (Consensus)":
                votes, probas = [], []
                for m_name, mdl in models_dict.items():
                    if m_name == "Dummy (Majority)":    continue
                    if m_name == "XGBoost" and not XGB_AVAILABLE: continue
                    p, pb = predict_student(in_data, mdl)
                    votes.append(int(p))
                    probas.append(pb)
                # Majority vote via mode (handles even/odd counts correctly)
                mode_result = scipy_mode(votes, keepdims=True)
                consensus_pred = int(mode_result.mode[0])
                consensus_proba = np.mean(probas, axis=0)
                st.session_state.update({
                    "prediction_made": True, "pred": consensus_pred,
                    "proba": consensus_proba, "model_name": "Consensus",
                    "input_data": in_data
                })
            else:
                p, pb = predict_student(in_data, models_dict[sel_mdl])
                st.session_state.update({
                    "prediction_made": True, "pred": p,
                    "proba": pb, "model_name": sel_mdl, "input_data": in_data
                })
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if st.session_state.prediction_made:
            p    = st.session_state["pred"]
            pb   = st.session_state["proba"]
            m_n  = st.session_state["model_name"]
            res_c = COLORS["success"] if p == 1 else COLORS["danger"]

            st.markdown(
                f'<div class="bento-card" style="text-align:center; border:2px solid {res_c};">'
                f'<h1 style="color:{res_c}; font-size:3.5rem; margin-bottom:0;">{"PASS" if p==1 else "FAIL"}</h1>'
                f'<p style="color:{COLORS["text_sec"]}; margin-top:4px;">Validated by {m_n} Engine</p>',
                unsafe_allow_html=True
            )

            # Dual probability display — show BOTH classes
            col_fail, col_pass = st.columns(2)
            with col_fail:
                st.metric("P(FAIL)", f"{pb[0]*100:.1f}%")
            with col_pass:
                st.metric("P(PASS)", f"{pb[1]*100:.1f}%")

            # Confidence gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pb[p] * 100,
                number={"suffix": "%"},
                title={"text": f"Confidence for {'PASS' if p==1 else 'FAIL'}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": res_c},
                    "bgcolor": COLORS["card"],
                    "borderwidth": 2,
                    "bordercolor": COLORS["grid_border"],
                    "threshold": {
                        "line": {"color": COLORS["warning"], "width": 4},
                        "thickness": 0.75,
                        "value": 50
                    }
                }
            ))
            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": COLORS["text"]}, height=280,
                margin=dict(l=40, r=40, t=60, b=0), template=PLOT_TEMPLATE
            )
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="bento-card" style="text-align:center; padding:120px 20px;">'
                '<h3>🔮 System Ready</h3>'
                '<p>Enter period scores and parameters, then run analysis.</p></div>',
                unsafe_allow_html=True
            )

# ============= TAB 2: LEADERBOARD =============
with tabs[1]:
    st.markdown("### 🏆 Performance Leaderboard")
    if not metrics_df.empty:
        try:
            display_cols = [c for c in ["Model","Accuracy","F1_Macro","F1_Fail","Recall_Fail","AUC_ROC","CV_F1_Mean"]
                            if c in metrics_df.columns]
            leaderboard = metrics_df[display_cols].copy()

            # Coerce AUC_ROC to numeric — it may contain "N/A" strings from Dummy
            if "AUC_ROC" in leaderboard.columns:
                leaderboard["AUC_ROC"] = pd.to_numeric(leaderboard["AUC_ROC"], errors="coerce")

            sort_col = "F1_Macro" if "F1_Macro" in leaderboard.columns else "Accuracy"
            dummy_mask = leaderboard["Model"].str.contains("Dummy", case=False, na=False)
            non_dummy  = leaderboard[~dummy_mask].sort_values(sort_col, ascending=False)
            dummy_row  = leaderboard[dummy_mask]
            final_lb   = pd.concat([non_dummy, dummy_row], ignore_index=True).reset_index(drop=True)

            st.caption("📌 Dummy (Majority) row is the baseline floor — real models must beat it.")
            # Round only numeric columns to avoid crash on 'Model' string column
            num_cols = final_lb.select_dtypes(include="number").columns
            final_lb[num_cols] = final_lb[num_cols].round(4)
            st.dataframe(final_lb, use_container_width=True)

            chart_col = "F1_Macro" if "F1_Macro" in final_lb.columns else "Accuracy"
            chart_df  = final_lb.dropna(subset=[chart_col])
            fig_lb = px.bar(
                chart_df, x=chart_col, y="Model", orientation="h",
                color=chart_col, color_continuous_scale="Viridis",
                template=PLOT_TEMPLATE, title=f"{chart_col} by Model"
            )
            st.plotly_chart(fig_lb, use_container_width=True)
        except Exception as e:
            st.error(f"Leaderboard render error: {e}")
    else:
        st.warning("No performance metrics available. Run: python train_model.py")

# ============= TAB 3: ANALYTICS =============
with tabs[2]:
    st.markdown("### 📊 Student Analytics")
    if st.session_state.prediction_made:
        c_p, c_r = st.columns(2, gap="large")

        with c_p:
            st.markdown('<div class="bento-card"><h4>🕸️ Input Profile Radar</h4>', unsafe_allow_html=True)
            in_d = st.session_state["input_data"]

            def _norm(val, lo, hi):
                return max(0.0, min(100.0, (val - lo) / (hi - lo) * 100)) if hi != lo else 50.0

            cats = ["Period 2\nScore", "Period 1\nScore", "Attendance\nProxy",
                    "Study\nHours", "Social\nLife", "No\nFailures"]
            vals = [
                _norm(in_d.get("period_2_score", 5),     0, 10),
                _norm(in_d.get("period_1_score", 5),     0, 10),
                _norm(in_d.get("absences_inverse", 75),  0, 100),
                _norm(in_d.get("studytime", 2),          1, 4),
                _norm(6 - in_d.get("goout", 3),          1, 5),
                _norm(3 - in_d.get("number_of_backlogs", 0), 0, 3),
            ]

            radar = go.Figure(data=go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself",
                line=dict(color=COLORS["accent"], width=3),
                fillcolor="rgba(0,217,255,0.15)"
            ))
            radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor=COLORS["grid_border"]),
                    angularaxis=dict(gridcolor=COLORS["grid_border"])
                ),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": COLORS["text"]}, height=400,
                margin=dict(l=60, r=60, t=30, b=30), template=PLOT_TEMPLATE
            )
            st.plotly_chart(radar, use_container_width=True)
            st.caption("All axes normalized 0–100 within their training range. Social Life axis is inverted (higher = less socialising).")
            st.markdown("</div>", unsafe_allow_html=True)

        with c_r:
            st.markdown('<div class="bento-card"><h4>📋 Input Summary</h4>', unsafe_allow_html=True)
            p2   = in_d.get("period_2_score", 5)
            p1   = in_d.get("period_1_score", 5)
            att  = in_d.get("absences_inverse", 75)
            st_v = in_d.get("studytime", 2)
            go_v = in_d.get("goout", 3)
            bl_v = in_d.get("number_of_backlogs", 0)
            summary_data = {
                "Feature":     ["Period 2 Score", "Period 1 Score", "Attendance Proxy",
                                "Study Hours", "Go-Out Freq.", "Course Failures"],
                "Your Value":  [f"{p2:.1f}/10", f"{p1:.1f}/10", f"{att:.0f}%",
                                {1:"<2h",2:"2-5h",3:"5-10h",4:">10h"}.get(st_v, str(st_v)),
                                {1:"Very Low",2:"Low",3:"Medium",4:"High",5:"Very High"}.get(go_v, str(go_v)),
                                str(int(bl_v))],
                "Normalized":  [f"{_norm(p2,0,10):.0f}%", f"{_norm(p1,0,10):.0f}%",
                                f"{att:.0f}%", f"{_norm(st_v,1,4):.0f}%",
                                f"{_norm(6-go_v,1,5):.0f}%", f"{_norm(3-bl_v,0,3):.0f}%"],
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("ℹ️ Run a prediction to see the input profile analysis.")

# ============= TAB 4: ADMIN PANEL =============
with tabs[3]:
    st.markdown("### 🛠️ Restricted Administration")

    if st.session_state.admin_locked:
        st.error("🔒 Admin panel locked after too many failed attempts. Restart the app to reset.")
    else:
        stored_pw  = get_admin_password()
        admin_auth = st.text_input("Enter Admin Password to Unlock Actions", type="password", key="admin_pw_input")

        if stored_pw is None:
            st.warning("⚙️ Admin password not configured. Add `admin_password = 'your-password'` to `.streamlit/secrets.toml`.")
        elif admin_auth == "":
            st.info("Enter the administrative password to access data management tools.")
        elif admin_auth == stored_pw:
            st.session_state.admin_attempts = 0
            st.success("✅ Secure Access Granted")

            # --- File Upload ---
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            st.markdown("#### 📁 Secure Bulk Import")
            st.caption(
                "Upload CSV/XLSX with columns: "
                + ", ".join(f"`{f}`" for f in FEATURES)
                + ", `target` (1=Pass, 0=Fail)"
            )
            up_f = st.file_uploader("Upload CSV/XLSX for Model Retraining", type=["csv", "xlsx"])
            if up_f:
                if up_f.size > 10 * 1024 * 1024:
                    st.error("❌ File too large. Maximum size is 10 MB.")
                else:
                    try:
                        df_up = pd.read_csv(up_f) if up_f.name.endswith(".csv") else pd.read_excel(up_f)
                        is_valid, err_msg = validate_uploaded_data(df_up)
                        if not is_valid:
                            st.error(f"❌ Validation Failed: {err_msg}")
                        else:
                            save_new_data(df_up[FEATURES + ["target"]])
                            st.success(f"✅ Successfully appended {len(df_up)} records.")
                    except Exception as e:
                        st.error(f"❌ Processing Error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

            # --- Retraining ---
            st.markdown("---")
            st.markdown("#### 🔁 Retraining Engine")
            st.caption("Retrains all models using the baseline dataset + any uploaded data.")
            if st.button("🏗️ INITIATE RETRAINING PIPELINE", use_container_width=True):
                with st.status("🛠️ Retraining all models...") as status:
                    try:
                        import train_model as _tm
                        importlib.reload(_tm)
                        _tm.train()
                        status.update(label="✅ Retraining Complete! Refreshing...", state="complete")
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Retraining failed: {e}")
        else:
            st.session_state.admin_attempts += 1
            remaining = max(0, 5 - st.session_state.admin_attempts)
            if st.session_state.admin_attempts >= 5:
                st.session_state.admin_locked = True
                st.error("🔒 Too many failed attempts. Admin panel locked.")
            else:
                st.error(f"❌ Incorrect Password. {remaining} attempt(s) remaining.")


# ============= FOOTER =============
st.markdown(
    f'<p style="text-align:center; color:{COLORS["text_sec"]}; font-size:0.8rem; margin-top:30px;">'
    f"Platform Version 3.0 | Data: UCI Student Performance (Math + Portuguese) | "
    f"<em>Predictions are exploratory tools, not academic verdicts.</em></p>",
    unsafe_allow_html=True
)
