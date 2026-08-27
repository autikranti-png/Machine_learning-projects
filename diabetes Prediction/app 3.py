import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# ============================================================

st.set_page_config(
    page_title="DiabetesAI - Medical Intelligence Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HISTORY_FILE = "prediction_history.csv"
DATASET_FILE = "diabetes.csv"

# Ensure History File Exists
if not os.path.exists(HISTORY_FILE):
    df_empty = pd.DataFrame(
        columns=[
            "Timestamp",
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Prediction",
            "Probability",
            "RiskLevel",
        ]
    )
    df_empty.to_csv(HISTORY_FILE, index=False)

# Hide Sidebar completely for full horizontal navigation experience
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 2. CUSTOM CSS & ADVANCED THEMING (HIGH VISIBILITY & CONTRAST)
# ============================================================

st.markdown(
    """
<style>
    /* Global Imports & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Lightened Overlay to Make Background Image Extremely Clear */
    .stApp {
        background: linear-gradient(rgba(241, 245, 249, 0.45), rgba(241, 245, 249, 0.55)),
                    url("https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: #0f172a;
    }

    /* Force all Streamlit native input labels to be high-contrast dark text */
    div[data-widget="number_input"] label,
    div[data-widget="selectbox"] label,
    div[data-widget="radio"] label,
    .stNumberInput label,
    label[data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        text-shadow: 0px 1px 2px rgba(255, 255, 255, 0.8) !important;
    }

    /* Glassmorphism Containers with High Contrast Text */
    .glass-card {
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(203, 213, 225, 0.9);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px -5px rgba(14, 165, 233, 0.2);
        border-color: rgba(14, 165, 233, 0.5);
    }

    /* Section Aesthetics */
    .section-overview { border-top: 5px solid #0284c7; }
    .section-prediction { border-top: 5px solid #0d9488; }
    .section-analytics { border-top: 5px solid #6366f1; }
    .section-history { border-top: 5px solid #8b5cf6; }
    .section-dataset { border-top: 5px solid #ec4899; }
    .section-architecture { border-top: 5px solid #f59e0b; }
    .section-clinical { border-top: 5px solid #10b981; }
    .section-about { border-top: 5px solid #64748b; }

    /* Metric KPI Cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .kpi-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin-bottom: 6px;
        font-weight: 700;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0284c7;
    }

    /* Risk Status Badges */
    .badge-low {
        background-color: #dcfce7;
        color: #14532d;
        border: 1px solid #86efac;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #7f1d1d;
        border: 1px solid #fca5a5;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 700;
        display: inline-block;
    }

    /* Header Banner */
    .header-container {
        padding: 24px 32px;
        background: linear-gradient(135deg, #0284c7 0%, #0d9488 100%);
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.4);
        color: #ffffff;
    }

    /* Streamlit Primary, Form Submit, & Download Button Customization */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(90deg, #0284c7 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 12px 28px !important;
        border: none !important;
        box-shadow: 0 4px 14px 0 rgba(2, 132, 199, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(90deg, #0369a1 0%, #0f766e 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px 0 rgba(2, 132, 199, 0.6) !important;
        transform: translateY(-1px) !important;
    }

    /* Target inner text for form submit, action, and download buttons */
    div[data-testid="stFormSubmitButton"] > button p,
    div[data-testid="stDownloadButton"] > button p,
    div.stButton > button p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Custom Navigation Radio Options (Horizontal Tab Fix for Contrast) */
    div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        justify-content: center !important;
        flex-wrap: wrap !important;
        gap: 10px !important;
        background: rgba(255, 255, 255, 0.95);
        padding: 12px 18px;
        border-radius: 14px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }

    div[data-testid="stRadio"] label {
        background: #ffffff;
        padding: 8px 18px;
        border-radius: 8px;
        border: 1px solid #94a3b8;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    div[data-testid="stRadio"] label p {
        color: #0f172a !important;
        font-weight: 700 !important;
        text-shadow: none !important;
    }

    div[data-testid="stRadio"] label:hover {
        background: #e2e8f0;
        border-color: #0284c7;
    }

    /* Disclaimer Footer Box */
    .disclaimer-box {
        background: rgba(255, 251, 235, 0.95);
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-top: 30px;
        font-size: 0.85rem;
        color: #78350f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 3. MODEL LOADING & CACHING
# ============================================================


@st.cache_resource
def load_ml_components():
    try:
        with open("diabetes_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("diabetes_features.pkl", "rb") as f:
            features = pickle.load(f)
        return model, features, None
    except Exception as e:
        return None, None, str(e)


model, features, load_error = load_ml_components()

if load_error:
    st.error(
        f"Critical Error Loading Model Assets: {load_error}\n\n"
        "Ensure `diabetes_model.pkl` and `diabetes_features.pkl` are located in the working directory."
    )
    st.stop()

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================


def get_risk_category(prob_percentage):
    if prob_percentage <= 30.0:
        return "Low Risk", "#16a34a", "🟢"
    elif prob_percentage <= 60.0:
        return "Moderate Risk", "#d97706", "🟡"
    elif prob_percentage <= 80.0:
        return "High Risk", "#ea580c", "🟠"
    else:
        return "Very High Risk", "#dc2626", "🔴"


def log_prediction(inputs_dict, prediction_val, probability_val, risk_level):
    row_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **inputs_dict,
        "Prediction": int(prediction_val),
        "Probability": round(float(probability_val), 2),
        "RiskLevel": risk_level,
    }
    df = pd.DataFrame([row_data])
    df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)


# ============================================================
# 5. TOP HEADER BANNER
# ============================================================

st.markdown(
    """
    <div class="header-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 style="color: white; margin: 0; font-size: 2rem; font-weight: 800;">
                    💉 DiabetesAI Medical Intelligence
                </h1>
                <p style="color: #f0f9ff; margin: 4px 0 0 0; font-size: 1rem; font-weight: 500;">
                    Machine Learning Predictive Analytics & Clinical Decision Support System
                </p>
            </div>
            <div style="text-align: right; background: rgba(255,255,255,0.25); padding: 8px 16px; border-radius: 8px;">
                <span style="font-size: 0.8rem; color: #ffffff; display: block; font-weight: 600;">ENGINE STATUS</span>
                <strong style="color: white; font-size: 0.95rem;">🟢 Model Online (8 Features)</strong>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 6. HORIZONTAL NAVIGATION BAR
# ============================================================

selected_page = st.radio(
    "",
    [
        "🏠 Overview",
        "🔮 Prediction Engine",
        "📊 Analytics Hub",
        "📜 Prediction History",
        "🗂 Dataset Explorer",
        "🤖 Model Architecture",
        "📚 Clinical Insights",
        "ℹ️ About",
    ],
    index=1,
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if selected_page == "🏠 Overview":
    st.markdown("### 🏠 Executive System Overview")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(
            """
            <div class="glass-card section-overview">
                <h3 style="color: #0284c7; margin-top:0;">🩸 Welcome to DiabetesAI Intelligence Platform</h3>
                <p style="color: #1e293b; line-height: 1.6; font-weight: 500;">
                    DiabetesAI is an advanced, healthcare-focused predictive platform built upon non-invasive physiological measurements. By utilizing modern decision tree logic trained on validated clinical datasets, the dashboard evaluates diabetes risk profiles in real time.
                </p>
                <div style="margin-top: 20px; display: flex; gap: 15px;">
                    <div style="flex: 1; background: #ffffff; padding: 12px; border-radius: 8px; border: 1px solid #cbd5e1;">
                        <strong style="color: #0284c7;">🩺 8 Feature Inputs</strong>
                        <p style="font-size: 0.8rem; color: #475569; margin: 4px 0 0 0; font-weight: 600;">Evaluates Glucose, BMI, Age, Insulin & Vitals</p>
                    </div>
                    <div style="flex: 1; background: #ffffff; padding: 12px; border-radius: 8px; border: 1px solid #cbd5e1;">
                        <strong style="color: #16a34a;">⚡ Deterministic Tree Path</strong>
                        <p style="font-size: 0.8rem; color: #475569; margin: 4px 0 0 0; font-weight: 600;">Non-blackbox rule progression logic</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.image(
            "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=800&q=80",
            caption="🔬 AI-Assisted Digital Diagnostics",
            use_container_width=True,
        )

    st.markdown("#### 🔬 Diagnostic Metric Scope")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-title">💉 Metabolic Baseline</div><div class="kpi-value">Glucose</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-title">⚖️ Body Composition</div><div class="kpi-value">BMI Index</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-title">🧪 Pancreatic Metric</div><div class="kpi-value">Insulin</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-title">🧬 Genetic Lineage</div><div class="kpi-value">Pedigree Fun.</div></div>',
            unsafe_allow_html=True,
        )

# ============================================================
# PAGE 2: PREDICTION ENGINE
# ============================================================
elif selected_page == "🔮 Prediction Engine":
    st.markdown("### 🔮 Patient Risk Calculator")

    validation_warnings = []

    with st.container():
        st.markdown(
            """
            <div class="glass-card section-prediction">
                <h4 style="color: #0d9488; margin-top:0; font-weight:700;">📋 Patient Clinical Metric Inputs</h4>
            """,
            unsafe_allow_html=True,
        )

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("<h5 style='color: #0f172a;'>👤 Demographics</h5>", unsafe_allow_html=True)
                pregnancies = st.number_input(
                    "Pregnancies 🤰",
                    min_value=0,
                    max_value=20,
                    value=1,
                    step=1,
                    help="Number of times pregnant",
                )
                age = st.number_input(
                    "Age (Years) 🎂",
                    min_value=1,
                    max_value=120,
                    value=33,
                    step=1,
                    help="Patient age in years",
                )

            with c2:
                st.markdown("<h5 style='color: #0f172a;'>💓 Hemodynamics</h5>", unsafe_allow_html=True)
                glucose = st.number_input(
                    "Glucose (mg/dL) 🍬",
                    min_value=0,
                    max_value=300,
                    value=117,
                    step=1,
                    help="Plasma glucose concentration (2 hours in an oral glucose tolerance test)",
                )
                blood_pressure = st.number_input(
                    "Blood Pressure (mmHg) 🩸",
                    min_value=0,
                    max_value=200,
                    value=72,
                    step=1,
                    help="Diastolic blood pressure",
                )
                bmi = st.number_input(
                    "BMI (kg/m²) ⚖️",
                    min_value=0.0,
                    max_value=70.0,
                    value=31.2,
                    step=0.1,
                    help="Body Mass Index",
                )

            with c3:
                st.markdown("<h5 style='color: #0f172a;'>🧪 Lab Measurements</h5>", unsafe_allow_html=True)
                skin_thickness = st.number_input(
                    "Skin Thickness (mm) 📏",
                    min_value=0,
                    max_value=100,
                    value=23,
                    step=1,
                    help="Triceps skin fold thickness",
                )
                insulin = st.number_input(
                    "Insulin (mu U/ml) 💉",
                    min_value=0,
                    max_value=900,
                    value=30,
                    step=1,
                    help="2-Hour serum insulin",
                )
                diabetes_pedigree = st.number_input(
                    "Pedigree Function 🧬",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.375,
                    step=0.001,
                    format="%.3f",
                    help="Diabetes pedigree score calculated from family history",
                )

            submit_predict = st.form_submit_button(
                "⚡ Execute Diagnostic Assessment", use_container_width=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    if glucose == 0:
        validation_warnings.append(
            "Glucose value registered as 0 mg/dL. In clinical settings, physiological glucose is non-zero."
        )
    if blood_pressure == 0:
        validation_warnings.append(
            "Diastolic Blood Pressure registered as 0 mmHg."
        )
    if bmi == 0.0:
        validation_warnings.append(
            "BMI registered as 0.0 kg/m². Ensure physiological dimensions are valid."
        )

    for warn in validation_warnings:
        st.warning(f"⚠️ **Validation Notice**: {warn}")

    if submit_predict:
        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age,
            ]],
            columns=features,
        )

        prediction_val = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        prob_no_diabetes = probabilities[0] * 100.0
        prob_diabetes = probabilities[1] * 100.0

        risk_label, risk_color, risk_icon = get_risk_category(prob_diabetes)

        input_dict = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age,
        }
        log_prediction(
            input_dict, prediction_val, prob_diabetes, risk_label
        )

        st.markdown("---")
        st.markdown("### 📊 Diagnostic Results & Assessment Summary")

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            status_title = (
                "DIABETES RISK DETECTED"
                if prediction_val == 1
                else "LOW RISK / NO DIABETES"
            )
            badge_class = "badge-high" if prediction_val == 1 else "badge-low"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Classification Output</div>
                    <div class="{badge_class}" style="margin-top:6px;">{status_title}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Diabetes Probability</div>
                    <div class="kpi-value" style="color:{risk_color};">{prob_diabetes:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Assessed Risk Tier</div>
                    <div class="kpi-value" style="color:{risk_color}; font-size:1.4rem;">{risk_icon} {risk_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k4:
            conf_val = max(prob_diabetes, prob_no_diabetes)
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Model Confidence</div>
                    <div class="kpi-value" style="color:#0284c7;">{conf_val:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        v_col1, v_col2 = st.columns([1, 1])

        with v_col1:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob_diabetes,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={
                        "text": "Risk Probability Meter (%)",
                        "font": {"color": "#0f172a", "size": 16},
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1,
                            "tickcolor": "#64748b",
                        },
                        "bar": {"color": risk_color},
                        "bgcolor": "rgba(255, 255, 255, 0.9)",
                        "borderwidth": 2,
                        "bordercolor": "#cbd5e1",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(34, 197, 94, 0.2)"},
                            {"range": [30, 60], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [60, 80], "color": "rgba(249, 115, 22, 0.2)"},
                            {"range": [80, 100], "color": "rgba(239, 68, 68, 0.2)"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#0f172a"},
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with v_col2:
            max_bounds = {
                "Glucose": 200,
                "BloodPressure": 120,
                "BMI": 50,
                "Insulin": 200,
                "Age": 80,
            }
            p_metrics = ["Glucose", "BloodPressure", "BMI", "Insulin", "Age"]
            p_vals = [
                min(glucose / max_bounds["Glucose"], 1.0),
                min(blood_pressure / max_bounds["BloodPressure"], 1.0),
                min(bmi / max_bounds["BMI"], 1.0),
                min(insulin / max_bounds["Insulin"], 1.0),
                min(age / max_bounds["Age"], 1.0),
            ]

            fig_radar = go.Figure(
                data=go.Scatterpolar(
                    r=p_vals,
                    theta=p_metrics,
                    fill="toself",
                    name="Patient Values",
                    line=dict(color="#0284c7"),
                )
            )
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, range=[0, 1], showticklabels=False
                    ),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                ),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0f172a"),
                title="Relative Metric Spectrum (Normalized)",
                height=300,
                margin=dict(l=40, r=40, t=50, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown(
            """
            <div class="glass-card section-prediction">
                <h4 style="color: #0d9488; margin-top:0;">🧠 Key Clinical Decision Factors</h4>
                <ul>
            """,
            unsafe_allow_html=True,
        )

        insights = []
        if glucose >= 140:
            insights.append(
                f"Elevated Plasma Glucose level ({glucose} mg/dL) significantly influences higher risk node traversal."
            )
        else:
            insights.append(
                f"Glucose measurement ({glucose} mg/dL) remains within nominal limits (<140 mg/dL)."
            )

        if bmi >= 30.0:
            insights.append(
                f"BMI metric ({bmi} kg/m²) indicates obesity classification, a primary weighting parameter in decision trees."
            )

        if age >= 45:
            insights.append(
                f"Age factor ({age} yrs) aligns with statistical higher-incidence clinical pathways."
            )

        for ins in insights:
            st.markdown(
                f"<li style='color: #1e293b; font-size:0.9rem; font-weight:500;'>{ins}</li>",
                unsafe_allow_html=True,
            )

        st.markdown("</ul></div>", unsafe_allow_html=True)

        report_df = pd.DataFrame([
            {
                "Parameter": "Patient Classification",
                "Value": (
                    "DIABETES POSITIVE"
                    if prediction_val == 1
                    else "NO DIABETES DETECTED"
                ),
            },
            {
                "Parameter": "Diabetes Probability",
                "Value": f"{prob_diabetes:.2f}%",
            },
            {"Parameter": "Assessed Risk Category", "Value": risk_label},
            {"Parameter": "Glucose", "Value": f"{glucose} mg/dL"},
            {"Parameter": "Blood Pressure", "Value": f"{blood_pressure} mmHg"},
            {"Parameter": "BMI", "Value": f"{bmi} kg/m²"},
            {"Parameter": "Insulin", "Value": f"{insulin} mu U/ml"},
            {"Parameter": "Age", "Value": f"{age} Years"},
            {
                "Parameter": "Pedigree Function",
                "Value": f"{diabetes_pedigree:.3f}",
            },
            {
                "Parameter": "Assessment Timestamp",
                "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        ])

        csv_report = report_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Patient Summary Report (CSV)",
            data=csv_report,
            file_name=f"DiabetesAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ============================================================
# PAGE 3: ANALYTICS HUB
# ============================================================
elif selected_page == "📊 Analytics Hub":
    st.markdown("### 📊 System Analytics & Insights")

    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
    else:
        df_hist = pd.DataFrame()

    if df_hist.empty:
        st.info(
            "ℹ️ No historical prediction records located. Execute assessments in the Prediction Engine to populate clinical analytics."
        )
    else:
        total_evals = len(df_hist)
        pos_evals = len(df_hist[df_hist["Prediction"] == 1])
        neg_evals = total_evals - pos_evals
        avg_prob = df_hist["Probability"].mean()

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">📋 Total Evaluations</div><div class="kpi-value">{total_evals}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">🔴 Positive Cases</div><div class="kpi-value" style="color:#dc2626;">{pos_evals}</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">🟢 Negative Cases</div><div class="kpi-value" style="color:#16a34a;">{neg_evals}</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-title">📈 Avg Risk Score</div><div class="kpi-value">{avg_prob:.1f}%</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        c_left, c_right = st.columns(2)

        with c_left:
            fig_pie = px.pie(
                df_hist,
                names="Prediction",
                title="Historical Classification Ratio",
                color="Prediction",
                color_discrete_map={0: "#16a34a", 1: "#dc2626"},
                hole=0.4,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#0f172a")
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_right:
            fig_bar = px.histogram(
                df_hist,
                x="RiskLevel",
                title="Evaluated Risk Tier Distribution",
                color="RiskLevel",
                color_discrete_map={
                    "Low Risk": "#16a34a",
                    "Moderate Risk": "#d97706",
                    "High Risk": "#ea580c",
                    "Very High Risk": "#dc2626",
                },
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0f172a"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# PAGE 4: PREDICTION HISTORY
# ============================================================
elif selected_page == "📜 Prediction History":
    st.markdown("### 📜 Prediction History & Logs")

    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
    else:
        df_hist = pd.DataFrame()

    if df_hist.empty:
        st.info("ℹ️ Prediction log history is currently empty.")
    else:
        st.markdown(
            "<div class='glass-card section-history'>Review and inspect past clinical assessment evaluations.</div>",
            unsafe_allow_html=True,
        )

        st.dataframe(
            df_hist,
            use_container_width=True,
            column_config={
                "Probability": st.column_config.NumberColumn(
                    "Risk Prob (%)", format="%.2f %%"
                ),
                "Timestamp": "Execution Time ⏰",
            },
        )

        csv_download = df_hist.to_csv(index=False)
        st.download_button(
            label="📥 Export Assessment Registry (CSV)",
            data=csv_download,
            file_name="diabetes_prediction_registry.csv",
            mime="text/csv",
        )

# ============================================================
# PAGE 5: DATASET EXPLORER
# ============================================================
elif selected_page == "🗂 Dataset Explorer":
    st.markdown("### 🗂 Dataset Explorer & EDA")

    if os.path.exists(DATASET_FILE):
        df_raw = pd.read_csv(DATASET_FILE)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Total Patient Records", df_raw.shape[0])
        with d2:
            st.metric("Feature Parameters", df_raw.shape[1] - 1)
        with d3:
            st.metric(
                "Positive Diagnoses",
                int(
                    df_raw["Outcome"].sum()
                    if "Outcome" in df_raw.columns
                    else 0
                ),
            )

        st.markdown("#### 📄 Dataset Sample")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.markdown("#### 📊 Metric Distribution Explorer")
        selected_feature = st.selectbox(
            "Select Feature Parameter to Inspect Distribution", features
        )

        fig_dist = px.histogram(
            df_raw,
            x=selected_feature,
            color="Outcome" if "Outcome" in df_raw.columns else None,
            marginal="box",
            title=f"Distribution Profile: {selected_feature}",
            color_discrete_map={0: "#0284c7", 1: "#dc2626"},
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f172a"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    else:
        st.warning(
            f"Dataset file `{DATASET_FILE}` was not found in the root directory. Place `diabetes.csv` to enable EDA features."
        )

# ============================================================
# PAGE 6: MODEL ARCHITECTURE
# ============================================================
elif selected_page == "🤖 Model Architecture":
    st.markdown("### 🤖 Decision Tree Model Specifications")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown(
            """
            <div class="glass-card section-architecture">
                <h4 style="color:#f59e0b; margin-top:0;">⚙️ Model Hyperparameters</h4>
                <table style="width:100%; color:#1e293b; font-size:0.9rem; font-weight: 500;">
                    <tr><td style="padding:6px 0;"><strong>Algorithm Type:</strong></td><td>Decision Tree Classifier</td></tr>
                    <tr><td style="padding:6px 0;"><strong>Max Depth:</strong></td><td>7</td></tr>
                    <tr><td style="padding:6px 0;"><strong>Min Samples Leaf:</strong></td><td>15</td></tr>
                    <tr><td style="padding:6px 0;"><strong>Min Samples Split:</strong></td><td>2</td></tr>
                    <tr><td style="padding:6px 0;"><strong>Target Outcome:</strong></td><td>Diabetes Binary Outcome (0/1)</td></tr>
                    <tr><td style="padding:6px 0;"><strong>Feature Count:</strong></td><td>8 Numerical Features</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """
            <div class="glass-card section-architecture">
                <h4 style="color:#f59e0b; margin-top:0;">📌 Feature Vector Ordering</h4>
                <ol style="color:#1e293b; font-size:0.9rem; padding-left: 20px; font-weight: 500;">
                    <li>Pregnancies 🤰</li>
                    <li>Glucose 🍬</li>
                    <li>BloodPressure 🩸</li>
                    <li>SkinThickness 📏</li>
                    <li>Insulin 💉</li>
                    <li>BMI ⚖️</li>
                    <li>DiabetesPedigreeFunction 🧬</li>
                    <li>Age 🎂</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# PAGE 7: CLINICAL INSIGHTS
# ============================================================
elif selected_page == "📚 Clinical Insights":
    st.markdown("### 📚 Clinical Reference & Guidance")

    st.markdown(
        """
        <div class="glass-card section-clinical">
            <h4 style="color: #10b981; margin-top:0;">🩸 Key Diagnostic Biomarkers</h4>
            <p style="color: #1e293b; font-size: 0.95rem; line-height:1.6; font-weight: 500;">
                <strong>🍬 Plasma Glucose:</strong> Measures circulating glucose. Readings >140 mg/dL 2-hours postprandial require medical verification.<br><br>
                <strong>⚖️ Body Mass Index (BMI):</strong> Proxy for body composition. BMI values over 30 kg/m² are associated with elevated insulin resistance.<br><br>
                <strong>💉 Insulin Levels:</strong> Key hormone for glucose homeostasis. Abnormal serum insulin reflects resistance or impaired secretion.<br><br>
                <strong>🧬 Diabetes Pedigree Score:</strong> Mathematical representation of genetic diabetes risk based on hereditary history.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# PAGE 8: ABOUT
# ============================================================
elif selected_page == "ℹ️ About":
    st.markdown("### ℹ️ About DiabetesAI Platform")

    st.markdown(
        """
        <div class="glass-card section-about">
            <h4 style="color:#64748b; margin-top:0;">💡 System Details & Technology</h4>
            <p style="color:#1e293b; font-size:0.95rem; font-weight: 500;">
                DiabetesAI provides interactive diagnostic support leveraging non-invasive parameters and machine learning classification tree structures.
            </p>
            <hr style="border-color: #cbd5e1;">
            <p style="color:#475569; font-size:0.85rem; font-weight: 600;">
                <strong>Tech Stack:</strong> Python, Streamlit, Scikit-Learn, Plotly, Pandas, NumPy
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# GLOBAL MEDICAL DISCLAIMER FOOTER
# ============================================================
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ MEDICAL DISCLAIMER:</strong> This application is intended strictly for portfolio, research, and informational demonstration purposes. It does not provide official medical diagnoses or substitute professional medical consultation.
    </div>
    """,
    unsafe_allow_html=True,
)