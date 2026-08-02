import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------------------------------------------
# 1. PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinAnalytics | Churn Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. ADVANCED STYLING & CUSTOM CSS (MATCHING ATTACHED DESIGN)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global App Background - Matches light grey-blue background in screenshot */
    .stApp {
        background-color: #EBF1F5;
        color: #1E293B;
    }

    /* Sidebar Styling - Matches dark navy sidebar in screenshot */
    section[data-testid="stSidebar"] {
        background-color: #102238 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* Global Typography Fixes for Light Background */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #0F172A !important;
    }
    
    /* Header Banner Styling */
    .banner-container {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    .banner-img {
        width: 100%;
        height: 200px;
        object-fit: cover;
        filter: brightness(0.4) contrast(1.1);
    }
    .banner-text {
        position: absolute;
        top: 50%;
        left: 5%;
        transform: translateY(-50%);
    }
    .banner-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
        background: linear-gradient(90deg, #10B981 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .banner-subtitle {
        font-size: 1.1rem;
        color: #E2E8F0 !important;
        margin-top: 0.5rem;
    }

    /* Light Theme Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748B !important;
        font-weight: 600;
    }

    /* Profile Card Styling */
    .profile-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #CBD5E1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. DATA & ML PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('Churn_Modelling.csv')
    return df

@st.cache_resource
def train_churn_model(df):
    # Preprocessing
    feature_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    X = df[feature_cols].copy()
    y = df['Exited']

    # One-Hot Encoding
    X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)
    
    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns.tolist(), acc

df = load_data()
model, feature_names, accuracy = train_churn_model(df)

# -----------------------------------------------------------------------------
# 4. TOP HERO BANNER (WITH IMAGE)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="banner-container">
    <img src="https://images.unsplash.com/photo-1551836022-d5d88e9218df?q=80&w=1600&auto=format&fit=crop" class="banner-img" alt="Header Banner">
    <div class="banner-text">
        <h1 class="banner-title">Customer Churn Prediction</h1>
        <p class="banner-subtitle">Predict customer departure • Discover actionable customer retention insights</p>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1563986768609-322da13575f3?q=80&w=400&auto=format&fit=crop", use_container_width=True)
st.sidebar.header("🔍 Global Dashboard Filters")

selected_geo = st.sidebar.multiselect("Geography", options=df['Geography'].unique(), default=df['Geography'].unique())
selected_gender = st.sidebar.multiselect("Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
age_range = st.sidebar.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (18, 70))

filtered_df = df[
    (df['Geography'].isin(selected_geo)) &
    (df['Gender'].isin(selected_gender)) &
    (df['Age'].between(age_range[0], age_range[1]))
]

# -----------------------------------------------------------------------------
# 6. MAIN APPLICATION NAVIGATION TABS
# -----------------------------------------------------------------------------
tab_dash, tab_predict, tab_insights = st.tabs([
    "📊 Executive Dashboard", 
    "🎯 AI Customer Predictor", 
    "💡 Model Intelligence"
])

# =============================================================================
# TAB 1: EXECUTIVE DASHBOARD
# =============================================================================
with tab_dash:
    st.markdown("### 📈 Key Performance Indicators")
    
    total_cust = len(filtered_df)
    churn_count = filtered_df['Exited'].sum()
    churn_rate = (churn_count / total_cust * 100) if total_cust > 0 else 0
    avg_balance = filtered_df['Balance'].mean()
    active_rate = (filtered_df['IsActiveMember'].sum() / total_cust * 100) if total_cust > 0 else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value" style="color: #0284C7;">{total_cust:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churn Rate</div>
            <div class="metric-value" style="color: #DC2626;">{churn_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Account Balance</div>
            <div class="metric-value" style="color: #059669;">${avg_balance:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Member Ratio</div>
            <div class="metric-value" style="color: #D97706;">{active_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Grid
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("🌍 Churn Analysis by Geography")
        geo_summary = filtered_df.groupby(['Geography', 'Exited']).size().reset_index(name='Count')
        geo_summary['Status'] = geo_summary['Exited'].map({0: 'Retained', 1: 'Churned'})
        
        fig_geo = px.bar(
            geo_summary, x='Geography', y='Count', color='Status',
            barmode='group',
            color_discrete_map={'Retained': '#0284C7', 'Churned': '#DC2626'},
            template="plotly_white"
        )
        fig_geo.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_geo, use_container_width=True)

    with col_chart2:
        st.subheader("👥 Age Distribution by Churn Status")
        fig_age = px.histogram(
            filtered_df, x="Age", color="Exited",
            marginal="box", barmode="overlay",
            color_discrete_map={0: '#059669', 1: '#DC2626'},
            labels={'Exited': 'Churned (1)'},
            template="plotly_white"
        )
        fig_age.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_age, use_container_width=True)

    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        st.subheader("📦 Churn vs Number of Products")
        prod_data = filtered_df.groupby('NumOfProducts')['Exited'].mean().reset_index()
        prod_data['Churn Probability'] = prod_data['Exited'] * 100
        
        fig_prod = px.bar(
            prod_data, x='NumOfProducts', y='Churn Probability',
            color='Churn Probability',
            color_continuous_scale='Reds',
            text_auto='.1f%',
            template="plotly_white"
        )
        fig_prod.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_chart4:
        st.subheader("💰 Balance vs Credit Score Matrix")
        fig_scat = px.scatter(
            filtered_df.sample(min(800, len(filtered_df))), 
            x="CreditScore", y="Balance", color="Exited",
            color_discrete_map={0: '#0284C7', 1: '#DC2626'},
            opacity=0.7, template="plotly_white"
        )
        fig_scat.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scat, use_container_width=True)

# =============================================================================
# TAB 2: AI CUSTOMER PREDICTOR
# =============================================================================
with tab_predict:
    st.markdown("### 🔮 Single Customer Risk Evaluator")
    st.write("Input profile values below to compute real-time prediction using the trained Machine Learning model.")

    col_input, col_output = st.columns([1.2, 1])

    with col_input:
        with st.form("prediction_form"):
            st.subheader("👤 Profile Parameters")
            
            c1, c2 = st.columns(2)
            with c1:
                input_geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
                input_gender = st.selectbox("Gender", ["Male", "Female"])
                input_age = st.number_input("Age", min_value=18, max_value=100, value=38)
                input_score = st.slider("Credit Score", 300, 850, 650)
                input_tenure = st.slider("Tenure (Years)", 0, 10, 5)

            with c2:
                input_balance = st.number_input("Account Balance ($)", min_value=0.0, value=60000.0, step=1000.0)
                input_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
                input_has_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)
                input_active = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)
                input_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=75000.0, step=1000.0)

            btn_predict = st.form_submit_button("🔥 Run Churn Prediction")

    with col_output:
        st.subheader("🎯 Evaluation Outcome")
        if btn_predict:
            # Prepare Input Dataframe matching one-hot encoding columns
            input_dict = {
                'CreditScore': [input_score],
                'Age': [input_age],
                'Tenure': [input_tenure],
                'Balance': [input_balance],
                'NumOfProducts': [input_products],
                'HasCrCard': [input_has_card],
                'IsActiveMember': [input_active],
                'EstimatedSalary': [input_salary],
                'Geography_Germany': [1 if input_geo == 'Germany' else 0],
                'Geography_Spain': [1 if input_geo == 'Spain' else 0],
                'Gender_Male': [1 if input_gender == 'Male' else 0]
            }
            
            # Align columns
            input_df = pd.DataFrame(input_dict)
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]

            # Model Prediction
            prob = model.predict_proba(input_df)[0][1]
            prob_pct = prob * 100

            # Radial Gauge Visual
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={'suffix': "%", 'font': {'size': 36, 'color': '#0F172A'}},
                title={'text': "Calculated Risk Score", 'font': {'size': 18, 'color': '#0F172A'}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#0F172A"},
                    'steps': [
                        {'range': [0, 30], 'color': "#10B981"},
                        {'range': [30, 60], 'color': "#F59E0B"},
                        {'range': [60, 100], 'color': "#EF4444"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), template="plotly_white", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)

            if prob > 0.5:
                st.error("🚨 **High Risk of Customer Churn!**")
                st.info("💡 **Recommended Action:** Priority outreach required. Offer promotional interest rate or personal account manager.")
            else:
                st.success("✅ **Low Churn Probability**")
                st.info("💡 **Recommended Action:** Standard retention sequence; cross-sell secondary products.")
        else:
            st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="Fill details and click Predict.")

# =============================================================================
# TAB 3: MODEL INTELLIGENCE
# =============================================================================
with tab_insights:
    st.markdown("### 🤖 ML Model Performance & Driver Analysis")
    
    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #0284C7;">
            <div class="metric-label">Model Accuracy Rate</div>
            <div class="metric-value" style="color: #0284C7;">{accuracy * 100:.2f}%</div>
            <p style="color: #64748B !important; font-size: 0.85rem; margin-top: 5px;">Trained on 8,000 historical records using Random Forest Classifier</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📌 Key Drivers of Customer Churn")
        
        # Feature Importance
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
        
        fig_fi = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Viridis',
            template="plotly_white"
        )
        fig_fi.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_m2:
        st.subheader("📷 Visual Summary & Insights")
        st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=800&auto=format&fit=crop", use_container_width=True)
        st.markdown("""
        #### Top Findings:
        1. **Age Factor**: Older customers (ages 40-60) exhibit significantly higher churn rates compared to younger segments.
        2. **Product Engagement**: Customers with 3 or 4 products have a near 80%+ churn rate—indicating potential cross-selling friction or product dissatisfaction.
        3. **Geographic Variance**: Customers based in **Germany** churn at almost double the rate of those in France or Spain.
        """)
