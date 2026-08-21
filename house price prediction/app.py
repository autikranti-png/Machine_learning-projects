import pandas as pd
import numpy as np
import streamlit as st
import pickle
import os
import datetime
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Light Dashboard Styling
st.markdown("""
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #f4f6f9;
        color: #2c3e50;
    }
    
    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2.2rem;
        border-radius: 12px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        color: #cbd5e1;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* Prediction Result Banner */
    .prediction-box {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    .prediction-price {
        font-size: 2.8rem;
        font-weight: 800;
    }

    /* Input Tabs Container Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff;
        padding: 8px 12px 0px 12px;
        border-radius: 10px 10px 0 0;
        border: 1px solid #e2e8f0;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0 0 10px 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
    }

    /* Section Headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #2563eb;
    }

    /* Hide Streamlit Brand Details */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. LOAD ML MODELS & ARTIFACTS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_artifacts():
    try:
        with open("house_price_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            dv = pickle.load(f)
        with open("encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open("features.pkl", "rb") as f:
            features = pickle.load(f)
        return model, dv, encoder, features
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None

model, dv, encoder, features = load_ml_artifacts()


# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & HISTORY MANAGEMENT
# -----------------------------------------------------------------------------
HISTORY_FILE = "prediction_history.csv"

def save_to_history(data_dict, pred_lakhs):
    """Appends prediction details to a persistent CSV file."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "Timestamp": now,
        "City": data_dict["City"].title(),
        "State": data_dict["State"].title(),
        "Property Type": data_dict["Property_Type_Raw"],
        "BHK": data_dict["BHK"],
        "Size (Sq.Ft)": data_dict["Size_in_SqFt"],
        "Predicted Price (Lakhs)": f"₹ {pred_lakhs:.2f} L"
    }
    df_new = pd.DataFrame([record])
    if os.path.exists(HISTORY_FILE):
        df_new.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(HISTORY_FILE, mode='w', header=True, index=False)

def load_history():
    """Loads prediction history from CSV file."""
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & INFO
# -----------------------------------------------------------------------------
st.sidebar.title("🏠 Navigation")
nav_choice = st.sidebar.radio("Go to", ["Prediction Dashboard", "About the System"])

st.sidebar.divider()
st.sidebar.markdown("""
### 📊 Model Details
- **Algorithm**: Decision Tree Regressor
- **Features**: 20 Inputs
- **Preprocessing**: DictVectorizer & Ordinal Encoding

### 🛠️ Tech Stack
- Python
- Scikit-Learn
- Pandas & NumPy
- Streamlit & Plotly
""")

st.sidebar.divider()
st.sidebar.info("💡 **Instructions**: Fill in the property specifications across tabs and click 'Predict House Price'.")


# -----------------------------------------------------------------------------
# 5. MAIN CONTENT - PAGE 1: PREDICTION DASHBOARD
# -----------------------------------------------------------------------------
if nav_choice == "Prediction Dashboard":

    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🏠 House Price Prediction System</div>
        <div class="hero-subtitle">AI-powered real estate market price estimation</div>
    </div>
    """, unsafe_allow_html=True)

    # Input Form Container using Tabs for clean UI
    st.subheader("📋 Enter Property Details")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📍 Location & Building", 
        "📐 Size & Pricing", 
        "🚌 Facilities & Connectivity", 
        "🔐 Amenities & Status"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            state = st.selectbox(
                "State",
                ["maharashtra", "karnataka", "gujarat", "delhi", "tamil nadu", "punjab", "rajasthan", "haryana", "uttar pradesh", "west bengal"],
                help="Select state location"
            )
            city = st.text_input("City", value="Mumbai", help="Enter city name").strip()
            property_type = st.selectbox(
                "Property Type",
                ["Apartment", "Independent House", "Villa"],
                help="Type of residential property"
            )
            bhk = st.slider("BHK (Bedrooms)", min_value=1, max_value=10, value=2)

        with col2:
            year = st.slider("Year Built", min_value=1980, max_value=2025, value=2018)
            age = st.number_input("Age of Property (Years)", min_value=0, max_value=100, value=max(0, 2026 - year))
            floor = st.number_input("Floor Number", min_value=0, max_value=100, value=2)
            total_floor = st.number_input("Total Floors in Building", min_value=1, max_value=100, value=10)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            size_sqft = st.number_input(
                "Property Size (Sq.Ft)",
                min_value=300,
                max_value=10000,
                value=1200,
                step=50
            )
        with col2:
            price_sqft = st.number_input(
                "Rate per Sq.Ft (₹)",
                min_value=100,
                max_value=100000,
                value=5000,
                step=100,
                help="Average rate per square feet in the locality"
            )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            school = st.slider("Nearby Schools (within 3km)", min_value=0, max_value=20, value=5)
            hospital = st.slider("Nearby Hospitals (within 3km)", min_value=0, max_value=20, value=3)
        with col2:
            transport = st.selectbox("Public Transport Accessibility", ["Low", "Medium", "High"], index=1)
            facing = st.selectbox("Property Facing Direction", ["South", "East", "West", "North"], index=1)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            furnished = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"], index=1)
            parking = st.selectbox("Parking Space Available", ["Yes", "No"], index=0)
            security = st.selectbox("24/7 Security Service", ["No", "Yes"], index=1)
        with col2:
            owner = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"], index=0)
            availability = st.selectbox("Availability Status", ["Ready to Move", "Under Construction"], index=0)
            amenities = st.text_input("Key Amenities (Optional)", value="Gym, Garden, Pool, Clubhouse", help="Comma-separated amenities")

    # Validation check
    if floor > total_floor:
        st.warning("⚠️ Floor Number cannot exceed Total Floors.")

    # Action Buttons
    st.divider()
    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        predict_btn = st.button("🚀 Predict House Price", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("🔄 Reset Inputs", use_container_width=False):
            st.rerun()

    # -----------------------------------------------------------------------------
    # PREDICTION LOGIC & PROCESSING
    # -----------------------------------------------------------------------------
    if predict_btn:
        if not city:
            st.error("Please enter a valid City name.")
        elif floor > total_floor:
            st.error("Please ensure Floor Number is not greater than Total Floors.")
        else:
            with st.spinner("🔍 Analyzing market data and predicting price..."):
                try:
                    # 1. Encode Ordinal Features
                    ordinal_df = pd.DataFrame({
                        "Property_Type": [property_type],
                        "Furnished_Status": [furnished],
                        "Public_Transport_Accessibility": [transport],
                        "Facing": [facing],
                        "Security": [security]
                    })

                    ordinal_encoded = encoder.transform(ordinal_df)

                    enc_property_type = ordinal_encoded[0][0]
                    enc_furnished = ordinal_encoded[0][1]
                    enc_transport = ordinal_encoded[0][2]
                    enc_facing = ordinal_encoded[0][3]
                    enc_security = ordinal_encoded[0][4]

                    # 2. Construct input feature dictionary
                    input_data = {
                        "State": state.lower(),
                        "City": city.lower(),
                        "Property_Type": enc_property_type,
                        "BHK": bhk,
                        "Size_in_SqFt": size_sqft,
                        "Price_per_SqFt": price_sqft,
                        "Year_Built": year,
                        "Furnished_Status": enc_furnished,
                        "Floor_No": floor,
                        "Total_Floors": total_floor,
                        "Age_of_Property": age,
                        "Nearby_Schools": school,
                        "Nearby_Hospitals": hospital,
                        "Public_Transport_Accessibility": enc_transport,
                        "Parking_Space": parking.lower(),
                        "Security": enc_security,
                        "Amenities": amenities.lower() if amenities else "",
                        "Facing": enc_facing,
                        "Owner_Type": owner.lower(),
                        "Availability_Status": availability.lower().replace(" ", "_")
                    }

                    # Preserve raw representation for summary
                    input_data_raw = input_data.copy()
                    input_data_raw["Property_Type_Raw"] = property_type

                    # 3. Vectorize and Predict
                    X = dv.transform([input_data])
                    prediction_lakhs = float(model.predict(X)[0])
                    prediction_rupees = prediction_lakhs * 100000
                    prediction_crores = prediction_lakhs / 100

                    # 4. Save to CSV History
                    save_to_history(input_data_raw, prediction_lakhs)

                    # -------------------------------------------------------------
                    # DISPLAY RESULTS
                    # -------------------------------------------------------------
                    st.balloons()
                    
                    # Large Banner Result
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.95;">Estimated Market Value</div>
                        <div class="prediction-price">₹ {prediction_lakhs:.2f} Lakhs</div>
                        <div style="font-size: 1.05rem; opacity: 0.9;">Approx. ₹ {prediction_rupees:,.0f} ({prediction_crores:.2f} Cr)</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.info("ℹ️ **Disclaimer**: This is an ML-based automated price estimation for reference and not an official appraisal.")

                    # Price Breakdown Metrics
                    st.markdown("### 💰 Price Breakdown")
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    with mcol1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Area</div>
                            <div class="metric-value">{size_sqft:,} <span style="font-size: 0.85rem; color: #64748b;">sq.ft</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    with mcol2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Rate / Sq.Ft</div>
                            <div class="metric-value">₹ {price_sqft:,}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with mcol3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Price in Lakhs</div>
                            <div class="metric-value">₹ {prediction_lakhs:.2f} L</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with mcol4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Price in Crores</div>
                            <div class="metric-value">₹ {prediction_crores:.2f} Cr</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("")

                    # Property Summary Card
                    st.markdown("### 🏠 Property Summary")
                    scol1, scol2, scol3, scol4 = st.columns(4)
                    scol1.write(f"**Location**: {city.title()}, {state.title()}")
                    scol1.write(f"**Property Type**: {property_type}")
                    scol2.write(f"**Configuration**: {bhk} BHK")
                    scol2.write(f"**Age of Property**: {age} Years")
                    scol3.write(f"**Furnishing**: {furnished}")
                    scol3.write(f"**Parking**: {parking}")
                    scol4.write(f"**Security**: {security}")
                    scol4.write(f"**Facing**: {facing}")

                    # Interactive Visualizations with Clean Light Backgrounds
                    st.divider()
                    st.markdown("### 📊 Property Analytics")
                    vcol1, vcol2 = st.columns(2)

                    with vcol1:
                        infra_df = pd.DataFrame({
                            "Facility": ["Schools", "Hospitals"],
                            "Count": [school, hospital]
                        })
                        fig1 = px.bar(
                            infra_df, x="Facility", y="Count", 
                            color="Facility", 
                            title="Nearby Facilities Overview",
                            color_discrete_sequence=["#1e293b", "#0f766e"]
                        )
                        fig1.update_layout(
                            showlegend=False, 
                            height=300, 
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig1, use_container_width=True)

                    with vcol2:
                        base_val = size_sqft * price_sqft / 100000
                        delta_val = prediction_lakhs - base_val
                        val_df = pd.DataFrame({
                            "Factor": ["Base (Area x Rate)", "ML Adjustment"],
                            "Value (Lakhs)": [base_val, max(0, delta_val)]
                        })
                        fig2 = px.pie(
                            val_df, values="Value (Lakhs)", names="Factor", 
                            title="Estimated Value Distribution",
                            color_discrete_sequence=["#2563eb", "#10b981"]
                        )
                        fig2.update_layout(
                            height=300, 
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                except Exception as e:
                    st.error("❌ Prediction Failed")
                    st.error(f"Error details: {e}")

    # Raw Feature Inspector Expander
    st.divider()
    with st.expander("📋 View Raw Input Feature Data"):
        if 'input_data' in locals():
            st.json(input_data)
        else:
            st.info("Submit a prediction to view raw model input parameters.")

    # History Table Section
    st.divider()
    st.markdown("### 📜 Recent Predictions History")
    history_df = load_history()
    if not history_df.empty:
        st.dataframe(history_df.tail(10).iloc[::-1], use_container_width=True)
    else:
        st.info("No prediction history recorded yet.")


# -----------------------------------------------------------------------------
# 6. MAIN CONTENT - PAGE 2: ABOUT THE SYSTEM
# -----------------------------------------------------------------------------
else:
    st.markdown("## ℹ️ About House Price Prediction System")
    st.write("""
    This application is an AI-driven real estate valuation tool designed to estimate residential house prices 
    based on comprehensive property, location, structural, and local infrastructure features.
    """)

    st.markdown("### 📌 Problem Statement")
    st.write("""
    Real estate price evaluation is often complex and subject to human bias or non-standardized market metrics. 
    Buyers, sellers, and agents require accurate, data-driven estimates that account for property size, location, age, 
    and surrounding facilities.
    """)

    st.markdown("### 💡 Proposed Machine Learning Solution")
    st.write("""
    By leveraging historic real estate transaction datasets, we trained a **Decision Tree Regressor** algorithm. 
    The system processes categorical attributes via **Ordinal Encoding** and converts categorical-dictionary structures 
    using **DictVectorizer** to perform accurate numeric regression.
    """)

    st.markdown("### ⚙️ System Workflow")
    st.markdown("""
    1. **User Input Collection**: Form inputs are grouped into tabs (Location, Size, Infrastructure, Facilities).
    2. **Categorical Encoding**: Ordinal attributes (Property Type, Furnishing, Transport Accessibility, Facing, Security) are encoded.
    3. **Feature Vectorization**: Feature dictionaries are converted into sparse matrices using `DictVectorizer`.
    4. **Model Inference**: The pre-trained `DecisionTreeRegressor` calculates the predicted price in Lakhs.
    5. **Dashboard Rendering**: Results are rendered alongside metric breakdowns, interactive Plotly charts, and historical logs.
    """)

    st.markdown("### 🛠️ Tech Stack & Libraries")
    st.markdown("""
    - **Language**: Python 3.x
    - **Web Framework**: Streamlit
    - **Machine Learning**: Scikit-Learn
    - **Data Preprocessing**: Pandas, NumPy
    - **Visualization**: Plotly Express
    """)


# -----------------------------------------------------------------------------
# 7. FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.875rem; padding: 1rem 0;">
    🏠 <b>House Price Prediction System</b> | Built with Python, Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)