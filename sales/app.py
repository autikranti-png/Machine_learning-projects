import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (High-Contrast Input Fields & Clean Header)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BigMart Retail Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Dark frosted styling with explicit input field visibility rules
st.markdown("""
<style>
    /* Full Page Background Image */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.65), rgba(15, 23, 42, 0.65)), 
                    url('https://images.unsplash.com/photo-1519567241046-7f570eee3ce6?auto=format&fit=crop&w=1920&q=80') center/cover fixed;
        color: #f8fafc;
    }

    /* Force all Streamlit labels, text, and headings to high-contrast white */
    .stApp label, .stApp .stMarkdown, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp span {
        color: #f8fafc !important;
        font-weight: 600;
    }

    /* Top Title Header Section (Image Removed) */
    .hero-container {
        background-color: rgba(15, 23, 42, 0.88);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
    }

    .hero-title {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        color: #cbd5e1 !important;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 6px;
    }

    /* Metric Cards */
    .metric-card {
        background-color: rgba(15, 23, 42, 0.88);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 5px solid #38bdf8;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    .metric-title {
        color: #94a3b8 !important;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .metric-value {
        color: #f8fafc !important;
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 4px;
    }

    /* Input Fields & Number Boxes Visibility Styling */
    div[data-baseweb="input"] input {
        color: #ffffff !important;
        background-color: #1e293b !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
    }
    
    div[data-baseweb="input"] {
        background-color: #1e293b !important;
        border: 1px solid #64748b !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border: 1px solid #64748b !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }

    /* Prediction Result Box */
    .result-card {
        background-color: rgba(20, 83, 45, 0.9);
        backdrop-filter: blur(8px);
        border: 1px solid #4ade80;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin-top: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    .result-title {
        color: #86efac !important;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .result-value {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 2px;
    }

    /* Primary Action Button */
    .stButton>button {
        background-color: #0284c7;
        color: white !important;
        border-radius: 8px;
        font-weight: 700;
        border: none;
        padding: 12px 20px;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4);
    }
    .stButton>button:hover {
        background-color: #0369a1;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open('knn_outlet_sales_model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# -----------------------------------------------------------------------------
# 3. Header Banner (Title Only, No Left Image)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛒 BigMart Sales & Prediction Dashboard</div>
    <div class="hero-subtitle">Real-time store metrics and live ML sales estimation for retail outlets</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Summary Metrics
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Avg Item MRP</div>
        <div class="metric-value">$140.99</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #4ade80;">
        <div class="metric-title">Total Outlets</div>
        <div class="metric-value">10 Stores</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #f59e0b;">
        <div class="metric-title">Top Category</div>
        <div class="metric-value">Fruits & Veggies</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #c084fc;">
        <div class="metric-title">Avg Outlet Sales</div>
        <div class="metric-value">$2,181.00</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Charts & Predictor Section
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1])

# --- Left Column: Analytics Charts ---
with left_col:
    st.subheader("🛍️ Category Distribution")
    categories = ['Fruits & Veggies', 'Snack Foods', 'Household', 'Frozen Foods', 'Others']
    counts = [1232, 1200, 910, 856, 1331]
    
    fig_pie = px.pie(
        names=categories, values=counts, hole=0.4,
        color_discrete_sequence=['#38bdf8', '#818cf8', '#c084fc', '#f472b6', '#cbd5e1']
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", size=13),
        legend=dict(font=dict(color="#f8fafc")),
        margin=dict(t=10, b=10, l=10, r=10),
        height=260
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("🏬 Sales by Store Type")
    df_chart = pd.DataFrame({
        'Store Type': ['Grocery', 'Supermarket T1', 'Supermarket T2', 'Supermarket T3'],
        'Avg Sales ($)': [340.0, 2316.0, 1995.0, 3694.0]
    })
    
    fig_bar = px.bar(
        df_chart, x='Store Type', y='Avg Sales ($)',
        color_discrete_sequence=['#38bdf8']
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", size=13),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#f8fafc")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#f8fafc")),
        margin=dict(t=10, b=10, l=10, r=10),
        height=260
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Right Column: Sales Prediction Form ---
with right_col:
    st.subheader("🔮 Predict Item Sales")

    if not model_loaded:
        st.error("⚠️ `knn_outlet_sales_model.pkl` file not found. Place it in the app directory to enable predictions.")
    else:
        f1, f2 = st.columns(2)
        with f1:
            item_weight = st.number_input("Item Weight (kg)", min_value=0.0, max_value=30.0, value=12.5, step=0.1)
            item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
            item_mrp = st.number_input("Item MRP ($)", min_value=0.0, max_value=500.0, value=140.0, step=1.0)
            outlet_establishment_year = st.selectbox("Establishment Year", [1985, 1987, 1997, 1998, 1999, 2002, 2004, 2007, 2009])

        with f2:
            item_fat_content = st.selectbox("Fat Content", ["Low Fat", "Regular"])
            item_type = st.selectbox("Item Category", [
                "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", 
                "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", 
                "Household", "Meat", "Others", "Seafood", "Snack Foods", 
                "Soft Drinks", "Starchy Foods"
            ])
            outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
            outlet_location_type = st.selectbox("Location Tier", ["Tier 1", "Tier 2", "Tier 3"])
            outlet_type = st.selectbox("Outlet Type", [
                "Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"
            ])

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Calculate Predicted Sales", use_container_width=True):
            raw_df = pd.DataFrame([{
                'Item_Weight': item_weight,
                'Item_Visibility': item_visibility,
                'Item_MRP': item_mrp,
                'Outlet_Establishment_Year': outlet_establishment_year,
                'Item_Fat_Content': item_fat_content,
                'Item_Type': item_type,
                'Outlet_Size': outlet_size,
                'Outlet_Location_Type': outlet_location_type,
                'Outlet_Type': outlet_type
            }])

            encoded_df = pd.get_dummies(raw_df)

            try:
                expected_columns = model.feature_names_in_
                final_input = encoded_df.reindex(columns=expected_columns, fill_value=0)

                prediction = model.predict(final_input)[0]

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">Estimated Sales Target</div>
                    <div class="result-value">${prediction:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")