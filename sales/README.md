🛒 BigMart Sales Prediction & Retail Dashboard

A machine learning project that predicts item sales for retail outlets using a K-Nearest Neighbors (KNN) regression model and presents the prediction through an interactive Streamlit dashboard.

The application combines retail analytics, visual insights, and a live sales prediction interface in one user-friendly dashboard.

📌 Project Overview

Retail businesses need accurate sales estimates to understand product demand and support better inventory and business decisions.

This project uses BigMart-style retail data to build a sales prediction application. The trained KNN regression model is loaded into a Streamlit web application, where users can enter product and outlet information and receive an estimated sales value.

The dashboard also provides:

📊 Retail summary metrics

🥧 Item category distribution

🏬 Average sales by store type

🔮 Interactive sales prediction

🎨 Professional dark-themed dashboard UI

⚡ Cached machine learning model loading

🎯 Objectives

Predict expected sales for a retail item.

Use product and outlet characteristics as prediction inputs.

Provide an interactive interface for making individual predictions.

Present useful retail statistics through charts and KPI cards.

Create a simple and professional machine learning deployment using Streamlit.

🧠 Machine Learning Approach

Algorithm

K-Nearest Neighbors (KNN) Regression

KNN Regression predicts a continuous numerical value by looking at the nearest observations in the training data and using their target values to estimate the prediction.

Prediction Target

The application predicts:

Item Outlet Sales

Input Features Used by the Application

The Streamlit prediction form accepts the following features:

Feature

Description

Item Weight

Weight of the product in kilograms

Item Visibility

Visibility of the item in the outlet

Item MRP

Maximum Retail Price of the item

Item Fat Content

Product fat-content category

Item Type

Product category

Outlet Establishment Year

Year in which the outlet was established

Outlet Size

Size of the outlet

Outlet Location Type

Location tier of the outlet

Outlet Type

Type of retail outlet

The application creates a one-row DataFrame from the entered values and applies one-hot encoding to categorical features before passing the data to the trained model.

📊 Dataset

The supplied dataset contains 8,523 records and 12 columns.

Dataset Columns

Item_Identifier

Item_Weight

Item_Fat_Content

Item_Visibility

Item_Type

Item_MRP

Outlet_Identifier

Outlet_Establishment_Year

Outlet_Size

Outlet_Location_Type

Outlet_Type

Item_Outlet_Sales

Data Types

The dataset contains:

Numerical features such as Item Weight, Item Visibility and Item MRP

Categorical features such as Item Type, Outlet Size and Outlet Type

Item_Outlet_Sales as the numerical sales target

📈 Dashboard Features

1. Summary KPI Cards

The dashboard displays four summary metrics:

Avg Item MRP: $140.99

Total Outlets: 10 Stores

Top Category: Fruits & Veggies

Avg Outlet Sales: $2,181.00

These values are displayed as dashboard-level summary indicators.

2. Category Distribution

A donut chart displays the distribution of product categories across the dashboard's predefined categories:

Fruits & Veggies

Snack Foods

Household

Frozen Foods

Others

3. Sales by Store Type

A bar chart compares average sales across:

Grocery

Supermarket T1

Supermarket T2

Supermarket T3

4. Sales Prediction

Users can enter product and outlet information and click:

Calculate Predicted Sales

The application then displays the estimated sales value in a highlighted result card.

🖥️ Application Interface

The Streamlit dashboard includes:

Wide-screen responsive layout

Dark frosted-glass visual design

High-contrast input fields

KPI cards

Interactive Plotly charts

Prediction form

Prediction result card

Error handling when the model is unavailable or prediction fails

The application is configured with the page title BigMart Retail Dashboard and uses a shopping-cart themed page icon.

🛠️ Technologies Used

Technology

Purpose

Python

Core programming language

Pandas

Data handling and DataFrame creation

Streamlit

Interactive web application

Plotly Express

Interactive charts

Pickle

Loading the trained ML model

KNN Regression

Sales prediction

HTML/CSS

Dashboard styling

📁 Project Structure

BigMart-Sales-Prediction/
│
├── app.py
├── knn_outlet_sales_model.pkl
├── KNN_reg_outlet_sales - KNN_reg_outlet_sales.csv
└── README.md

Make sure the model file knn_outlet_sales_model.pkl is placed in the same directory as app.py when running the application.

⚙️ How the Application Works

Retail Dataset
      ↓
Data Preparation
      ↓
KNN Regression Model
      ↓
Trained Model (.pkl)
      ↓
Streamlit Application
      ↓
User Enters Product & Outlet Details
      ↓
Categorical Encoding
      ↓
Feature Alignment
      ↓
KNN Model Prediction
      ↓
Estimated Item Outlet Sales

🚀 Installation

1. Clone the Repository

git clone https://github.com/autikranti-png/Machine_learning-projects.git
cd Machine_learning-projects

2. Install Required Libraries

pip install streamlit pandas plotly

If you have a requirements.txt file, you can install dependencies using:

pip install -r requirements.txt

3. Add the Model File

Place:

knn_outlet_sales_model.pkl

in the same folder as:

app.py

▶️ Run the Application

Open Command Prompt or Anaconda Prompt and run:

streamlit run app.py

After running the command, Streamlit will provide a local URL. Open that URL in your browser to access the dashboard.

🔮 Making a Prediction

Open the Streamlit dashboard.

Go to Predict Item Sales.

Enter or select:

Item Weight

Item Visibility

Item MRP

Establishment Year

Fat Content

Item Category

Outlet Size

Location Tier

Outlet Type

Click Calculate Predicted Sales.

The predicted sales amount will be displayed under Estimated Sales Target.

🔄 Feature Processing in the Application

The application creates a DataFrame containing the user's inputs.

Categorical variables are converted using one-hot encoding:

encoded_df = pd.get_dummies(raw_df)

The encoded input is then aligned with the feature names expected by the trained model:

expected_columns = model.feature_names_in_
final_input = encoded_df.reindex(
    columns=expected_columns,
    fill_value=0
)

Finally, the model generates the prediction:

prediction = model.predict(final_input)[0]

🧩 Model Loading

The trained model is loaded from:

knn_outlet_sales_model.pkl

The application uses Streamlit's resource caching so that the model does not need to be repeatedly loaded during app interactions.

@st.cache_resource
def load_model():
    with open('knn_outlet_sales_model.pkl', 'rb') as file:
        return pickle.load(file)

🎨 Dashboard Design

The application uses a professional dark-themed interface with:

Frosted-glass cards

High-contrast white text

Rounded input fields

Highlighted KPI cards

Interactive Plotly visualizations

Dedicated prediction result section

Responsive two-column layout

The styling is implemented using custom HTML and CSS inside Streamlit.

⚠️ Important Notes

The application requires knn_outlet_sales_model.pkl to make predictions.

The model file must be in the expected application directory.

The input categories should match the categories used by the trained model.

The prediction depends on the trained KNN model and its learned feature structure.

If the model cannot be loaded, the dashboard displays an error message instead of attempting a prediction.

📌 Future Improvements

Possible improvements for the project include:

Add model evaluation metrics such as MAE, MSE and R².

Add data preprocessing and model-training notebooks.

Add more detailed EDA visualizations.

Add actual dynamic KPI calculations directly from the dataset.

Add prediction history.

Add downloadable prediction results.

Add model comparison with Linear Regression, Random Forest and XGBoost.

Add a dedicated requirements.txt.

Deploy the Streamlit application online.

Add a model performance section to the dashboard.

💡 Key Learning Outcomes

Through this project, I worked with:

Machine learning regression

KNN Regression

Retail sales prediction

Categorical feature encoding

Pandas DataFrames

Pickle model deployment

Streamlit application development

Plotly data visualization

Interactive dashboard design

Basic ML model integration into a web application

👩‍💻 Author

Kranti Auti

Engineering Student | Aspiring Data Analyst & Machine Learning Enthusiast

Skills Demonstrated

Python Pandas Machine Learning KNN Regression Streamlit Plotly Data Visualization

Machine Learning Projects Repository:
https://github.com/autikranti-png/Machine_learning-projects
