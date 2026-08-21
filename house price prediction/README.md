🏠 House Price Prediction using Machine Learning

A complete House Price Prediction machine learning project that uses Linear Regression and Decision Tree Regression to predict property prices in lakhs.

The project covers the complete machine learning workflow — from Exploratory Data Analysis (EDA) and preprocessing to categorical encoding, model training, evaluation, and saving the final model components for future deployment.

📌 Project Overview

The goal of this project is to develop regression models capable of predicting the price of a house in lakhs using property, location, construction, accessibility, and amenity-related features.

Two regression algorithms are explored:

📈 Linear Regression

🌳 Decision Tree Regression

The models are evaluated using:

MAE — Mean Absolute Error

MSE — Mean Squared Error

RMSE — Root Mean Squared Error

R² Score — Coefficient of Determination

Based on the final evaluation in the notebook, the Decision Tree Regressor achieved an R² score of approximately 0.9022, making it the stronger model for this dataset.

🎯 Objectives

Develop a regression model for house price prediction.

Perform exploratory data analysis on property data.

Remove irrelevant columns.

Encode ordinal categorical variables.

Handle remaining categorical variables using DictVectorizer.

Analyze correlations among numerical variables.

Train and compare multiple regression models.

Evaluate model performance using standard regression metrics.

Save the trained model and preprocessing objects for future use.

📊 Dataset

The dataset used in the project is:

E06_house price data less.csv

The original dataset contains:

999 rows

23 columns

The dataset contains numerical and categorical information related to residential properties.

Dataset Features

Feature

Description

ID

Unique property identifier

State

State where the property is located

City

City of the property

Locality

Locality information

Property_Type

Type of property

BHK

Number of bedrooms

Size_in_SqFt

Property size in square feet

Price_in_Lakhs

Target house price in lakhs

Price_per_SqFt

Price per square foot

Year_Built

Year the property was built

Furnished_Status

Furnishing status

Floor_No

Floor number

Total_Floors

Total floors in the building

Age_of_Property

Age of the property

Nearby_Schools

Number/measure of nearby schools

Nearby_Hospitals

Number/measure of nearby hospitals

Public_Transport_Accessibility

Accessibility level

Parking_Space

Parking availability

Security

Security availability

Amenities

Available amenities

Facing

Property facing direction

Owner_Type

Type of ownership

Availability_Status

Property availability status

Target Variable

Price_in_Lakhs

The model predicts the house price in lakhs.

🔎 Exploratory Data Analysis

The notebook performs several initial data inspection steps, including:

Dataset Shape

(999, 23)

Dataset Information

The dataset contains:

9 integer columns

2 floating-point columns

12 categorical/object columns

The notebook also checks:

Data types

Missing values

Descriptive statistics

Categorical feature statistics

First and last records

Dataset dimensions

Missing Values

The notebook checks missing values using:

df.isnull().sum()

The displayed result shows no missing values for the ID column, and the notebook proceeds with preprocessing without a missing-value imputation step.

🧹 Data Preprocessing

Several preprocessing operations are performed before model training.

1. Remove the ID Column

The ID column is dropped because it is an identifier and does not provide meaningful predictive information.

df.drop('ID', axis=1, inplace=True)

2. Remove Locality

The Locality column is removed because the notebook considers it insufficiently meaningful for the modeling task.

df.drop('Locality', axis=1, inplace=True)

After these removals, the working dataset contains:

999 rows × 21 columns

3. Ordinal Encoding

The following ordinal categorical features are encoded using OrdinalEncoder:

Property_Type
Furnished_Status
Public_Transport_Accessibility
Facing
Security

The notebook defines ordered categories for these variables.

For example:

Furnished_Status:
Unfurnished → Semi-furnished → Furnished

and:

Public_Transport_Accessibility:
Low → Medium → High

The encoder is then fitted and applied to the selected columns.

4. Separate Categorical and Numerical Features

The notebook identifies features based on their data types:

categorical_col = [
    column for column in df.columns
    if df[column].dtype == 'object'
]

numerical_col = [
    column for column in df.columns
    if df[column].dtype != 'object'
]

The resulting feature groups contain:

6 categorical columns

15 numerical columns

5. Standardize Categorical Text

Categorical string values are converted to lowercase to improve consistency:

for column in categorical_col:
    df[column] = df[column].str.lower()

📈 Correlation Analysis

A correlation matrix is calculated for numerical variables:

correlation_matrix = df[numerical_col].corr()

A heatmap is then created using Seaborn to visually inspect relationships between numerical features.

This helps identify:

Positive relationships

Negative relationships

Weak correlations

Stronger relationships between numerical variables

✂️ Dataset Sampling

To reduce training time because of limited computational resources, the notebook samples 40% of the available data:

df_sampled = df.sample(frac=0.4, random_state=1)

The sampled data is then divided into:

80% training data

20% testing data

The resulting split is:

Training samples: 320
Testing samples: 80

The target variable is:

y = df_sampled['Price_in_Lakhs']

and the remaining columns are used as input features:

X = df_sampled.drop('Price_in_Lakhs', axis=1)

🔤 Categorical Feature Encoding

For the remaining categorical features, the project uses:

DictVectorizer(sparse=False)

Training records are converted into dictionaries:

train_dicts = X_train.to_dict(orient='records')
test_dicts = X_test.to_dict(orient='records')

The vectorizer is fitted on the training data and then applied to both training and testing data:

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_test = dv.transform(test_dicts)

This converts categorical variables into a numerical representation suitable for machine learning models.

🤖 Machine Learning Models

1. Linear Regression

The first model tested is Linear Regression.

model = LinearRegression(n_jobs=-1)
model.fit(X_train, y_train)

Linear Regression Results

Metric

Score

MAE

118.77

MSE

21,936.66

RMSE

148.11

R² Score

0.0312

The Linear Regression model provides a simple and interpretable baseline for the prediction task.

2. Decision Tree Regression

The second model is a Decision Tree Regressor:

model = DecisionTreeRegressor(
    random_state=42,
    max_depth=10
)

The model is trained using:

model.fit(X_train, y_train)

Decision Tree Results

Metric

Score

MAE

35.21

MSE

2,214.28

RMSE

47.06

R² Score

0.9022

🏆 Best Model

Based on the final notebook evaluation, the Decision Tree Regressor performs substantially better than the Linear Regression model for this dataset.

Its R² score is approximately:

0.9022

This means the final Decision Tree model explains a large proportion of the variation in the target values on the evaluated test set.

Note: Model performance is based on the notebook's sampled dataset and its specific train/test split.

📊 Model Comparison

Model

MAE

MSE

RMSE

R² Score

Linear Regression

118.77

21,936.66

148.11

0.0312

Decision Tree Regressor

35.21

2,214.28

47.06

0.9022

Interpretation

The Decision Tree model produces substantially lower error values and a much higher R² score than Linear Regression in the notebook's final evaluation.

Therefore, the Decision Tree model is selected as the final model for saving and future prediction use.

💾 Model Saving

The final Decision Tree model and preprocessing objects are saved using Python's pickle module.

Saved Files

house_price_model.pkl
vectorizer.pkl
encoder.pkl
features.pkl

1. Trained Model

with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

2. DictVectorizer

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(dv, f)

3. Ordinal Encoder

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

4. Feature Names

with open("features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

The notebook confirms:

All files saved successfully.

📁 Recommended Project Structure

House-Price-Prediction/
│
├── E06_House Price Prediction.ipynb
├── E06_house price data less.csv
│
├── house_price_model.pkl
├── vectorizer.pkl
├── encoder.pkl
├── features.pkl
│
├── requirements.txt
└── README.md

🛠️ Technologies & Libraries

Programming Language

Python

Data Analysis

Pandas

NumPy

Data Visualization

Matplotlib

Seaborn

Machine Learning

Scikit-learn

Models

Linear Regression

Decision Tree Regression

Preprocessing

OrdinalEncoder

DictVectorizer

Train/Test Split

Model Evaluation

Mean Absolute Error

Mean Squared Error

Root Mean Squared Error

R² Score

Model Persistence

Pickle

⚙️ Installation

Clone the repository:

git clone https://github.com/autikranti-png/Machine_learning-projects.git

Move into the project directory:

cd Machine_learning-projects

Install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

Or create a requirements.txt file containing:

pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter

Then install:

pip install -r requirements.txt

▶️ How to Run the Project

Option 1 — Jupyter Notebook

Start Jupyter Notebook:

jupyter notebook

Open:

E06_House Price Prediction.ipynb

Run the notebook cells from top to bottom.

Option 2 — JupyterLab

jupyter lab

Then open the notebook and execute the cells sequentially.

🔄 Machine Learning Workflow

House Price Dataset
        ↓
Data Loading
        ↓
Exploratory Data Analysis
        ↓
Missing Value Check
        ↓
Data Type Analysis
        ↓
Remove ID & Locality
        ↓
Ordinal Encoding
        ↓
Categorical/Numerical Separation
        ↓
Text Standardization
        ↓
Correlation Analysis
        ↓
40% Data Sampling
        ↓
Train/Test Split
        ↓
DictVectorizer Encoding
        ↓
Model Training
   ↙             ↘
Linear        Decision
Regression      Tree
   ↓             ↓
Evaluation & Comparison
        ↓
Select Decision Tree
        ↓
Save Model & Preprocessing Objects

📌 Key Learning Outcomes

This project provided practical experience with:

End-to-end regression workflow

Exploratory Data Analysis

Feature preprocessing

Ordinal categorical encoding

One-hot-style categorical vectorization using DictVectorizer

Train/test splitting

Linear Regression

Decision Tree Regression

Regression evaluation metrics

Model comparison

Model serialization using Pickle

Preparing preprocessing objects for future deployment

🚀 Future Improvements

The project can be extended with:

🔹 Hyperparameter tuning using GridSearchCV

🔹 Feature importance analysis

🔹 Cross-validation

🔹 Additional regression models such as Random Forest, Gradient Boosting and XGBoost

🔹 Interactive prediction interface using Streamlit

🔹 Dynamic visualization dashboard

🔹 Automated preprocessing pipeline

🔹 Model deployment

🔹 Prediction history

🔹 Improved validation strategy

🔹 More extensive feature engineering

⚠️ Important Notes

The notebook samples 40% of the dataset to reduce computational requirements.

Results depend on the sampled data and the fixed random states used in the notebook.

The final Decision Tree model uses max_depth=10.

The saved preprocessing objects should be used consistently with the trained model during deployment.

The reported metrics represent the notebook's final test-set evaluation and should not be interpreted as a universal performance guarantee.

🏆 Conclusion

This project demonstrates a complete machine learning workflow for house price prediction.

Both Linear Regression and Decision Tree Regression were implemented and evaluated. Linear Regression served as a simple baseline, while the Decision Tree model captured the relationships in the dataset more effectively.

Based on the notebook's final evaluation, the Decision Tree Regressor achieved an R² score of 0.9022, with an MAE of 35.21 and RMSE of 47.06.

The trained model and supporting preprocessing objects were also serialized using Pickle, making the project ready for further development toward an interactive prediction application or deployment.

👩‍💻 Author

Kranti Auti

Engineering Student | Aspiring Data Analyst & Machine Learning Enthusiast

Skills demonstrated:

Python Pandas NumPy Scikit-learn Machine Learning Regression Decision Tree Data Analysis Data Visualization

⭐ Repository

Explore more of my Machine Learning and Data Analytics projects:

GitHub:
https://github.com/autikranti-png/Machine_learning-projects


📜 Project Summary

Project: House Price Prediction
Domain: Real Estate / Machine Learning
Problem Type: Regression
Target: Price_in_Lakhs
Models: Linear Regression & Decision Tree Regression
Best Model: Decision Tree Regressor
R² Score: 0.9022
Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook
