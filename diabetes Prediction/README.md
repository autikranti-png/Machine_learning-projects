# 🩸 DiabetesAI – Diabetes Prediction Using Decision Tree

> **Machine Learning Predictive Analytics & Interactive Clinical Decision Support Dashboard**

DiabetesAI is a machine learning-based diabetes prediction project that uses a **Decision Tree Classifier** to predict whether a patient is likely to have diabetes based on clinical and physiological parameters.

The project includes both a **Jupyter Notebook machine learning workflow** and an interactive **Streamlit web application**. The trained model is saved using Python's `pickle` module and integrated into the dashboard for real-time predictions.

⚠️ **Medical Disclaimer:** This project is intended for educational, portfolio, research, and demonstration purposes only. It is **not a medical diagnostic system** and should not be used as a substitute for professional medical advice.

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Project Objectives](#-project-objectives)
* [Key Features](#-key-features)
* [Technology Stack](#-technology-stack)
* [Dataset](#-dataset)
* [Input Features](#-input-features)
* [Machine Learning Workflow](#-machine-learning-workflow)
* [Decision Tree Model](#-decision-tree-model)
* [Hyperparameter Tuning](#-hyperparameter-tuning)
* [Model Evaluation](#-model-evaluation)
* [Streamlit Application](#-streamlit-application)
* [Application Modules](#-application-modules)
* [Risk Classification](#-risk-classification)
* [Project Architecture](#-project-architecture)
* [Project Structure](#-project-structure)
* [Installation](#️-installation)
* [How to Run](#️-how-to-run)
* [How the Prediction Works](#-how-the-prediction-works)
* [Prediction History](#-prediction-history)
* [Report Export](#-report-export)
* [Example Workflow](#-example-workflow)
* [Future Enhancements](#-future-enhancements)
* [Limitations](#-limitations)
* [Conclusion](#-conclusion)
* [Author](#-author)

---

# 🧠 Project Overview

Diabetes is a common chronic disease that can be associated with factors such as glucose level, BMI, age, insulin level, blood pressure, and family history.

This project applies **supervised machine learning** to the diabetes dataset and builds a Decision Tree classification model.

The machine learning workflow includes:

1. Importing required libraries
2. Loading the diabetes dataset
3. Exploring the dataset
4. Separating features and target
5. Splitting the dataset into training and testing sets
6. Training a Decision Tree Classifier
7. Evaluating the initial model
8. Performing hyperparameter tuning using `GridSearchCV`
9. Training the optimized Decision Tree
10. Evaluating the optimized model
11. Saving the trained model
12. Saving the feature names
13. Integrating the model with a Streamlit application

The Streamlit application then provides an interactive interface for entering patient parameters and obtaining a model-based diabetes prediction.

---

# 🎯 Project Objectives

The main objectives of this project are:

* Build a machine learning model for diabetes classification.
* Understand the complete Decision Tree workflow.
* Compare model performance before and after hyperparameter tuning.
* Optimize the Decision Tree using `GridSearchCV`.
* Save the trained model for deployment.
* Build an interactive Streamlit dashboard.
* Allow users to enter patient parameters.
* Generate diabetes probability and risk classification.
* Maintain prediction history.
* Provide interactive analytics and dataset exploration.
* Export individual prediction reports as CSV files.

---

# ✨ Key Features

## 🤖 Machine Learning

* Decision Tree Classification
* Gini impurity criterion
* Train-test split
* Model evaluation
* Confusion matrix
* Classification report
* Accuracy calculation
* Hyperparameter optimization using GridSearchCV

## 🖥️ Interactive Dashboard

The Streamlit application provides multiple sections:

* 🏠 Overview
* 🔮 Prediction Engine
* 📊 Analytics Hub
* 📜 Prediction History
* 🗂 Dataset Explorer
* 🤖 Model Architecture
* 📚 Clinical Insights
* ℹ️ About

The application uses a horizontal navigation interface instead of the default sidebar.

## 📊 Visualization

The dashboard uses Plotly to create:

* Risk probability gauge
* Patient metric radar chart
* Historical classification pie chart
* Risk-level distribution histogram
* Feature distribution histogram

## 📁 Prediction Management

The application can:

* Store previous predictions
* Display prediction history
* Calculate analytics from previous predictions
* Export prediction records
* Generate individual patient summary reports

---

# 🛠️ Technology Stack

| Technology           | Purpose                                        |
| -------------------- | ---------------------------------------------- |
| **Python**           | Main programming language                      |
| **Pandas**           | Data loading and manipulation                  |
| **NumPy**            | Numerical operations and hyperparameter ranges |
| **Scikit-learn**     | Machine learning and model evaluation          |
| **Decision Tree**    | Classification algorithm                       |
| **GridSearchCV**     | Hyperparameter optimization                    |
| **Matplotlib**       | Static visualization                           |
| **Seaborn**          | Statistical visualization                      |
| **Plotly**           | Interactive dashboard visualizations           |
| **Streamlit**        | Web application                                |
| **Pickle**           | Model serialization                            |
| **CSV**              | Dataset and prediction-history storage         |
| **Jupyter Notebook** | Model development and experimentation          |

The Streamlit application specifically uses Python, Streamlit, Scikit-learn, Plotly, Pandas, and NumPy.

---

# 📊 Dataset

The project uses a diabetes dataset stored as:

```text
diabetes.csv
```

The target column is:

```text
Outcome
```

The target represents the binary diabetes classification:

```text
0 → No Diabetes
1 → Diabetes
```

The notebook separates the dataset into input features and target using:

```python
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
```

The dataset is then divided into training and testing subsets using an **80:20 split**.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

---

# 🧬 Input Features

The model uses **8 numerical features**.

| No. | Feature                  | Description                                  |
| --: | ------------------------ | -------------------------------------------- |
|   1 | Pregnancies              | Number of pregnancies                        |
|   2 | Glucose                  | Plasma glucose concentration                 |
|   3 | BloodPressure            | Diastolic blood pressure                     |
|   4 | SkinThickness            | Triceps skin fold thickness                  |
|   5 | Insulin                  | 2-hour serum insulin                         |
|   6 | BMI                      | Body Mass Index                              |
|   7 | DiabetesPedigreeFunction | Family-history-based diabetes pedigree score |
|   8 | Age                      | Patient age                                  |

The deployed application preserves this exact feature ordering when constructing the prediction input.

---

# 🔄 Machine Learning Workflow

```text
                 ┌─────────────────────┐
                 │   Diabetes Dataset  │
                 │    diabetes.csv     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Data Exploration  │
                 │ df.info(), head()   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Feature / Target    │
                 │ Separation          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Train-Test Split    │
                 │       80 : 20       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Decision Tree       │
                 │ Classifier          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Model Evaluation    │
                 │ Accuracy / CM /     │
                 │ Classification      │
                 │ Report              │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ GridSearchCV        │
                 │ Hyperparameter      │
                 │ Tuning              │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Optimized Decision  │
                 │ Tree Model          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Save Model (.pkl)   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit DiabetesAI│
                 │ Dashboard           │
                 └─────────────────────┘
```

---

# 🌳 Decision Tree Model

The initial machine learning model is a:

```text
DecisionTreeClassifier
```

The first model uses:

```python
DecisionTreeClassifier(criterion='gini')
```

The optimized model used by the deployed application has the following configuration:

| Parameter          | Value                    |
| ------------------ | ------------------------ |
| Algorithm          | Decision Tree Classifier |
| Criterion          | Gini                     |
| Max Depth          | 7                        |
| Min Samples Leaf   | 15                       |
| Min Samples Split  | 2                        |
| Number of Features | 8                        |
| Target             | Binary Outcome (0/1)     |

These hyperparameters are also displayed in the application's **Model Architecture** section.

---

# ⚙️ Hyperparameter Tuning

To improve the Decision Tree model, `GridSearchCV` is used.

The search space includes:

```python
hyperparameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(2, 10),
    'min_samples_split': np.arange(2, 15),
    'min_samples_leaf': np.arange(2, 16)
}
```

The model performs **5-fold cross-validation**:

```python
gscv_dt_model = GridSearchCV(
    dt_model,
    hyperparameters,
    cv=5
)
```

The best configuration identified in the notebook and used for deployment is:

```python
best_dt_model = DecisionTreeClassifier(
    max_depth=7,
    min_samples_leaf=15,
    min_samples_split=2
)
```

The optimized model is then trained using:

```python
best_dt_model.fit(X_train, y_train)
```

---

# 📈 Model Evaluation

The project evaluates the model using:

### 1. Accuracy

Accuracy measures the proportion of correctly classified observations.

```python
accuracy_score(y_test, y_pred_test)
```

### 2. Classification Report

The project generates a classification report containing standard classification metrics:

```python
classification_report(
    y_test,
    y_pred_test
)
```

### 3. Confusion Matrix

A confusion matrix is generated to analyze correct and incorrect classifications:

```python
confusion_matrix(
    y_test,
    y_pred_test
)
```

The notebook evaluates both training and testing performance before and after GridSearchCV optimization.

> **Note:** The exact numerical accuracy values should be taken from the executed notebook rather than hard-coded into this README.

---

# 💾 Model Serialization

After training, the optimized model is saved using Python's `pickle` library.

```python
with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(best_dt_model, file)
```

The feature names are separately stored:

```python
features = X.columns.tolist()

with open("diabetes_features.pkl", "wb") as file:
    pickle.dump(features, file)
```

The Streamlit application loads these files when starting the application.

---

# 🖥️ Streamlit Application

The project contains an interactive web dashboard called:

## 💉 DiabetesAI Medical Intelligence

The application is designed as a machine-learning predictive analytics dashboard.

It provides real-time model predictions from eight patient parameters and displays the prediction probability, risk tier, model confidence, and visual analytics.

The dashboard describes itself as a machine-learning predictive analytics and clinical decision-support system.

---

# 📑 Application Modules

## 🏠 1. Overview

Provides an introduction to the DiabetesAI platform and explains the purpose of the system.

It highlights:

* 8 feature inputs
* Glucose
* BMI
* Insulin
* Diabetes pedigree function
* Decision Tree-based prediction

---

## 🔮 2. Prediction Engine

This is the main prediction module.

Users can enter:

* Pregnancies
* Age
* Glucose
* Blood Pressure
* BMI
* Skin Thickness
* Insulin
* Diabetes Pedigree Function

The application then sends these values to the trained Decision Tree model.

The model returns:

```text
Prediction
+
Probability
+
Risk Category
+
Model Confidence
```

The prediction process uses both:

```python
model.predict()
```

and:

```python
model.predict_proba()
```

The application then calculates the diabetes probability and assigns a risk category.

---

# 📊 3. Analytics Hub

The Analytics Hub analyzes previously recorded predictions.

It displays:

* Total evaluations
* Positive cases
* Negative cases
* Average risk score

It also provides interactive charts such as:

### Historical Classification Ratio

A pie chart showing the distribution of predicted classes.

### Risk Tier Distribution

A histogram showing the distribution of:

* Low Risk
* Moderate Risk
* High Risk
* Very High Risk

---

# 📜 4. Prediction History

Every completed prediction is recorded in:

```text
prediction_history.csv
```

The stored information includes:

* Timestamp
* Patient input parameters
* Prediction
* Probability
* Risk level

The application displays these historical records in a table and allows the registry to be exported as CSV.

---

# 🗂️ 5. Dataset Explorer

The Dataset Explorer provides basic exploratory data analysis.

When `diabetes.csv` is available, the dashboard displays:

* Total patient records
* Number of feature parameters
* Positive diagnoses
* First 10 dataset records
* Feature distributions

Users can select individual feature parameters and inspect their distributions using interactive Plotly histograms.

---

# 🤖 6. Model Architecture

This section explains the deployed Decision Tree model.

It displays:

* Algorithm type
* Maximum depth
* Minimum samples per leaf
* Minimum samples split
* Target outcome
* Feature count

It also shows the exact feature-vector ordering used by the application.

---

# 📚 7. Clinical Insights

The application provides informational explanations of important diabetes-related parameters, including:

* Plasma glucose
* BMI
* Insulin
* Diabetes pedigree function

These insights are provided for educational context and are not intended as medical advice.

---

# ℹ️ 8. About

The About section provides:

* Project description
* System information
* Technology stack
* Purpose of the application

The implemented application identifies its main technology stack as:

```text
Python
Streamlit
Scikit-learn
Plotly
Pandas
NumPy
```

---

# 🚦 Risk Classification

The application converts the predicted diabetes probability into a risk category.

| Probability | Risk Level        |
| ----------: | ----------------- |
|    0% – 30% | 🟢 Low Risk       |
|  >30% – 60% | 🟡 Moderate Risk  |
|  >60% – 80% | 🟠 High Risk      |
| >80% – 100% | 🔴 Very High Risk |

These thresholds are implemented directly in the application's `get_risk_category()` function.

---

# 📊 Prediction Visualization

After prediction, the application provides two major visualizations.

### Risk Probability Gauge

Displays the estimated diabetes probability from:

```text
0% → 100%
```

with different visual ranges for different risk levels.

### Relative Metric Spectrum

A radar chart compares normalized values for:

* Glucose
* Blood Pressure
* BMI
* Insulin
* Age

This gives users a visual representation of the entered patient metrics.

---

# 📁 Project Architecture

```text
User Input
    │
    ▼
Streamlit Interface
    │
    ▼
Input Validation
    │
    ▼
Pandas DataFrame
    │
    ▼
Saved Decision Tree Model
    │
    ├── Prediction
    │
    └── Probability
            │
            ▼
       Risk Classification
            │
            ├── Low Risk
            ├── Moderate Risk
            ├── High Risk
            └── Very High Risk
            │
            ▼
     Dashboard Visualization
            │
            ├── Gauge Chart
            ├── Radar Chart
            └── Result Cards
            │
            ▼
    Prediction History CSV
            │
            ▼
      Analytics Dashboard
```


### File Description

| File                                           | Description                                     |
| ---------------------------------------------- | ----------------------------------------------- |
| `app.py`                                       | Streamlit application                           |
| `diabetes.csv`                                 | Diabetes dataset                                |
| `diabetes_model.pkl`                           | Trained Decision Tree model                     |
| `diabetes_features.pkl`                        | Saved model feature names                       |
| `prediction_history.csv`                       | Prediction history generated by the application |
| `Day 58_Decision Tree Diabetes Data (1).ipynb` | Machine learning development notebook           |
| `DT_Diabetes.png`                              | Decision Tree visualization                     |
| `requirements.txt`                             | Required Python libraries                       |
| `README.md`                                    | Project documentation                           |
| `.gitignore`                                   | Files excluded from Git                         |

> Your current Streamlit file is named `app 3.py`. For a cleaner GitHub repository, you can rename it to `app.py`, provided you update your run command accordingly.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/DiabetesAI.git
```

Move into the project directory:

```bash
cd DiabetesAI
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

Create a `requirements.txt` file containing:

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
plotly
streamlit
```

Then install:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

Make sure the following files are located in the same directory as the Streamlit application:

```text
app.py
diabetes_model.pkl
diabetes_features.pkl
diabetes.csv
```

The application explicitly loads the model and feature files from the working directory.

Run:

```bash
streamlit run app.py
```

If you keep the original filename:

```bash
streamlit run "app 3.py"
```

Streamlit will start the local application and provide a browser URL.

---

# 🔍 How the Prediction Works

### Step 1 — Enter Patient Information

The user provides eight numerical parameters.

### Step 2 — Create Feature DataFrame

The entered values are converted into a Pandas DataFrame.

### Step 3 — Model Prediction

The saved Decision Tree model predicts:

```text
0 → No Diabetes
1 → Diabetes
```

### Step 4 — Probability Calculation

The model's `predict_proba()` output is converted into percentages.

### Step 5 — Risk Categorization

The diabetes probability is mapped to one of four risk categories.

### Step 6 — Visualization

The dashboard displays:

* Classification result
* Diabetes probability
* Risk tier
* Model confidence
* Gauge chart
* Radar chart

### Step 7 — Save Prediction

The prediction is appended to:

```text
prediction_history.csv
```

---

# 🧾 Prediction History

The application automatically creates the prediction history file if it does not already exist.

The initial columns include:

```text
Timestamp
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Prediction
Probability
RiskLevel
```

The application appends each completed assessment to this CSV file.

---

# 📥 Report Export

The Prediction Engine provides an option to export a patient assessment summary.

The generated report contains:

* Patient classification
* Diabetes probability
* Risk category
* Glucose
* Blood pressure
* BMI
* Insulin
* Age
* Diabetes pedigree function
* Assessment timestamp

The report is downloaded as a CSV file from the Streamlit dashboard.

---

# 📊 Analytics Workflow

Once predictions are generated, the Analytics Hub can calculate:

```text
Total Evaluations
        │
        ├── Positive Cases
        │
        └── Negative Cases
```

It also calculates:

```text
Average Risk Score
```

and generates graphical representations of prediction outcomes and risk categories.

This makes the application more than a single prediction interface—it also provides a simple prediction-monitoring dashboard.

---

# 🧪 Model Development Workflow

The Jupyter Notebook follows this process:

```text
Import Libraries
       ↓
Load Dataset
       ↓
Explore Dataset
       ↓
Separate X and y
       ↓
Train-Test Split
       ↓
Initial Decision Tree
       ↓
Training Evaluation
       ↓
Testing Evaluation
       ↓
Decision Tree Visualization
       ↓
GridSearchCV
       ↓
Best Parameters
       ↓
Optimized Decision Tree
       ↓
Training Evaluation
       ↓
Testing Evaluation
       ↓
Save Model
       ↓
Save Feature Names
```

---

# 📌 Important Implementation Details

### Training Split

```text
Training Data: 80%
Testing Data: 20%
Random State: 42
```

### Cross Validation

```text
GridSearchCV
CV = 5
```

### Final Model

```text
Decision Tree Classifier
Max Depth = 7
Min Samples Leaf = 15
Min Samples Split = 2
```

### Saved Assets

```text
diabetes_model.pkl
diabetes_features.pkl
```

---

# 🎨 Dashboard Design

The Streamlit application includes a custom interface with:

* Responsive wide layout
* Horizontal navigation
* Custom CSS styling
* Glassmorphism cards
* KPI cards
* Risk badges
* Interactive Plotly charts
* Custom header banner
* Medical disclaimer

The application uses a custom background, high-contrast labels, glass-style cards, and customized navigation/button styling.

---

# 🔐 Data & Privacy Considerations

This project is designed as an educational machine learning application.

If deployed publicly:

* Avoid entering real patient-identifiable information.
* Do not store sensitive medical information without appropriate security controls.
* Protect prediction-history files.
* Use secure storage for production applications.
* Follow applicable healthcare and data-protection regulations.

---

# ⚠️ Limitations

This project has several limitations:

1. It is an educational machine learning project.
2. Predictions depend on the quality and representativeness of the training dataset.
3. A Decision Tree may not capture all complex relationships between clinical variables.
4. The application does not replace laboratory testing.
5. The model should not be interpreted as an official medical diagnosis.
6. The displayed risk thresholds are application-defined categories, not clinical diagnostic thresholds.
7. The prediction history is stored locally in a CSV file.
8. The application does not implement production-grade authentication or database security.

---

# 🚀 Future Enhancements

Possible improvements include:

### 🤖 Machine Learning

* Compare Decision Tree with Random Forest.
* Implement XGBoost.
* Try Logistic Regression.
* Perform cross-validation comparison.
* Add ROC-AUC analysis.
* Add feature importance visualization.
* Implement explainable AI using SHAP.

### 📊 Dashboard

* Add interactive filters.
* Add more advanced analytics.
* Add model performance dashboard.
* Add feature importance charts.
* Add downloadable PDF reports.
* Add patient trend analysis.

### 🗄️ Database

Instead of storing predictions in CSV, future versions could use:

```text
MySQL
PostgreSQL
MongoDB
```

### 🔐 Security

A production version could include:

* User authentication
* Role-based access
* Secure database storage
* Encryption
* Audit logs

### ☁️ Deployment

The application could be deployed using:

* Streamlit Community Cloud
* Docker
* Cloud platforms
* Other Python-compatible hosting services

---

# 💡 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Python programming
* Data loading with Pandas
* Exploratory Data Analysis
* Feature-target separation
* Train-test splitting
* Decision Tree Classification
* Model training
* Hyperparameter tuning
* GridSearchCV
* Cross-validation
* Accuracy evaluation
* Confusion matrix
* Classification report
* Model serialization
* Streamlit development
* Interactive Plotly visualization
* CSV data management
* Machine learning deployment

---

# 🏆 Project Highlights

### Machine Learning

```text
✔ Decision Tree Classifier
✔ Hyperparameter Optimization
✔ GridSearchCV
✔ Model Evaluation
✔ Model Serialization
```

### Application

```text
✔ Interactive Streamlit Dashboard
✔ Real-Time Prediction
✔ Probability Estimation
✔ Risk Categorization
✔ Prediction History
✔ Dataset Explorer
✔ Analytics Dashboard
✔ CSV Report Export
```

### Visualization

```text
✔ Plotly Gauge Chart
✔ Radar Chart
✔ Pie Chart
✔ Histogram
✔ Feature Distribution Analysis
```

---

# 📷 Screenshots

Add screenshots of your application here after uploading them to GitHub.

Example:

```markdown
## 📸 Screenshots

### 🏠 Overview
![Overview](screenshots/overview.png)

### 🔮 Prediction Engine
![Prediction Engine](screenshots/prediction.png)

### 📊 Analytics Hub
![Analytics](screenshots/analytics.png)

### 🤖 Model Architecture
![Model Architecture](screenshots/model.png)
```

Recommended screenshot folder:

```text
screenshots/
├── overview.png
├── prediction.png
├── analytics.png
├── history.png
├── dataset.png
└── model.png
```

---

# 🔗 Project Files

The main machine learning notebook contains the complete model-building workflow, including data loading, train-test splitting, Decision Tree training, evaluation, hyperparameter tuning, and model serialization.

The Streamlit application integrates the saved model into an interactive dashboard.

---

# ⚠️ Medical Disclaimer

> **This application is strictly intended for educational, portfolio, research, and informational demonstration purposes.**
>
> It does not provide official medical diagnoses, treatment recommendations, or professional medical consultation.
>
> Any real-world medical decision should be made only by qualified healthcare professionals using appropriate clinical evaluation and diagnostic testing.

The same disclaimer is explicitly included in the application interface.

---

# 📌 Conclusion

**DiabetesAI** demonstrates an end-to-end machine learning workflow, from dataset exploration and Decision Tree training to hyperparameter optimization, model serialization, and deployment through an interactive Streamlit dashboard.

The project combines:

**Machine Learning + Data Analysis + Visualization + Web Application Development**

into a single practical application.

It provides a strong demonstration of how a trained classification model can be integrated into an interactive application for educational and portfolio purposes.

---

# 👩‍💻 Author

**Kranti Auti**

Engineering Student | Aspiring Data Scientist | Machine Learning & Data Analytics Enthusiast

### Skills Demonstrated

```text
Python
Machine Learning
Data Science
Pandas
NumPy
Scikit-learn
Decision Trees
GridSearchCV
Streamlit
Plotly
Data Visualization
Jupyter Notebook
Git & GitHub
```

---
 
