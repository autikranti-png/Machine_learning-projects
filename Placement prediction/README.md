# 🎓 Student Placement Prediction System

<p align="center">
  <b>Predicting student placement outcomes using Machine Learning and Flask</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" />
  <img src="https://img.shields.io/badge/Framework-Flask-green?logo=flask" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" />
</p>

---

## 📌 Project Overview

The **Student Placement Prediction System** is a Machine Learning-based web application designed to predict whether a student is likely to be **Placed** or **Not Placed** based on academic and profile-related information.

The project uses a trained Machine Learning model to analyze student details and generate a prediction through a simple and user-friendly **Flask web interface**.

This project demonstrates the complete Machine Learning workflow, including:

- Data collection
- Data preprocessing
- Feature selection
- Model training
- Model evaluation
- Model saving
- Web application development
- Prediction deployment using Flask

---

## 🎯 Project Objective

The main objective of this project is to develop an intelligent system that can estimate a student's placement outcome using important student-related features.

The system helps demonstrate how Machine Learning can be applied to educational data for predictive analysis.

---

## ✨ Key Features

✅ Predicts whether a student is likely to be **Placed** or **Not Placed**

✅ Simple and interactive web interface

✅ Accepts student information as input

✅ Uses a trained Machine Learning model

✅ Provides quick prediction results

✅ Performs input validation

✅ Uses Flask for web deployment

✅ Easy to run and understand

---

## 📊 Dataset Information

The dataset contains information related to student academic performance and profile details.

### Dataset Features

| Feature | Description |
|---|---|
| `cgpa` | Student's Cumulative Grade Point Average |
| `iq` | Student's IQ score |
| `profile_score` | Student's overall profile score |
| `placed` | Placement status |

### Target Variable

The target variable is:

| Value | Prediction |
|---:|---|
| `1` | Placed |
| `0` | Not Placed |

---

## 🧠 Machine Learning Workflow

```text
Student Dataset
       ↓
Data Preprocessing
       ↓
Feature Selection
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Save Trained Model
       ↓
Flask Web Application
       ↓
Placement Prediction
```

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Machine Learning Libraries

- Scikit-learn
- Pandas
- NumPy

### Web Framework

- Flask

### Frontend Technologies

- HTML
- CSS

### Other Tools

- Pickle
- Jupyter Notebook
- Visual Studio Code
- GitHub

---

## 📁 Project Structure

```text
Placement prediction/
│
├── app.py
├── model.pkl
├── students_placement.csv
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/autikranti-png/Machine_learning-projects.git
```

### Step 2: Open the Project Folder

```bash
cd Machine_learning-projects
```

### Step 3: Open the Placement Prediction Folder

```bash
cd "Placement prediction"
```

### Step 4: Install Required Libraries

```bash
pip install flask pandas numpy scikit-learn
```

### Step 5: Run the Flask Application

```bash
python app.py
```

### Step 6: Open the Application

After running the application, open the following address in your browser:

```text
http://127.0.0.1:5000/
```

---

## 🖥️ How to Use

1. Open the Student Placement Prediction website.
2. Enter the student's **CGPA**.
3. Enter the student's **IQ score**.
4. Enter the student's **Profile Score**.
5. Click the **Predict** button.
6. View the placement prediction result.

---

## 📈 Expected Output

The application displays one of the following results:

```text
🎉 The student is likely to be Placed
```

or

```text
📚 The student is currently predicted as Not Placed
```

> The prediction is generated using the trained Machine Learning model and should be treated as an educational project result.

---

## 🚀 Future Improvements

The following features can be added in future versions:

- Add more student-related features
- Compare multiple Machine Learning algorithms
- Improve model performance
- Add model accuracy and evaluation metrics
- Store prediction history in a database
- Add user login and authentication
- Create an admin dashboard
- Add graphs and data visualizations
- Deploy the application online
- Make the application responsive for mobile devices

---

## 📚 Learning Outcomes

Through this project, the following concepts were explored:

- Machine Learning fundamentals
- Data preprocessing
- Feature selection
- Model training
- Model prediction
- Saving and loading models using Pickle
- Flask web development
- Integrating Machine Learning with a web application
- Git and GitHub project management

---

## 👩‍💻 Author

**Kranti Auti**

Engineering Student | Machine Learning Enthusiast

---

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome.

If you find this project useful, consider giving the repository a **⭐ Star**.

---

<p align="center">
  Made with ❤️ using Python, Machine Learning, and Flask
</p>
