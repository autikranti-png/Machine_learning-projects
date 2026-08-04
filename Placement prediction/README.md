🎓 Student Placement Prediction

A Machine Learning project that predicts whether a student is likely to be placed or not placed based on academic and profile-related information.

📌 Project Overview

This project uses a Machine Learning model to analyze student data and predict placement outcomes. The prediction is based on the following input features:

CGPA – Student’s academic performance

IQ – Student’s IQ score

Profile Score – Overall student profile score

The model predicts:

1 → Placed

0 → Not Placed

📊 Dataset Information

The dataset contains 300 student records with the following columns:

Feature	Description
cgpa	Student’s Cumulative Grade Point Average

iq	Student’s IQ score

profile_score	Student’s overall profile score

placed	Placement status: 1 for Placed and 0 for Not Placed

🛠️ Technologies Used
Python

Machine Learning

Pandas

NumPy

Scikit-learn

Flask

HTML

CSS

Pickle

VS Code

GitHub

⚙️ Project Workflow

Load the student placement dataset.

Perform data preprocessing.

Select important features.

Train the Machine Learning model.

Save the trained model using Pickle.

Build a web application using Flask.

Enter student details in the web interface.

Display the placement prediction result.

🚀 How to Run the Project

1. Clone the Repository
git clone https://github.com/autikranti-png/Machine_learning-projects.git
2. Open the Project Folder
cd Machine_learning-projects
3. Install Required Libraries
pip install pandas numpy scikit-learn flask
4. Run the Flask Application
python app.py
5. Open in Browser
http://127.0.0.1:5000/

📁 Project Structure

Student_Placement_Prediction/
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
💡 Features

User-friendly web interface
Placement prediction using Machine Learning

Input validation

Fast prediction results

Flask-based deployment

Easy-to-use student placement prediction system

🎯 Future Improvements

Add more student-related features.

Improve model accuracy using advanced algorithms.

Add prediction history.

Add graphical data visualization.

Deploy the application online.

Add user authentication.

👩‍💻 Author

Kranti Auti

Engineering Student | Machine Learning Enthusiast
