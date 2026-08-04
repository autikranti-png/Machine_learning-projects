import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model safely
MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None  # Prevents startup crash if file is missing

HISTORY_FILE = "prediction_history.csv"
COLUMNS = ["cgpa", "iq", "profile_score", "Prediction"]


def init_history_file():
    """Ensure the history file exists with headers."""
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(HISTORY_FILE, index=False)


# Initialize CSV on start
init_history_file()


def get_recent_history(n=5):
    """Safely fetch the most recent predictions (latest first)."""
    if os.path.exists(HISTORY_FILE):
        try:
            history = pd.read_csv(HISTORY_FILE)
            if history.empty:
                return []
            # Return last N entries, reversed so newest is at the top
            return history.tail(n).iloc[::-1].values.tolist()
        except pd.errors.EmptyDataError:
            return []
    return []


def save_prediction(cgpa, iq, profile_score, result):
    """Helper function to safely append a new record to CSV."""
    new_record = pd.DataFrame(
        [[cgpa, iq, profile_score, result]], 
        columns=COLUMNS
    )
    new_record.to_csv(
        HISTORY_FILE, 
        mode="a", 
        header=not os.path.exists(HISTORY_FILE), 
        index=False
    )


@app.route("/")
def home():
    return render_template("index.html", history=get_recent_history())


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            raise RuntimeError("Model file 'model.pkl' not found. Please upload or train the model.")

        # Cast and validate input types
        cgpa = float(request.form["cgpa"])
        iq = int(request.form["iq"])
        profile_score = int(request.form["profile_score"])

        # Input Validation
        if not (0.0 <= cgpa <= 10.0):
            raise ValueError("CGPA must be between 0 and 10.")
        if iq <= 0:
            raise ValueError("IQ must be greater than 0.")
        if not (0 <= profile_score <= 100):
            raise ValueError("Profile Score must be between 0 and 100.")

        # Prepare input DataFrame for model (matches training feature names)
        input_data = pd.DataFrame(
            [[cgpa, iq, profile_score]],
            columns=["cgpa", "iq", "profile_score"]
        )

        # Make prediction
        prediction = model.predict(input_data)
        result = "Placed" if prediction[0] == 1 else "Not Placed"

        # Save record
        save_prediction(cgpa, iq, profile_score, result)

        return render_template(
            "index.html",
            result=result,
            history=get_recent_history()
        )

    except ValueError as ve:
        # Catch specific user input validation errors
        return render_template(
            "index.html",
            error=str(ve),
            history=get_recent_history()
        )
    except Exception as e:
        # Catch any general model or system errors
        return render_template(
            "index.html",
            error=f"An unexpected error occurred: {str(e)}",
            history=get_recent_history()
        )


if __name__ == "__main__":
    app.run(debug=True)