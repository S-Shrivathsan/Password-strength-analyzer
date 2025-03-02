from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import numpy as np
import pickle
import os
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Database file
DB_FILE = "password_data.db"
CSV_FILE = "passwords.csv"  # Your dataset

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    if 'password' not in df.columns or 'strength' not in df.columns:
        raise ValueError("CSV file must contain 'password' and 'strength' columns.")
    
    # Drop missing password values
    df.dropna(subset=['password', 'strength'], inplace=True)
    
    return df

# Extract features from passwords
def extract_features(password):
    if not isinstance(password, str):  # Handle missing or non-string values
        password = ""

    return [
        len(password),
        sum(char.isdigit() for char in password),
        sum(char.isupper() for char in password),
        sum(char.islower() for char in password),
        sum(char in string.punctuation for char in password)
    ]

# Load dataset and preprocess
data = load_dataset(CSV_FILE)
data['password'] = data['password'].astype(str)

# Convert strength labels to numbers
label_encoder = LabelEncoder()
data['strength_encoded'] = label_encoder.fit_transform(data['strength'])

# Feature extraction
data['features'] = data['password'].apply(extract_features)
X = np.array(data['features'].tolist())  # Ensure NumPy array
y = np.array(data['strength_encoded'])  # Ensure NumPy array

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoder
MODEL_FILE = "model.pkl"
ENCODER_FILE = "label_encoder.pkl"
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)
with open(ENCODER_FILE, "wb") as f:
    pickle.dump(label_encoder, f)

# Ensure database exists
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS passwords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            password TEXT NOT NULL,
            strength TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        password = data.get("password", "")

        if not password:
            return jsonify({"error": "Password cannot be empty"}), 400

        # Load trained model and encoder
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        with open(ENCODER_FILE, "rb") as f:
            label_encoder = pickle.load(f)

        # Extract features and make a prediction
        features = extract_features(password)
        prediction_index = model.predict([features])[0]

        # Convert numerical prediction back to label
        if isinstance(prediction_index, np.ndarray):
            prediction_index = prediction_index[0]

        prediction_label = label_encoder.inverse_transform([int(prediction_index)])[0]

        # Debugging: Print the predicted index and label in the Flask console
        print(f"Predicted Index: {prediction_index}, Label: {prediction_label}")

        # Ensure correct JSON response
        return jsonify({"strength": prediction_label})

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": "An error occurred during prediction"}), 500


if __name__ == "__main__":
    app.run(debug=True)
