import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import os

# Load and prepare the data
DATA_PATH = "D:/DATASET/Netflix.csv"
LOCAL_DATA_PATH = "model/cleaned_data.csv"
MODEL_PATH = "model/model.pkl"
COLUMNS_PATH = "model/feature_columns.pkl"

df = None
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    df.rename(columns={"Movie Title": "Title"}, inplace=True)
elif os.path.exists(LOCAL_DATA_PATH):
    df = pd.read_csv(LOCAL_DATA_PATH, encoding='utf-8')
    if "Title" not in df.columns and "Movie Title" not in df.columns:
        df["Title"] = "Unknown Title"
    elif "Movie Title" in df.columns:
        df.rename(columns={"Movie Title": "Title"}, inplace=True)

if df is not None:
    # Create Age Category based on Average Rating
    def categorize_age(rating):
        if rating <= 3:
            return "Kids"
        elif rating <= 6:
            return "Teens"
        else:
            return "Adults"

    if "Age Category" not in df.columns:
        df["Age Category"] = df["Average Rating"].apply(categorize_age)

    # Drop missing
    required_columns = ["Title", "Genre", "Review Highlights", "Average Rating", "Age Category"]
    df = df[required_columns].dropna()

    # Split features and label
    X = df[["Genre", "Review Highlights", "Average Rating"]]
    y = df["Age Category"]

    # Define column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Genre", "Review Highlights"]),
            ("num", "passthrough", ["Average Rating"])
        ]
    )

    # Build pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, solver='liblinear'))
    ])

    try:
        # Train model only if it doesn't exist or if forced
        if not os.path.exists(MODEL_PATH):
            print("Training model...")
            pipeline.fit(X, y)
            # Save model and columns
            pickle.dump(pipeline, open(MODEL_PATH, "wb"))
            pickle.dump(required_columns, open(COLUMNS_PATH, "wb"))
    except Exception as e:
        print(f"Training failed: {e}. Will attempt to use existing model.")

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    if df is None and not os.path.exists(MODEL_PATH):
        return "Error: No dataset found and no pre-trained model available. Please provide model/cleaned_data.csv."
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return "Error: Model file not found. Training might have failed."
        
    genre = request.form["genre"]
    review = request.form["review"]
    rating = float(request.form["rating"])

    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        input_data = pd.DataFrame([[genre, review, rating]], columns=["Genre", "Review Highlights", "Average Rating"])
        prediction_raw = model.predict(input_data)[0]

        # Map numerical prediction to string if necessary
        label_map = {0: "Kids", 1: "Teens", 2: "Adults", "0": "Kids", "1": "Teens", "2": "Adults"}
        prediction = label_map.get(prediction_raw, prediction_raw)

        # Recommendations
        if df is not None:
            # Ensure filtering works regardless of type (string/int)
            recs_df = df[df["Age Category"].astype(str) == str(prediction_raw)]
            
            if len(recs_df) >= 3:
                recs = recs_df.sample(3)[["Title", "Genre", "Review Highlights"]].values.tolist()
            elif len(recs_df) > 0:
                recs = recs_df[["Title", "Genre", "Review Highlights"]].values.tolist()
            else:
                recs = [["No matching recommendations found.", "", "Try a different genre or review."]]
        else:
            recs = [["Dataset not loaded.", "Cannot provide recommendations.", "Model used for prediction only."]]
    except Exception as e:
        return f"Prediction failed: {e}"

    return render_template("index.html", prediction=prediction, recommendations=recs)

if __name__ == "__main__":
    app.run(debug=True)
