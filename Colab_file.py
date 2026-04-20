# Colab_file.py (Model Training Script)

import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("D:/DATASET/Netflix.csv", encoding='utf-8')

# Assign age category based on Average Rating
def assign_age_category(rating):
    if rating <= 3:
        return "Kids"
    elif rating <= 6:
        return "Teens"
    else:
        return "Adults"

df = df[["Movie Title", "Genre", "Review Highlights", "Average Rating"]].dropna()
df.rename(columns={"Movie Title": "Title"}, inplace=True)
df["Age Category"] = df["Average Rating"].apply(assign_age_category)

# Features and target
X = df[["Genre", "Review Highlights", "Average Rating"]]
y = df["Age Category"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
categorical_features = ["Genre", "Review Highlights"]
numeric_features = ["Average Rating"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ("num", StandardScaler(), numeric_features)
])

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
import os
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Save df with Age Category for Flask app
df.to_csv("model/cleaned_data.csv", index=False)
print("\nModel and cleaned data saved successfully.")
