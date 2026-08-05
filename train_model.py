import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
import os

print(os.getcwd())
print(os.path.exists("dataset/attendance.csv"))

data = pd.read_csv("dataset/attendance.csv")


# Features
X = data[["attendance_percentage", "leave_days"]]

# Target
y = data["status"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "attendance_model.pkl")

print("AI Model Trained Successfully!")
