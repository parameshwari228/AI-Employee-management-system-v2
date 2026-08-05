import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import joblib

# Load dataset
data = pd.read_csv("dataset/promotion_data.csv")

print(data.head())

# Convert text columns into numbers
performance_encoder = LabelEncoder()
promotion_encoder = LabelEncoder()

data["Performance"] = performance_encoder.fit_transform(data["Performance"])
data["Promotion"] = promotion_encoder.fit_transform(data["Promotion"])

# Input features
X = data[["Attendance", "Experience", "Projects", "Performance"]]

# Output
y = data["Promotion"]

# Train Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# Save model and encoder
joblib.dump(model, "promotion_model.pkl")
joblib.dump(performance_encoder, "performance_encoder.pkl")

print("Promotion Prediction Model Created Successfully!")