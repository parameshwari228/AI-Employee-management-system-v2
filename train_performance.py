import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("dataset/performance_data.csv")

# Display dataset
print(data.head())

# Input features
X = data[["Attendance", "Leave_Days", "Experience", "Projects"]]

# Output label
y = data["Performance"]

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)

# Save the model
joblib.dump(model, "performance_model.pkl")

print("Performance Prediction Model Created Successfully!")