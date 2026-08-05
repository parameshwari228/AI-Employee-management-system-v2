import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Read the CSV file
data = pd.read_csv("leave_data.csv")
print(data.head())
print(data.columns.tolist())

# Input columns
X = data[["Attendance", "Leave days ", "experience", "previous leaves"]]

# Output column
y = data["approved"]

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, "leave_model.pkl")

print("Leave model created successfully!")
