import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load dataset
data = pd.read_csv("dataset/employee_cluster.csv")

print(data.head())

# Select features
X = data[["Attendance", "Leave_Days", "Experience", "Salary"]]

# Create K-Means model
model = KMeans(n_clusters=3, random_state=42)

# Train the model
model.fit(X)

# Save the model
joblib.dump(model, "cluster_model.pkl")

print("Employee Clustering Model Created Successfully!")