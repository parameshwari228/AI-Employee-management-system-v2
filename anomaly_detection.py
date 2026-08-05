import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model


# Load new employee data
data = pd.read_csv(
    "dataset/new_employee_data.csv"
)


# Store Employee ID
employee_ids = data["Employee_ID"]


# Remove Employee ID
X = data.drop(
    "Employee_ID",
    axis=1
)


# Load scaler
scaler = joblib.load(
    "models/autoencoder_scaler.pkl"
)


# Scale data
X_scaled = scaler.transform(X)


# Load Autoencoder model
model = load_model(
    "models/autoencoder.keras"
)


# Predict reconstruction
reconstruction = model.predict(
    X_scaled
)


# Calculate anomaly score
error = np.mean(
    np.square(X_scaled - reconstruction),
    axis=1
)


# Set threshold
threshold = np.percentile(
    error,
    80
)


# Result
result = pd.DataFrame({

    "Employee_ID": employee_ids,

    "Anomaly_Score": error,

    "Status": np.where(
        error > threshold,
        "Anomaly",
        "Normal"
    )

})


print("\nEmployee Anomaly Detection Result\n")
print(result)


# Save output
result.to_csv(
    "dataset/anomaly_result.csv",
    index=False
)

print("\nSaved: dataset/anomaly_result.csv")

