import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Load training dataset
data = pd.read_csv(
    "dataset/employee_anomaly.csv"
)


# Remove Employee ID
X = data.drop(
    "Employee_ID",
    axis=1
)


# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Save scaler
joblib.dump(
    scaler,
    "models/autoencoder_scaler.pkl"
)


# Create Autoencoder model

model = Sequential([

    Dense(
        8,
        activation="relu",
        input_shape=(X_scaled.shape[1],)
    ),

    Dense(
        4,
        activation="relu"
    ),

    Dense(
        8,
        activation="relu"
    ),

    Dense(
        X_scaled.shape[1],
        activation="linear"
    )

])


# Compile model

model.compile(
    optimizer="adam",
    loss="mse"
)


# Train model

model.fit(
    X_scaled,
    X_scaled,
    epochs=50,
    batch_size=4
)


# Save trained model

model.save(
    "models/autoencoder.keras"
)


print("Autoencoder Training Completed Successfully")
print("Model saved: models/autoencoder.keras")
print("Scaler saved: models/autoencoder_scaler.pkl")