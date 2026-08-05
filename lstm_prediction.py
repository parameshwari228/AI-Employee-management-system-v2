import numpy as np
import joblib

from tensorflow.keras.models import load_model


# Load LSTM model
model = load_model(
    "models/attendance_lstm.keras"
)


# Load scaler
scaler = joblib.load(
    "models/lstm_scaler.pkl"
)


# Previous attendance history
# 1 = Present, 0 = Absent

previous_days = np.array([
    [1],
    [1],
    [0]
])


# Scale input

scaled_data = scaler.transform(
    previous_days
)


# Reshape for LSTM
# (samples, time steps, features)

X_input = np.array([
    scaled_data
])


# Prediction

prediction = model.predict(
    X_input
)


# Convert back to original scale

future_attendance = scaler.inverse_transform(
    prediction
)


if future_attendance[0][0] >= 0.5:
    result = "Present"
else:
    result = "Absent"


print("Future Attendance Prediction:")
print(result)