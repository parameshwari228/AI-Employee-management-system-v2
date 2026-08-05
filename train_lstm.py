import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Load attendance dataset

data = pd.read_csv(
    "dataset/employee_attendance.csv"
)


# Select attendance column

attendance = data["Attendance"].values.reshape(-1,1)


# Scale data

scaler = MinMaxScaler()

attendance_scaled = scaler.fit_transform(
    attendance
)


# Save scaler

joblib.dump(
    scaler,
    "models/lstm_scaler.pkl"
)


# Create sequences for LSTM

X = []
y = []


sequence_length = 3


for i in range(
    len(attendance_scaled) - sequence_length
):

    X.append(
        attendance_scaled[i:i+sequence_length]
    )

    y.append(
        attendance_scaled[i+sequence_length]
    )


X = np.array(X)

y = np.array(y)


# LSTM input shape:
# (samples, time steps, features)

print("Input shape:", X.shape)


# Create LSTM model

model = Sequential([

    LSTM(
        50,
        activation="relu",
        input_shape=(sequence_length,1)
    ),

    Dense(1)

])


# Compile model

model.compile(
    optimizer="adam",
    loss="mse"
)


# Train model

model.fit(
    X,
    y,
    epochs=50,
    batch_size=4
)


# Save model

model.save(
    "models/attendance_lstm.keras"
)


print("LSTM Attendance Prediction Model Created Successfully")
