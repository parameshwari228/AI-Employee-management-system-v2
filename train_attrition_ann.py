import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = pd.read_csv("dataset/employee_attrition_100.csv")

# Input features
X = data[[
    "Age",
    "Experience",
    "Salary",
    "Job_Satisfaction",
    "Overtime_Hours",
    "Leave_Days",
    "Attendance"
]]

# Target
y = data["Attrition"]

# Encode target (Stay / Leave)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler and encoder
joblib.dump(scaler, "models/attrition_scaler.pkl")
joblib.dump(encoder, "models/attrition_encoder.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build ANN model
model = Sequential([
    Dense(16, activation="relu", input_shape=(7,)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# Save model
model.save("models/attrition_ann.keras")

print("Employee Attrition ANN Model Saved Successfully!")