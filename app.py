from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
import sqlite3
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np 
model = joblib.load("attendance_model.pkl")
leave_model = joblib.load("leave_model.pkl")
cluster_model = joblib.load("cluster_model.pkl")
performance_model = joblib.load("performance_model.pkl")
promotion_model = joblib.load("promotion_model.pkl")
performance_encoder=joblib.load("performance_encoder.pkl")
attrition_model = load_model("models/attrition_ann.keras")
attrition_scaler = joblib.load("models/attrition_scaler.pkl")
attrition_encoder = joblib.load("models/attrition_encoder.pkl")
anomaly_model = load_model("models/autoencoder.keras")
anomaly_scaler = joblib.load("models/autoencoder_scaler.pkl")
lstm_model = load_model("models/attendance_lstm.keras")
lstm_scaler = joblib.load("models/lstm_scaler.pkl")
bert_model = BertForSequenceClassification.from_pretrained(
    "models/bert_sentiment_model"
)

bert_tokenizer = BertTokenizer.from_pretrained(
    "models/bert_sentiment_model"
)

bert_encoder = joblib.load(
    "models/bert_label_encoder.pkl"
)

def get_db_connection():
    conn = sqlite3.connect("employee.db")
    conn.row_factory = sqlite3.Row
    return conn

from flask import Flask, render_template, request, redirect, url_for, session
from datetime import datetime

app = Flask(__name__)
app.secret_key = "employee_management_secret_key"

# ---------------- HOME ----------------
@app.route("/")
def home():
    return redirect(url_for("login"))

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        # Simple Login
        if username == "admin" and password == "admin123":
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html",
                                   error="Invalid Username or Password")

    return render_template("login.html")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():

    if "username" not in session:
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        username=session["username"],
        total_employees=20,
        present=18,
        leave=2,
        salary_generated=20
    )

# ---------------- PROFILE ----------------
@app.route("/profile")
def profile():

    if "username" not in session:
        return redirect(url_for("login"))

    employee = {
        "id": "EMP001",
        "name": "Admin User",
        "department": "Human Resource",
        "role": "HR Manager",
        "email": "admin@gmail.com",
        "phone": "9876543210"
    }

    return render_template("profile.html", employee=employee)

# ---------------- ATTENDANCE ----------------
@app.route("/attendance", methods=["GET", "POST"])
def attendance():

    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    message = ""

    current_date = datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.now().strftime("%H:%M:%S")

    if request.method == "POST":

        action = request.form["action"]

        if action == "Punch In":

            existing = conn.execute("""
                SELECT * FROM attendance
                WHERE employee_id=? AND date=?
            """, ("EMP001", current_date)).fetchone()

            if existing:
                message = "Already Punched In Today"
            else:
                conn.execute("""
                    INSERT INTO attendance
                    (employee_id, date, punch_in, status)
                    VALUES (?, ?, ?, ?)
                """, ("EMP001", current_date, current_time, "Present"))

                conn.commit()
                message = "Punch In Successful"

        elif action == "Punch Out":

            conn.execute("""
                UPDATE attendance
                SET punch_out=?
                WHERE employee_id=? AND date=?
            """, (current_time, "EMP001", current_date))

            conn.commit()
            message = "Punch Out Successful"

    attendance = conn.execute("""
        SELECT * FROM attendance
        WHERE employee_id=?
        ORDER BY id DESC
        LIMIT 1
    """, ("EMP001",)).fetchone()

    conn.close()

    return render_template(
        "attendance.html",
        current_date=current_date,
        message=message,
        attendance=attendance
    )
#------------attandance_records----------------
@app.route("/attendance_records")
def attendance_records():

    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()

    attendance = conn.execute(
        "SELECT * FROM attendance"
    ).fetchall()

    conn.close()

    return render_template(
        "attendance_records.html",
        attendance=attendance
    )

# ---------------- LEAVE ----------------
@app.route("/leave", methods=["GET", "POST"])
def leave():

    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    message = ""

    if request.method == "POST":

        employee_id = "EMP001"

        leave_type = request.form["leave_type"]
        from_date = request.form["from_date"]
        to_date = request.form["to_date"]
        reason = request.form["reason"]

        conn.execute("""
        INSERT INTO leave_requests
        (employee_id, leave_type, from_date, to_date, reason, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (employee_id, leave_type, from_date, to_date, reason, "Pending"))

        conn.commit()

        message = "Leave request submitted successfully."

    leaves = conn.execute("""
        SELECT * FROM leave_requests
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return render_template(
        "leave.html",
        message=message,
        leaves=leaves
    )
#-----------------approval of leave----------------
@app.route("/leave_approval")
def leave_approval():

    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()

    leaves = conn.execute(
        "SELECT * FROM leave_requests ORDER BY id DESC"
    ).fetchall()

    conn.close()

    return render_template("leave_approval.html", leaves=leaves)


@app.route("/approve_leave/<int:id>")
def approve_leave(id):

    conn = get_db_connection()

    conn.execute(
        "UPDATE leave_requests SET status='Approved' WHERE id=?",
        (id,)
    )

    conn.commit()
    conn.close()

    return redirect(url_for("leave_approval"))


@app.route("/reject_leave/<int:id>")
def reject_leave(id):

    conn = get_db_connection()

    conn.execute(
        "UPDATE leave_requests SET status='Rejected' WHERE id=?",
        (id,)
    )

    conn.commit()
    conn.close()

    return redirect(url_for("leave_approval"))
#------------------leave prediction----------------
@app.route("/leave_prediction", methods=["GET", "POST"])
def leave_prediction():

    if "username" not in session:
        return redirect(url_for("login"))

    result = ""

    if request.method == "POST":

        attendance = float(request.form["attendance"])
        leave_days = float(request.form["leave_days"])
        experience = float(request.form["experience"])
        previous_leaves = float(request.form["previous_leaves"])

        prediction = leave_model.predict(
            [[attendance, leave_days, experience, previous_leaves]]
        )

        if prediction[0] == 1:
            result = "Approved ✅"
        else:
            result = "Rejected ❌"

    return render_template(
        "leave_prediction.html",
        result=result
    )
# ---------------- SALARY ----------------
@app.route("/salary")
def salary():

    if "username" not in session:
        return redirect(url_for("login"))

    salary = {
        "employee": "Admin User",
        "month": "July 2026",
        "basic": 30000,
        "hra": 5000,
        "bonus": 2000,
        "deduction": 1500,
        "net": 35500
    }

    return render_template("salary.html", salary=salary)
#----------------download salary slip----------------
@app.route("/download_salary")
def download_salary():

    if "username" not in session:
        return redirect(url_for("login"))

    pdf = SimpleDocTemplate("Salary_Slip.pdf")

    data = [
        ["Employee Salary Slip", ""],
        ["Employee Name", "Admin User"],
        ["Employee ID", "EMP001"],
        ["Month", "July 2026"],
        ["Basic Salary", "₹30,000"],
        ["HRA", "₹5,000"],
        ["Bonus", "₹2,000"],
        ["Deduction", "₹1,500"],
        ["Net Salary", "₹35,500"]
    ]

    table = Table(data)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.blue),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("BACKGROUND", (0,1), (-1,-1), colors.beige),
        ("BOTTOMPADDING", (0,0), (-1,0), 10),
    ]))

    pdf.build([table])

    return send_file("Salary_Slip.pdf", as_attachment=True)
#------------------ EMPLOYEES ----------------
@app.route("/employees")
def employees():
    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    employees = conn.execute("SELECT * FROM employees").fetchall()
    conn.close()

    return render_template("employees.html", employees=employees)
#---------------- ADD EMPLOYEE ----------------

@app.route("/add_employee", methods=["GET", "POST"])
def add_employee():

    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        employee_id = request.form["employee_id"]
        name = request.form["name"]
        department = request.form["department"]
        designation = request.form["designation"]
        email = request.form["email"]
        phone = request.form["phone"]

        conn = get_db_connection()

        conn.execute("""
        INSERT INTO employees
        (employee_id,name,department,designation,email,phone)
        VALUES(?,?,?,?,?,?)
        """,(employee_id,name,department,designation,email,phone))

        conn.commit()
        conn.close()

        return redirect(url_for("employees"))

    return render_template("add_employee.html")
#---------------delete_employee------------------
import sqlite3
from flask import redirect, url_for, flash

@app.route('/delete_employee/<int:id>')
def delete_employee(id):
    try:
        conn = sqlite3.connect("employee.db")   # Replace with your database file name if different
        cursor = conn.cursor()

        cursor.execute("DELETE FROM employees WHERE id = ?", (id,))
        conn.commit()

        cursor.close()
        conn.close()

        flash("Employee deleted successfully!", "success")

    except Exception as e:
        flash(f"Error: {e}", "danger")

    return redirect(url_for('employees'))

#---------------prediction----------------
@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    if "username" not in session:
        return redirect(url_for("login"))

    result = ""

    if request.method == "POST":

        attendance = float(request.form["attendance"])
        leave = float(request.form["leave"])

        prediction = model.predict([[attendance, leave]])

        result = prediction[0]

    return render_template("prediction.html", result=result)
#-----------------clustering----------------
@app.route("/employee_clustering", methods=["GET", "POST"])
def employee_clustering():

    if "username" not in session:
        return redirect(url_for("login"))

    result = ""

    if request.method == "POST":

        attendance = float(request.form["attendance"])
        leave_days = float(request.form["leave_days"])
        experience = float(request.form["experience"])
        salary = float(request.form["salary"])

        cluster = cluster_model.predict(
            [[attendance, leave_days, experience, salary]]
        )

        if cluster[0] == 2:
            result = "⭐ High Performer"

        elif cluster[0] == 1:
            result = "👍 Average Performer"

        else:
            result = "📉 Needs Improvement"

    return render_template("cluster.html", result=result)
#-------------------performance_prediction-----------------
@app.route("/performance_prediction", methods=["GET", "POST"])
def performance_prediction():

    if "username" not in session:
        return redirect(url_for("login"))

    result = ""

    if request.method == "POST":

        attendance = float(request.form["attendance"])
        leave_days = float(request.form["leave_days"])
        experience = float(request.form["experience"])
        projects = float(request.form["projects"])

        prediction = performance_model.predict(
            [[attendance, leave_days, experience, projects]]
        )

        result = prediction[0]

    return render_template(
        "performance_prediction.html",
        result=result
    )
#-------------promotion_prediction-----------
@app.route("/promotion_prediction", methods=["GET", "POST"])
def promotion_prediction():

    if "username" not in session:
        return redirect(url_for("login"))

    result = ""

    if request.method == "POST":

        attendance = float(request.form["attendance"])
        experience = float(request.form["experience"])
        projects = float(request.form["projects"])

        performance = request.form["performance"]

        performance = performance_encoder.transform([performance])[0]

        prediction = promotion_model.predict(
            [[attendance, experience, projects, performance]]
        )

        if prediction[0] == 1:
            result = "🎉 Promotion Eligible"
        else:
            result = "❌ Not Eligible"

    return render_template(
        "promotion_prediction.html",
        result=result
    )
# ---------------- ATTRITION PREDICTION (ANN) ----------------
@app.route("/attrition_prediction", methods=["GET", "POST"])
def attrition_prediction():

    if "username" not in session:
        return redirect(url_for("login"))

    result = ""

    if request.method == "POST":

        age = float(request.form["age"])
        experience = float(request.form["experience"])
        salary = float(request.form["salary"])
        job_satisfaction = float(request.form["job_satisfaction"])
        overtime_hours = float(request.form["overtime_hours"])
        leave_days = float(request.form["leave_days"])
        attendance = float(request.form["attendance"])

        data = [[
            age,
            experience,
            salary,
            job_satisfaction,
            overtime_hours,
            leave_days,
            attendance
        ]]

        data = attrition_scaler.transform(data)

        prediction = attrition_model.predict(data)

        predicted = (prediction > 0.5).astype(int)

        result = attrition_encoder.inverse_transform(predicted)[0]

    return render_template(
        "attrition_prediction.html",
        result=result
    )
# ---------------- ANOMALY DETECTION (AUTOENCODER) ----------------

@app.route("/anomaly_detection")
def anomaly_detection():

    if "username" not in session:
        return redirect(url_for("login"))

    # Load new employee data
    employee_data = pd.read_csv(
        "dataset/new_employee_data.csv"
    )

    # Store employee IDs
    employee_ids = employee_data["Employee_ID"]


    # Remove Employee ID for prediction
    X = employee_data.drop(
        "Employee_ID",
        axis=1
    )


    # Use saved scaler from training
    X_scaled = anomaly_scaler.transform(
        X
    )


    # Autoencoder prediction
    reconstruction = anomaly_model.predict(
        X_scaled
    )


    # Calculate reconstruction error
    anomaly_score = np.mean(
        np.square(X_scaled - reconstruction),
        axis=1
    )


    # Set threshold
    threshold = np.percentile(
        anomaly_score,
        80
    )


    # Create result dataframe
    result = pd.DataFrame({

        "Employee_ID": employee_ids,

        "Anomaly_Score": anomaly_score,

        "Status": np.where(
            anomaly_score > threshold,
            "Anomaly",
            "Normal"
        )
    })


    # Convert to HTML data
    anomaly_data = result.to_dict(
        orient="records"
    )


    return render_template(
        "anomaly.html",
        data=anomaly_data
    )
# ---------------- FUTURE ATTENDANCE PREDICTION (LSTM) ----------------

@app.route("/future_attendance_prediction", methods=["GET", "POST"])
def future_attendance_prediction():

    if "username" not in session:
        return redirect(url_for("login"))

    result = ""

    if request.method == "POST":

        day1 = float(request.form["day1"])
        day2 = float(request.form["day2"])
        day3 = float(request.form["day3"])


        data = np.array([
            [day1],
            [day2],
            [day3]
        ])


        scaled_data = lstm_scaler.transform(
            data
        )


        X_input = np.array([
            scaled_data
        ])


        prediction = lstm_model.predict(
            X_input
        )


        future = lstm_scaler.inverse_transform(
            prediction
        )


        if future[0][0] >= 0.5:
            result = "Present ✅"
        else:
            result = "Absent ❌"


    return render_template(
        "future_attendance_prediction.html",
        result=result
    )

#--------------feedback------------------
@app.route("/feedback_sentiment", methods=["GET", "POST"])
def feedback_sentiment():

    if "username" not in session:
        return redirect(url_for("login"))

    result = ""

    if request.method == "POST":

        feedback = request.form["feedback"]

        inputs = bert_tokenizer(
            feedback,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = bert_model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=1).item()

        result = bert_encoder.inverse_transform([prediction])[0]

    return render_template(
        "feedback_sentiment.html",
        result=result
    )

        

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():

    session.pop("username", None)

    return redirect(url_for("login"))

# ---------------- RUN ----------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    