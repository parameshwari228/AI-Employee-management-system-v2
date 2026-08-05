import sqlite3

# Connect to SQLite database
conn = sqlite3.connect("employee.db")
cursor = conn.cursor()

# ---------------- USERS TABLE ----------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL
)
""")

# ---------------- EMPLOYEES TABLE ----------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT UNIQUE,
    name TEXT NOT NULL,
    department TEXT,
    designation TEXT,
    email TEXT,
    phone TEXT
)
""")

# ---------------- ATTENDANCE TABLE ----------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT,
    date TEXT,
    punch_in TEXT,
    punch_out TEXT,
    status TEXT
)
""")

# ---------------- LEAVE REQUESTS TABLE ----------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS leave_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT,
    leave_type TEXT,
    from_date TEXT,
    to_date TEXT,
    reason TEXT,
    status TEXT DEFAULT 'Pending'
)
""")

# ---------------- SALARY TABLE ----------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS salary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT,
    month TEXT,
    basic REAL,
    hra REAL,
    bonus REAL,
    deduction REAL,
    net REAL
)
""")
#-------------------leave approval table----------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS leave_requests(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT,
    leave_type TEXT,
    from_date TEXT,
    to_date TEXT,
    reason TEXT,
    status TEXT
);

# ---------------- DEFAULT ADMIN ----------------
cursor.execute("""
INSERT OR IGNORE INTO users (username, password, role)
VALUES ('admin', 'admin123', 'Admin')
""")

conn.commit()
conn.close()

print("✅ Database created successfully!")


ss