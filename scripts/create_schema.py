import mysql.connector

# Configuración de conexión
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123",  # Cambia esto
    "database": "demo_db"
}

# Conexión a MySQL
conn = mysql.connector.connect(
    host=DB_CONFIG["host"],
    port=DB_CONFIG["port"],
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"]
)
cur = conn.cursor()

# Crear base de datos si no existe
cur.execute("CREATE DATABASE IF NOT EXISTS demo_db")
print("✔ Base de datos 'demo_db' creada")

# Usar la base de datos
cur.execute("USE demo_db")

# =========================================
# CREACIÓN DE TABLAS
# =========================================

# Tabla estudiantes
cur.execute("""
CREATE TABLE IF NOT EXISTS students (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    age INT CHECK (age >= 16 AND age <= 100),
    enrollment_date DATE NOT NULL,
    status ENUM('active', 'inactive', 'graduated') NOT NULL,
    gpa DECIMAL(3,2) CHECK (gpa >= 0.0 AND gpa <= 4.0),
    city VARCHAR(50)
)
""")
print("✔ Tabla 'students' creada")

# Tabla cursos
cur.execute("""
CREATE TABLE IF NOT EXISTS courses (
    course_id INT AUTO_INCREMENT PRIMARY KEY,
    course_name VARCHAR(100) NOT NULL,
    course_code VARCHAR(10) NOT NULL UNIQUE,
    credits INT CHECK (credits BETWEEN 1 AND 10),
    department VARCHAR(50),
    level ENUM('beginner', 'intermediate', 'advanced'),
    max_students INT CHECK (max_students > 0),
    professor_name VARCHAR(100),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE
)
""")
print("✔ Tabla 'courses' creada")

# Tabla inscripciones
cur.execute("""
CREATE TABLE IF NOT EXISTS enrollments (
    enrollment_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    course_id INT NOT NULL,
    enrollment_date DATE NOT NULL,
    grade DECIMAL(5,2) CHECK (grade BETWEEN 0 AND 100),
    attendance_percentage DECIMAL(5,2) CHECK (attendance_percentage BETWEEN 0 AND 100),
    status ENUM('enrolled','completed','dropped','failed') NOT NULL,
    semester VARCHAR(20) NOT NULL,
    year INT CHECK (year >= 2000 AND year <= 2100),
    final_exam_score DECIMAL(5,2) CHECK (final_exam_score BETWEEN 0 AND 100),
    midterm_score DECIMAL(5,2) CHECK (midterm_score BETWEEN 0 AND 100),
    UNIQUE(student_id, course_id, semester, year),
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE
)
""")
print("✔ Tabla 'enrollments' creada")

# Tabla auditoría
cur.execute("""
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INT AUTO_INCREMENT PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    action_type ENUM('INSERT','UPDATE','DELETE') NOT NULL,
    record_id INT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    user_name VARCHAR(50),
    action_date DATETIME NOT NULL
)
""")
print("✔ Tabla 'audit_log' creada")

conn.commit()
cur.close()
conn.close()
print("✔ Todas las tablas creadas correctamente")
import mysql.connector

# Configuración de conexión
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123",  # Cambia esto
    "database": "demo_db"
}

# Conexión a MySQL
conn = mysql.connector.connect(
    host=DB_CONFIG["host"],
    port=DB_CONFIG["port"],
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"]
)
cur = conn.cursor()

# Crear base de datos si no existe
cur.execute("CREATE DATABASE IF NOT EXISTS demo_db")
print("✔ Base de datos 'demo_db' creada")

# Usar la base de datos
cur.execute("USE demo_db")

# =========================================
# CREACIÓN DE TABLAS
# =========================================

# Tabla estudiantes
cur.execute("""
CREATE TABLE IF NOT EXISTS students (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    age INT CHECK (age >= 16 AND age <= 100),
    enrollment_date DATE NOT NULL,
    status ENUM('active', 'inactive', 'graduated') NOT NULL,
    gpa DECIMAL(3,2) CHECK (gpa >= 0.0 AND gpa <= 4.0),
    city VARCHAR(50)
)
""")
print("✔ Tabla 'students' creada")

# Tabla cursos
cur.execute("""
CREATE TABLE IF NOT EXISTS courses (
    course_id INT AUTO_INCREMENT PRIMARY KEY,
    course_name VARCHAR(100) NOT NULL,
    course_code VARCHAR(10) NOT NULL UNIQUE,
    credits INT CHECK (credits BETWEEN 1 AND 10),
    department VARCHAR(50),
    level ENUM('beginner', 'intermediate', 'advanced'),
    max_students INT CHECK (max_students > 0),
    professor_name VARCHAR(100),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE
)
""")
print("✔ Tabla 'courses' creada")

# Tabla inscripciones
cur.execute("""
CREATE TABLE IF NOT EXISTS enrollments (
    enrollment_id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    course_id INT NOT NULL,
    enrollment_date DATE NOT NULL,
    grade DECIMAL(5,2) CHECK (grade BETWEEN 0 AND 100),
    attendance_percentage DECIMAL(5,2) CHECK (attendance_percentage BETWEEN 0 AND 100),
    status ENUM('enrolled','completed','dropped','failed') NOT NULL,
    semester VARCHAR(20) NOT NULL,
    year INT CHECK (year >= 2000 AND year <= 2100),
    final_exam_score DECIMAL(5,2) CHECK (final_exam_score BETWEEN 0 AND 100),
    midterm_score DECIMAL(5,2) CHECK (midterm_score BETWEEN 0 AND 100),
    UNIQUE(student_id, course_id, semester, year),
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE
)
""")
print("✔ Tabla 'enrollments' creada")

# Tabla auditoría
cur.execute("""
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INT AUTO_INCREMENT PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    action_type ENUM('INSERT','UPDATE','DELETE') NOT NULL,
    record_id INT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    user_name VARCHAR(50),
    action_date DATETIME NOT NULL
)
""")
print("✔ Tabla 'audit_log' creada")

conn.commit()
cur.close()
conn.close()
print("✔ Todas las tablas creadas correctamente")
