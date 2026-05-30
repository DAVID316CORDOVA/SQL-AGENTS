import mysql.connector
from datetime import datetime

# Configuración de conexión
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123",
    "database": "demo_db"
}

conn = mysql.connector.connect(**DB_CONFIG)
cur = conn.cursor()

# =========================================
# INSERTAR 10 FILAS POR TABLA
# =========================================

# Estudiantes
students_data = [
    ("Juan García López", "estudiante1@universidad.edu", 20, "2021-09-01", "active", 3.5, "Madrid"),
    ("María Pérez Sánchez", "estudiante2@universidad.edu", 22, "2020-09-01", "graduated", 3.8, "Barcelona"),
    ("Carlos López Rivera", "estudiante3@universidad.edu", 19, "2022-01-15", "active", 3.2, "Valencia"),
    ("Ana Díaz Torres", "estudiante4@universidad.edu", 21, "2021-03-10", "inactive", 2.9, "Sevilla"),
    ("Luis Morales Cruz", "estudiante5@universidad.edu", 23, "2019-09-01", "graduated", 3.7, "Zaragoza"),
    ("Sofia Jiménez Ruiz", "estudiante6@universidad.edu", 18, "2023-02-01", "active", 3.1, "Málaga"),
    ("Diego Mendoza Vargas", "estudiante7@universidad.edu", 24, "2018-09-01", "graduated", 3.9, "Murcia"),
    ("Laura Rojas Ortiz", "estudiante8@universidad.edu", 20, "2021-10-01", "active", 3.6, "Bilbao"),
    ("Pedro Castro Ramírez", "estudiante9@universidad.edu", 22, "2020-02-01", "inactive", 2.8, "Córdoba"),
    ("Carmen Gutiérrez Gómez", "estudiante10@universidad.edu", 21, "2021-09-15", "active", 3.4, "Valencia")
]
cur.executemany("""
    INSERT INTO students (name, email, age, enrollment_date, status, gpa, city)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
""", students_data)
print(f"✔ {cur.rowcount} estudiantes insertados")

# Cursos
courses_data = [
    ("Matemáticas I", "MAT101", 4, "Matemáticas", "beginner", 30, "Dr. Juan García", "Curso de Matemáticas I", True),
    ("Física II", "FIS102", 4, "Física", "intermediate", 25, "Dr. María Pérez", "Curso de Física II", True),
    ("Química Básico", "QUI103", 3, "Ciencias", "beginner", 40, "Dr. Carlos López", "Curso de Química Básico", True),
    ("Programación Avanzado", "PRG104", 5, "Computación", "advanced", 35, "Dr. Ana Díaz", "Curso de Programación Avanzado", True),
    ("Bases de Datos I", "BDD105", 3, "Computación", "beginner", 50, "Dr. Luis Morales", "Curso de Bases de Datos I", True),
    ("Algoritmos II", "ALG106", 4, "Ingeniería", "intermediate", 45, "Dr. Sofia Jiménez", "Curso de Algoritmos II", True),
    ("Cálculo Avanzado", "CAL107", 5, "Matemáticas", "advanced", 20, "Dr. Diego Mendoza", "Curso de Cálculo Avanzado", True),
    ("Estadística I", "EST108", 4, "Matemáticas", "beginner", 30, "Dr. Laura Rojas", "Curso de Estadística I", True),
    ("Redes I", "RED109", 3, "Computación", "beginner", 40, "Dr. Pedro Castro", "Curso de Redes I", True),
    ("Sistemas Operativos Avanzado", "SOP110", 5, "Computación", "advanced", 25, "Dr. Carmen Gutiérrez", "Curso de Sistemas Operativos Avanzado", True)
]
cur.executemany("""
    INSERT INTO courses (course_name, course_code, credits, department, level, max_students, professor_name, description, is_active)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
""", courses_data)
print(f"✔ {cur.rowcount} cursos insertados")

# Inscripciones
enrollments_data = [
    (1, 1, "2021-09-01", 90.5, 95.0, "completed", "Fall 2021", 2021, 92.0, 88.0),
    (2, 2, "2020-09-01", 85.0, 90.0, "completed", "Fall 2020", 2020, 87.0, 83.0),
    (3, 3, "2022-01-15", None, 75.0, "enrolled", "Spring 2022", 2022, None, 78.0),
    (4, 4, "2021-03-10", 70.0, 65.0, "failed", "Spring 2021", 2021, 72.0, 68.0),
    (5, 5, "2019-09-01", 95.0, 98.0, "completed", "Fall 2019", 2019, 96.0, 94.0),
    (6, 6, "2023-02-01", None, 85.0, "enrolled", "Spring 2023", 2023, None, 80.0),
    (7, 7, "2018-09-01", 88.0, 92.0, "completed", "Fall 2018", 2018, 90.0, 86.0),
    (8, 8, "2021-10-01", None, 100.0, "enrolled", "Fall 2021", 2021, None, 95.0),
    (9, 9, "2020-02-01", 82.0, 87.0, "completed", "Spring 2020", 2020, 85.0, 80.0),
    (10, 10, "2021-09-15", None, 90.0, "enrolled", "Fall 2021", 2021, None, 88.0)
]
cur.executemany("""
    INSERT INTO enrollments (student_id, course_id, enrollment_date, grade, attendance_percentage, status, semester, year, final_exam_score, midterm_score)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""", enrollments_data)
print(f"✔ {cur.rowcount} inscripciones insertadas")

# Auditoría
audit_data = [
    ("students", "INSERT", 1, None, "Nuevo estudiante Juan García López", "user1", "2021-09-01 10:00:00"),
    ("students", "INSERT", 2, None, "Nuevo estudiante María Pérez Sánchez", "user2", "2020-09-01 09:00:00"),
    ("courses", "INSERT", 1, None, "Nuevo curso Matemáticas I", "user1", "2021-09-01 11:00:00"),
    ("courses", "INSERT", 2, None, "Nuevo curso Física II", "user2", "2020-09-01 12:00:00"),
    ("enrollments", "INSERT", 1, None, "Estudiante 1 inscrito en curso 1", "user1", "2021-09-01 13:00:00"),
    ("enrollments", "INSERT", 2, None, "Estudiante 2 inscrito en curso 2", "user2", "2020-09-01 14:00:00"),
    ("students", "UPDATE", 3, "Old data", "Actualización estudiante 3", "user3", "2022-01-15 15:00:00"),
    ("courses", "UPDATE", 3, "Old data", "Actualización curso 3", "user3", "2022-01-15 16:00:00"),
    ("enrollments", "UPDATE", 3, "Old data", "Actualización inscripción 3", "user3", "2022-01-15 17:00:00"),
    ("audit_log", "INSERT", 1, None, "Registro de auditoría inicial", "user1", "2021-09-01 18:00:00")
]
cur.executemany("""
    INSERT INTO audit_log (table_name, action_type, record_id, old_value, new_value, user_name, action_date)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
""", audit_data)
print(f"✔ {cur.rowcount} registros de auditoría insertados")

conn.commit()
cur.close()
conn.close()
print("✔ Datos de ejemplo insertados correctamente")
