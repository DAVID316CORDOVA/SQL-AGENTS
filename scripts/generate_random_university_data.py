import mysql.connector
import random
from faker import Faker
from datetime import datetime

fake = Faker()

# ==========================
# CONFIG
# ==========================

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123",
    "database": "demo_db"
}

NUM_STUDENTS = 500
NUM_COURSES = 50
NUM_ENROLLMENTS = 2000


# ==========================
# CONNECTION
# ==========================

conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()


# ==========================
# HELPERS
# ==========================

def generate_loyalty_and_risk(gpa):

    if gpa >= 3.6:
        return "high", random.randint(0, 20)

    elif gpa >= 2.8:
        return "medium", random.randint(20, 60)

    else:
        return "low", random.randint(60, 100)


# ==========================
# INSERT STUDENTS
# ==========================

def insert_students():

    print("Insertando students...")

    query = """
    INSERT INTO students (
        name,
        email,
        age,
        enrollment_date,
        status,
        gpa,
        city,
        risk_score,
        loyalty_level
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    ids = []

    for _ in range(NUM_STUDENTS):

        gpa = round(random.uniform(2.0, 4.0), 2)

        loyalty, risk = generate_loyalty_and_risk(gpa)

        values = (
            fake.name(),
            fake.unique.email(),
            random.randint(18, 40),
            fake.date_between(start_date="-4y", end_date="today"),
            random.choice(["active", "inactive", "graduated"]),
            gpa,
            fake.city(),
            risk,
            loyalty
        )

        cursor.execute(query, values)

        ids.append(cursor.lastrowid)

    conn.commit()

    print(f"{len(ids)} students insertados")

    return ids


# ==========================
# INSERT COURSES
# ==========================

def insert_courses():

    print("Insertando courses...")

    query = """
    INSERT INTO courses (
        course_name,
        course_code,
        credits,
        department,
        level,
        max_students,
        professor_name,
        description,
        is_active
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    ids = []

    departments = [
        "Computer Science",
        "Mathematics",
        "Physics",
        "Business",
        "Biology"
    ]

    for i in range(NUM_COURSES):

        values = (
            fake.catch_phrase(),
            f"C{i+100}",
            random.randint(1, 5),
            random.choice(departments),
            random.choice(["beginner", "intermediate", "advanced"]),
            random.randint(30, 100),
            fake.name(),
            fake.text(),
            random.choice([True, False])
        )

        cursor.execute(query, values)

        ids.append(cursor.lastrowid)

    conn.commit()

    print(f"{len(ids)} courses insertados")

    return ids


# ==========================
# INSERT ENROLLMENTS
# ==========================

def insert_enrollments(student_ids, course_ids):

    print("Insertando enrollments...")

    query = """
    INSERT IGNORE INTO enrollments (
        student_id,
        course_id,
        enrollment_date,
        grade,
        attendance_percentage,
        status,
        semester,
        year,
        final_exam_score,
        midterm_score
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    count = 0

    semesters = ["Spring", "Summer", "Fall", "Winter"]

    for _ in range(NUM_ENROLLMENTS):

        grade = round(random.uniform(50, 100), 2)

        values = (
            random.choice(student_ids),
            random.choice(course_ids),
            fake.date_between(start_date="-3y", end_date="today"),
            grade,
            random.uniform(60, 100),
            random.choice(["enrolled", "completed", "dropped", "failed"]),
            random.choice(semesters),
            random.randint(2020, 2025),
            random.uniform(50, 100),
            random.uniform(50, 100)
        )

        cursor.execute(query, values)

        count += 1

    conn.commit()

    print(f"{count} enrollments insertados")


# ==========================
# INSERT AUDIT LOG
# ==========================


def insert_audit_log(student_ids):

    print("Insertando audit_log...")

    cursor.execute("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME = 'audit_log'
    """, (DB_CONFIG["database"],))

    columns = [row[0] for row in cursor.fetchall()]

    if "table_name" not in columns:
        print("ERROR: audit_log no tiene columna table_name")
        return

    query = """
    INSERT INTO audit_log (
        table_name,
        action_type,
        record_id,
        old_value,
        new_value,
        user_name,
        action_date
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s)
    """

    for student_id in student_ids[:200]:

        values = (
            "students",
            random.choice(["INSERT","UPDATE","DELETE"]),
            student_id,
            fake.text(50),
            fake.text(50),
            fake.user_name(),
            datetime.now()
        )

        cursor.execute(query, values)

    conn.commit()

    print("audit_log insertado correctamente")


# ==========================
# MAIN
# ==========================

def main():

    print("\nGENERANDO DATA UNIVERSITARIA\n")

    student_ids = insert_students()

    course_ids = insert_courses()

    insert_enrollments(student_ids, course_ids)

    insert_audit_log(student_ids)

    print("\nDATOS GENERADOS CORRECTAMENTE\n")


if __name__ == "__main__":
    main()

    cursor.close()
    conn.close()
