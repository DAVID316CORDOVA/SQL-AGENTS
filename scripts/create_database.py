import mysql.connector

# Conexión SIN especificar la base de datos
conn = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="123"  # Cambia esto
)

cur = conn.cursor()

# Crear la base de datos si no existe
cur.execute("CREATE DATABASE IF NOT EXISTS demo_db")
print("✔ Base de datos 'demo_db' creada")

# Usar la base de datos
cur.execute("USE demo_db")

cur.close()
conn.close()

print("✔ Listo para crear tablas")