"""
jobs/schema_extractor.py

Dispatcher de conveniencia: selecciona automaticamente el extractor correcto
segun el motor de base de datos configurado en config.py (DB_TYPE).

NO contiene logica propia. Simplemente redirige la ejecucion a:
  - jobs/mysql/schema_extractor.py    si DB_TYPE == "mysql"  (o cualquier valor que no sea "postgres")
  - jobs/postgres/schema_extractor.py si DB_TYPE == "postgres"

Para mayor claridad, se recomienda invocar el extractor especifico directamente:
    py jobs/mysql/schema_extractor.py
    py jobs/postgres/schema_extractor.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from config import DB_TYPE
except ImportError:
    # Si config.py no esta disponible, asumir MySQL como motor por defecto
    DB_TYPE = "mysql"

# Delegar la funcion extract_schema al modulo del motor activo
if DB_TYPE.lower() == "postgres":
    from jobs.postgres.schema_extractor import extract_schema
else:
    from jobs.mysql.schema_extractor import extract_schema

if __name__ == "__main__":
    extract_schema()
