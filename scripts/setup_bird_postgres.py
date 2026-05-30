"""
scripts/setup_bird_postgres.py

Carga las bases de datos de BIRD (SQLite) a PostgreSQL.

Uso:
  python scripts/setup_bird_postgres.py --bird_path /ruta/al/mini_dev

El script:
  1. Lee cada .sqlite de databases/ en el BIRD mini-dev
  2. Crea una base de datos PostgreSQL por cada una
  3. Migra tablas, tipos y datos (SQLite → PostgreSQL)

Requisitos:
  pip install psycopg2-binary sqlite3  (sqlite3 viene con Python)

BIRD mini-dev se descarga desde: bird-bench.github.io
"""

import os
import sys
import sqlite3
import argparse
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Mapeo de tipos SQLite → PostgreSQL
SQLITE_TO_PG = {
    "INTEGER":   "INTEGER",
    "INT":       "INTEGER",
    "TINYINT":   "SMALLINT",
    "SMALLINT":  "SMALLINT",
    "BIGINT":    "BIGINT",
    "REAL":      "DOUBLE PRECISION",
    "FLOAT":     "DOUBLE PRECISION",
    "DOUBLE":    "DOUBLE PRECISION",
    "NUMERIC":   "NUMERIC",
    "DECIMAL":   "NUMERIC",
    "TEXT":      "TEXT",
    "VARCHAR":   "TEXT",
    "CHAR":      "TEXT",
    "BLOB":      "BYTEA",
    "BOOLEAN":   "BOOLEAN",
    "DATE":      "DATE",
    "DATETIME":  "TIMESTAMP",
    "TIMESTAMP": "TIMESTAMP",
}


def get_pg_type(sqlite_type: str) -> str:
    if not sqlite_type:
        return "TEXT"
    base = sqlite_type.upper().split("(")[0].strip()
    return SQLITE_TO_PG.get(base, "TEXT")


def sanitize_value(val):
    """Convierte valores SQLite a tipos compatibles con PostgreSQL."""
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return val


def create_pg_connection(pg_config: dict, dbname: str = None):
    cfg = pg_config.copy()
    if dbname:
        cfg["dbname"] = dbname
    return psycopg2.connect(**cfg)


def drop_and_create_db(pg_config: dict, dbname: str):
    """Borra (si existe) y crea la BD en PostgreSQL."""
    conn = create_pg_connection(pg_config, dbname="postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute(
        sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [dbname]
    )
    if cur.fetchone():
        cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(dbname)))
        print(f"    BD '{dbname}' eliminada")
    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    print(f"    BD '{dbname}' creada")
    cur.close()
    conn.close()


def migrate_sqlite_to_pg(sqlite_path: str, pg_config: dict, dbname: str):
    """Migra una BD SQLite completa a PostgreSQL."""
    sq_conn = sqlite3.connect(sqlite_path)
    sq_conn.row_factory = sqlite3.Row
    sq_cur = sq_conn.cursor()

    pg_conn = create_pg_connection(pg_config, dbname=dbname)
    pg_cur = pg_conn.cursor()

    # Obtener lista de tablas
    sq_cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in sq_cur.fetchall()]

    for table in tables:
        # Obtener schema de la tabla
        sq_cur.execute(f"PRAGMA table_info('{table}')")
        columns = sq_cur.fetchall()

        if not columns:
            continue

        col_defs = []
        for col in columns:
            col_name = col[1]
            col_type = get_pg_type(col[2])
            not_null = "NOT NULL" if col[3] else ""
            is_pk = col[5]  # 1 si es pk

            if is_pk:
                col_defs.append(f'"{col_name}" {col_type} PRIMARY KEY')
            elif not_null:
                col_defs.append(f'"{col_name}" {col_type} NOT NULL')
            else:
                col_defs.append(f'"{col_name}" {col_type}')

        create_sql = f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(col_defs)})'
        try:
            pg_cur.execute(create_sql)
        except Exception as e:
            print(f"      [WARN] No se pudo crear tabla '{table}': {e}")
            pg_conn.rollback()
            continue

        # Migrar datos
        sq_cur.execute(f'SELECT * FROM "{table}"')
        rows = sq_cur.fetchall()

        if rows:
            col_names = [f'"{col[1]}"' for col in columns]
            placeholders = ", ".join(["%s"] * len(columns))
            insert_sql = f'INSERT INTO "{table}" ({", ".join(col_names)}) VALUES ({placeholders})'

            batch = []
            for row in rows:
                batch.append(tuple(sanitize_value(v) for v in row))
                if len(batch) >= 500:
                    try:
                        pg_cur.executemany(insert_sql, batch)
                    except Exception as e:
                        print(f"      [WARN] Error insertando en '{table}': {e}")
                        pg_conn.rollback()
                        batch = []
                        break
                    batch = []
            if batch:
                try:
                    pg_cur.executemany(insert_sql, batch)
                except Exception as e:
                    print(f"      [WARN] Error insertando en '{table}': {e}")
                    pg_conn.rollback()

        pg_conn.commit()
        print(f"      Tabla '{table}': {len(rows)} filas migradas")

    pg_cur.close()
    pg_conn.close()
    sq_conn.close()


def setup_bird(bird_path: str, pg_config: dict, prefix: str = "bird_"):
    """
    Carga todas las BDs del BIRD mini-dev en PostgreSQL.
    bird_path: ruta a la carpeta raiz del BIRD mini-dev
               (debe contener databases/ con subcarpetas por BD)
    """
    db_dir = os.path.join(bird_path, "databases")
    if not os.path.exists(db_dir):
        # Intentar ruta alternativa (algunos releases usan dev_databases/)
        db_dir = os.path.join(bird_path, "dev_databases")
    if not os.path.exists(db_dir):
        print(f"ERROR: No se encontro la carpeta 'databases/' en {bird_path}")
        print("  Asegurate de que bird_path apunte a la carpeta raiz del BIRD mini-dev")
        sys.exit(1)

    # Listar BDs disponibles
    db_names = [
        d for d in os.listdir(db_dir)
        if os.path.isdir(os.path.join(db_dir, d))
    ]

    if not db_names:
        print("ERROR: No se encontraron BDs en databases/")
        sys.exit(1)

    print(f"\nBDs encontradas: {len(db_names)}")
    for name in sorted(db_names):
        print(f"  - {name}")

    print(f"\nIniciando migracion a PostgreSQL...\n")

    ok = 0
    failed = []

    for db_name in sorted(db_names):
        sqlite_file = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
        if not os.path.exists(sqlite_file):
            print(f"  [SKIP] {db_name}: .sqlite no encontrado")
            failed.append(db_name)
            continue

        pg_dbname = f"{prefix}{db_name}".lower().replace("-", "_")
        print(f"  Migrando '{db_name}' -> PostgreSQL BD '{pg_dbname}'")

        try:
            drop_and_create_db(pg_config, pg_dbname)
            migrate_sqlite_to_pg(sqlite_file, pg_config, pg_dbname)
            ok += 1
            print(f"  OK '{db_name}' migrado exitosamente\n")
        except Exception as e:
            print(f"  FAIL Error en '{db_name}': {e}\n")
            failed.append(db_name)

    print("=" * 50)
    print(f"Resultado: {ok}/{len(db_names)} BDs migradas exitosamente")
    if failed:
        print(f"Fallidas: {', '.join(failed)}")
    print(f"\nPrefix usado: '{prefix}'")
    print(f"Ejemplo de conexion: dbname='{prefix}{db_names[0]}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Carga las BDs de BIRD mini-dev en PostgreSQL"
    )
    parser.add_argument(
        "--bird_path", required=True,
        help="Ruta a la carpeta raiz del BIRD mini-dev"
    )
    parser.add_argument(
        "--host", default="localhost", help="Host PostgreSQL (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=5432, help="Puerto PostgreSQL (default: 5432)"
    )
    parser.add_argument(
        "--user", default="postgres", help="Usuario PostgreSQL (default: postgres)"
    )
    parser.add_argument(
        "--password", default="postgres", help="Password PostgreSQL"
    )
    parser.add_argument(
        "--prefix", default="bird_",
        help="Prefijo para nombres de BD en PostgreSQL (default: bird_)"
    )
    args = parser.parse_args()

    pg_config = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "password": args.password,
    }

    setup_bird(args.bird_path, pg_config, prefix=args.prefix)
