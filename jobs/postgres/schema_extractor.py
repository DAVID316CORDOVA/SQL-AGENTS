"""
jobs/postgres/schema_extractor.py

Extrae el schema completo de la base de datos PostgreSQL activa y lo escribe en:
    agents/MCP/metadata/postgres/<dataset>/schema.json

Equivalente al extractor MySQL pero adaptado a las APIs y tipos de datos de PostgreSQL.
El JSON generado tiene la misma estructura que el de MySQL para que APS y MCP
puedan consumirlos de forma uniforme independientemente del motor.

Ejecutar cada vez que cambie la estructura de la BD:
    py jobs/postgres/schema_extractor.py

La base de datos objetivo se configura en config.py (PG_CONFIG y PG_SCHEMA_PATH).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from config import (
        PG_CONFIG, PG_SCHEMA_PATH, PG_DICCIONARIO_PATH,
    )
except ImportError:
    # Valores por defecto si config.py no puede importarse (ejecucion standalone)
    PG_CONFIG = {"host": "localhost", "port": 5433,
                 "user": "postgres", "password": "123", "dbname": "demo_db"}
    PG_SCHEMA_PATH      = "agents/MCP/metadata/postgres/demo_db/schema.json"
    PG_DICCIONARIO_PATH = "agents/MCP/metadata/postgres/demo_db/diccionario_datos.json"

import psycopg2


def get_connection():
    # Abre una conexion a PostgreSQL usando los parametros de config.py (PG_CONFIG)
    return psycopg2.connect(**PG_CONFIG)


def get_tables(cursor, schema="public"):
    # Obtiene solo tablas BASE (excluye vistas y tablas de sistema) del esquema indicado
    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """, (schema,))
    return [r[0] for r in cursor.fetchall()]


def get_columns(cursor, table, schema="public"):
    # Extrae la definicion de cada columna ordenada por posicion en la tabla.
    # Incluye udt_name para detectar tipos ENUM definidos por el usuario.
    cursor.execute("""
        SELECT column_name, data_type, character_maximum_length,
               is_nullable, column_default, udt_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """, (schema, table))
    return [
        {"name": r[0], "type": r[1], "length": r[2],
         "null": r[3], "default": r[4], "udt": r[5]}
        for r in cursor.fetchall()
    ]


def get_indexes(cursor, table, schema="public"):
    # Consulta los catalogos de sistema pg_class, pg_index y pg_attribute para obtener
    # indices con sus columnas en el orden correcto. Se agrupa por nombre de indice
    # porque los indices compuestos tienen una fila por columna en pg_attribute.
    cursor.execute("""
        SELECT i.relname AS index_name,
               ix.indisunique AS is_unique,
               array_agg(a.attname ORDER BY k.n) AS columns
        FROM pg_class t
        JOIN pg_index ix ON t.oid = ix.indrelid
        JOIN pg_class i  ON i.oid = ix.indexrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS k(attnum, n)
             ON TRUE
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
        WHERE t.relname = %s AND n.nspname = %s
        GROUP BY i.relname, ix.indisunique
    """, (table, schema))
    indexes = {}
    for r in cursor.fetchall():
        indexes[r[0]] = {"columns": list(r[2]), "unique": r[1], "type": "BTREE"}
    return indexes


def get_foreign_keys(cursor, table, schema="public"):
    # Extrae FKs usando information_schema. Combina tres vistas para obtener
    # tabla origen, columna FK y tabla/columna referenciada.
    cursor.execute("""
        SELECT kcu.column_name, ccu.table_name, ccu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
             ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema   = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
             ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema   = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = %s AND tc.table_name = %s
    """, (schema, table))
    return [{"column": r[0], "ref_table": r[1], "ref_column": r[2]}
            for r in cursor.fetchall()]


def get_row_count(cursor, table, schema="public"):
    """
    Devuelve el numero de filas de la tabla.

    Estrategia en dos pasos:
      1. pg_class.reltuples: estimado del planificador, actualizado por ANALYZE.
         Lectura de microsegundos; no escanea la tabla.
      2. COUNT(*) exacto si reltuples es 0 o negativo (tabla recien creada
         sin ANALYZE efectivo o tabla vacia).
    """
    try:
        cursor.execute("""
            SELECT c.reltuples::bigint
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = %s AND n.nspname = %s
        """, (table, schema))
        row = cursor.fetchone()
        if row and row[0] is not None and row[0] > 0:
            return int(row[0])
    except Exception:
        pass
    # Fallback exacto: escaneo completo, solo si el catalogo no tiene datos validos
    cursor.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
    return cursor.fetchone()[0]


def analyze_table(cursor, table, schema="public"):
    """
    Refresca las estadisticas del planificador PostgreSQL para la tabla.

    Actualiza pg_stats y pg_class.reltuples. Sin este paso, una tabla recien
    cargada puede reportar reltuples=0 aunque tenga millones de filas, lo que
    llevaria a get_row_count a ejecutar un COUNT(*) innecesario.
    Es idempotente: costo minimo si la tabla no cambio.
    """
    try:
        cursor.execute(f'ANALYZE "{schema}"."{table}"')
    except Exception:
        # Si falla (permisos, conexion en transaccion abortada), se continua
        # con estadisticas potencialmente desactualizadas pero funcionales
        pass


def classify_columns(columns):
    """
    Clasifica cada columna PostgreSQL en una de tres categorias:
      - numeric    : integer, bigint, smallint, numeric, decimal, real, double precision,
                     serial, bigserial
      - categorical: boolean, user-defined (ENUM), character varying/varchar/char con
                     longitud <= 100
      - date_cols  : date, timestamp (con y sin zona horaria), time

    TEXT sin longitud definida se omite: podria contener texto libre largo
    que no sirve como filtro categorico.
    """
    numeric, categorical, date_cols = [], [], []
    NUMERIC_TYPES = {"integer", "bigint", "smallint", "numeric", "decimal",
                     "real", "double precision", "serial", "bigserial"}
    DATE_TYPES    = {"date", "timestamp", "timestamp without time zone",
                     "timestamp with time zone", "time", "time without time zone"}

    for col in columns:
        dt  = col["type"].lower()
        cn  = col["name"]
        udt = (col.get("udt") or "").lower()

        if dt in DATE_TYPES:
            date_cols.append(cn)
        elif dt == "boolean":
            categorical.append(cn)
        elif dt in NUMERIC_TYPES:
            numeric.append(cn)
        elif dt == "user-defined" and udt:
            # Tipos ENUM creados con CREATE TYPE ... AS ENUM en PostgreSQL
            categorical.append(cn)
        elif dt in ("character varying", "varchar", "char", "character"):
            length = col.get("length")
            if length and length <= 100:
                categorical.append(cn)
        # TEXT sin longitud se omite intencionalmente

    return numeric, categorical, date_cols


def load_diccionario():
    # Carga el diccionario de datos opcional con descripciones de tablas y columnas.
    # Retorna None si no existe el archivo — el schema se genera sin descripciones.
    if not os.path.exists(PG_DICCIONARIO_PATH):
        return None
    with open(PG_DICCIONARIO_PATH, encoding="utf-8") as f:
        return json.load(f)


def extract_schema(schema="public"):
    """
    Funcion principal. Conecta a PostgreSQL, extrae el schema completo de todas
    las tablas del esquema indicado (por defecto 'public') y lo escribe en el
    archivo JSON del MCP.

    Para cada tabla:
      1. ANALYZE para refrescar estadisticas del planificador.
      2. Extraer columnas, indices, FKs, row_count y clasificacion de columnas.
      3. Enriquecer con descripciones del diccionario de datos si existe.

    Al final, consolida todas las relaciones FK y las rutas comunes de JOIN.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    db     = PG_CONFIG["dbname"]

    tables      = get_tables(cursor, schema)
    entities    = {}
    all_fks     = []
    diccionario = load_diccionario()
    dict_tablas = diccionario.get("tablas", {}) if diccionario else {}

    # Procesar cada tabla: extraer metadata completa y clasificar columnas
    for table in tables:
        # ANALYZE primero: sin esto, pg_class.reltuples puede ser 0 en tablas recien
        # cargadas y get_row_count ejecutaria un COUNT(*) completo innecesariamente
        analyze_table(cursor, table, schema)

        columns   = get_columns(cursor, table, schema)
        indexes   = get_indexes(cursor, table, schema)
        fks       = get_foreign_keys(cursor, table, schema)
        row_count = get_row_count(cursor, table, schema)
        numeric, categorical, date_cols = classify_columns(columns)

        entity = {
            "all_columns":         [c["name"] for c in columns],
            "numeric_columns":     numeric,
            "date_columns":        date_cols,
            "categorical_columns": categorical,
            "indexes":             indexes,
            "partitions":          None,   # PostgreSQL maneja particiones de forma diferente
            "row_count":           row_count,
        }

        # Enriquecer con descripciones del diccionario si la tabla esta documentada
        if table in dict_tablas:
            entity["table_description"] = dict_tablas[table].get("descripcion", "")
            col_descs = {}
            for cn, ci in dict_tablas[table].get("columnas", {}).items():
                col_descs[cn] = ci.get("descripcion", cn)
            entity["column_descriptions"] = col_descs

        entities[table] = entity

        # Registrar cada FK como una relacion con su hint de JOIN listo para usar
        for fk in fks:
            all_fks.append({
                "from_table": table, "from_column": fk["column"],
                "to_table":   fk["ref_table"], "to_column": fk["ref_column"],
                "join_hint":  f"{table}.{fk['column']} = {fk['ref_table']}.{fk['ref_column']}"
            })

    # Agregar relaciones logicas del diccionario sin FK formal en la BD
    if diccionario:
        for rel in diccionario.get("relaciones", []):
            if rel.get("tipo") == "JOIN por nombre":
                pf = rel["desde"].split(".")
                pt = rel["hacia"].split(".")
                if len(pf) == 2 and len(pt) == 2:
                    # Solo agregar si no existe ya una FK equivalente para evitar duplicados
                    if not any(r["from_table"] == pf[0] and r["from_column"] == pf[1]
                               for r in all_fks):
                        all_fks.append({
                            "from_table": pf[0], "from_column": pf[1],
                            "to_table":   pt[0], "to_column":   pt[1],
                            "join_hint":  f"{rel['desde']} = {rel['hacia']}",
                            "nota":       "JOIN por nombre, sin FK formal"
                        })

    # Recopilar rutas comunes de JOIN del diccionario (ejemplos de consultas frecuentes)
    joins_comunes = []
    if diccionario:
        for ruta in diccionario.get("rutas_comunes_de_join", []):
            joins_comunes.append({
                "caso": ruta.get("caso", ""),
                "ruta": ruta.get("ruta", ""),
                "sql":  ruta.get("sql_ejemplo", "")
            })

    result = {
        "database":           db,
        "available_entities": entities,
        "relationships":      all_fks,
        "metadata": {
            "generated_by":      "schema_extractor/postgres",
            "diccionario_usado": diccionario is not None,
            "joins_comunes":     joins_comunes,
        }
    }

    # Escribir el JSON en la ruta del MCP; crear directorios si no existen
    os.makedirs(os.path.dirname(PG_SCHEMA_PATH), exist_ok=True)
    with open(PG_SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[PostgreSQL] Schema guardado: {PG_SCHEMA_PATH}")
    print(f"  Base de datos : {db}")
    print(f"  Tablas        : {len(entities)}")
    for name, info in entities.items():
        print(f"    {name}: {len(info['all_columns'])} cols, "
              f"{info['row_count']} filas, "
              f"{len(info.get('categorical_columns', []))} cols categoricas")
    print(f"  Relaciones    : {len(all_fks)}")

    cursor.close()
    conn.close()
    return result


if __name__ == "__main__":
    extract_schema()
