"""
jobs/mysql/schema_extractor.py

Extrae el schema completo de la base de datos MySQL activa y lo escribe en:
    agents/MCP/metadata/mysql/<dataset>/schema.json

Este script es el paso previo obligatorio para incorporar una nueva base de datos
al sistema. El JSON generado es consumido por:
  - APS (SchemaMatcherAgent): lo indexa en ChromaDB para busqueda vectorial.
  - MCP (ar_tools): lo lee para generar descripciones de la BD en lenguaje natural.

Ejecutar cada vez que cambie la estructura de la BD (nuevas tablas, columnas, FKs):
    py jobs/mysql/schema_extractor.py

La base de datos objetivo se configura en config.py (DB_CONFIG y SCHEMA_PATH).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from config import (
        DB_CONFIG, SCHEMA_PATH, DICCIONARIO_PATH,
    )
except ImportError:
    # Valores por defecto si config.py no puede importarse (ejecucion standalone)
    DB_CONFIG = {"host": "localhost", "port": 3306,
                 "user": "root", "password": "123", "database": "demo_db"}
    SCHEMA_PATH      = "agents/MCP/metadata/mysql/demo_db/schema.json"
    DICCIONARIO_PATH = "agents/MCP/metadata/mysql/demo_db/diccionario_datos.json"

import mysql.connector


def get_connection():
    # Abre una conexion a MySQL usando los parametros de config.py
    return mysql.connector.connect(**DB_CONFIG)


def get_tables(cursor):
    # Obtiene la lista de todas las tablas de la BD activa
    cursor.execute("SHOW TABLES")
    return [row[0] for row in cursor.fetchall()]


def get_columns(cursor, table):
    # Obtiene la definicion de cada columna: nombre, tipo, nulabilidad, clave, default
    cursor.execute(f"DESCRIBE `{table}`")
    return [
        {"name": r[0], "type": r[1], "null": r[2], "key": r[3],
         "default": r[4], "extra": r[5] if len(r) > 5 else ""}
        for r in cursor.fetchall()
    ]


def get_indexes(cursor, table):
    # Extrae todos los indices de la tabla: nombre, columnas, unicidad y tipo (BTREE, HASH, etc.)
    # Se agrupan por nombre de indice porque un indice compuesto tiene una fila por columna.
    cursor.execute(f"SHOW INDEX FROM `{table}`")
    indexes = {}
    for r in cursor.fetchall():
        idx_name = r[2]
        if idx_name not in indexes:
            indexes[idx_name] = {
                "columns": [], "unique": (r[1] == 0),
                "type": r[10] if len(r) > 10 else "BTREE"
            }
        indexes[idx_name]["columns"].append(r[4])
    return indexes


def get_foreign_keys(cursor, database, table):
    # Consulta INFORMATION_SCHEMA para obtener las claves foraneas de la tabla.
    # Solo devuelve FKs que apuntan a otra tabla (REFERENCED_TABLE_NAME IS NOT NULL).
    cursor.execute("""
        SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
          AND REFERENCED_TABLE_NAME IS NOT NULL
    """, (database, table))
    return [{"column": r[0], "ref_table": r[1], "ref_column": r[2]}
            for r in cursor.fetchall()]


def get_row_count(cursor, table, database=None):
    """
    Devuelve el numero de filas de la tabla.

    Estrategia en dos pasos para minimizar el costo de la operacion:
      1. Leer del catalogo (information_schema.TABLES.TABLE_ROWS): estimado del
         optimizador, refrescado por ANALYZE. Lectura de microsegundos.
      2. Si el catalogo devuelve NULL o 0 (tabla recien creada, motor MEMORY, etc.),
         ejecutar COUNT(*) exacto que escanea la tabla completa.
    """
    db = database or DB_CONFIG.get("database")
    try:
        cursor.execute("""
            SELECT TABLE_ROWS FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        """, (db, table))
        row = cursor.fetchone()
        if row and row[0] is not None and row[0] > 0:
            return int(row[0])
    except Exception:
        pass
    # Fallback exacto: caro en tablas grandes pero necesario cuando el catalogo no tiene datos
    cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
    return cursor.fetchone()[0]


def analyze_table(cursor, table):
    """
    Refresca las estadisticas del optimizador MySQL para la tabla.

    Necesario antes de leer row_count del catalogo o tomar decisiones de muestreo,
    porque sin ANALYZE el valor de TABLE_ROWS puede estar desactualizado.
    Es idempotente: si la tabla no cambio, el costo es minimo.
    """
    try:
        cursor.execute(f"ANALYZE TABLE `{table}`")
        # ANALYZE devuelve un resultset de estado; se consume para limpiar el cursor
        try:
            cursor.fetchall()
        except Exception:
            pass
    except Exception:
        # Si ANALYZE falla (permisos insuficientes o motor que no lo soporta),
        # se continua con estadisticas potencialmente desactualizadas pero validas
        pass


def get_partitions(cursor, database, table):
    # Obtiene el metodo y expresion de particionamiento de la tabla (si esta particionada).
    # Retorna None si la tabla no usa particiones.
    cursor.execute("""
        SELECT PARTITION_METHOD, PARTITION_EXPRESSION
        FROM INFORMATION_SCHEMA.PARTITIONS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
          AND PARTITION_METHOD IS NOT NULL
        LIMIT 1
    """, (database, table))
    r = cursor.fetchone()
    return {"method": r[0], "expression": r[1]} if r else None


def classify_columns(columns):
    """
    Clasifica cada columna en una de tres categorias segun su tipo MySQL:
      - numeric    : INT, DECIMAL, FLOAT, DOUBLE, NUMERIC
      - categorical: TINYINT(1) (booleanos), columnas is_*, ENUM, SET,
                     VARCHAR/CHAR con longitud <= 100
      - date_cols  : DATE, TIMESTAMP, DATETIME

    La clasificacion guia al APS para decidir que columnas son utiles como
    filtros y cuales como valores de agregacion.
    """
    numeric, categorical, date_cols = [], [], []
    for col in columns:
        ct = col["type"].upper()
        cn = col["name"]
        if any(t in ct for t in ["DATE", "TIMESTAMP", "DATETIME"]):
            date_cols.append(cn)
        elif "TINYINT(1)" in ct or cn.startswith("is_"):
            # TINYINT(1) es el tipo MySQL para booleanos; is_* es convencion de nombre
            categorical.append(cn)
        elif any(t in ct for t in ["INT", "DECIMAL", "FLOAT", "DOUBLE", "NUMERIC"]):
            numeric.append(cn)
        elif "ENUM" in ct or "SET" in ct:
            categorical.append(cn)
        elif any(t in ct for t in ["VARCHAR", "CHAR"]):
            # VARCHAR corto (<=100 chars) probablemente almacena valores categoricos
            # como nombres, paises, estados — no texto libre
            try:
                length = int(ct.split("(")[1].split(")")[0])
                if length <= 100:
                    categorical.append(cn)
            except (IndexError, ValueError):
                pass
    return numeric, categorical, date_cols


def load_diccionario():
    # Carga el diccionario de datos opcional (descripciones de tablas y columnas).
    # Si no existe, retorna None y el schema se genera sin descripciones textuales.
    if not os.path.exists(DICCIONARIO_PATH):
        return None
    with open(DICCIONARIO_PATH, encoding="utf-8") as f:
        return json.load(f)


def extract_schema():
    """
    Funcion principal. Conecta a MySQL, extrae el schema completo de todas las
    tablas y lo escribe en el archivo JSON del MCP.

    Para cada tabla:
      1. ANALYZE para refrescar estadisticas del optimizador.
      2. Extraer columnas, indices, FKs, row_count, particiones y clasificacion.
      3. Agregar descripciones del diccionario de datos si existe.

    Al final, agrega las relaciones FK y las rutas comunes de JOIN del diccionario
    para que el APS pueda sugerir JOINs correctos sin consultarle al LLM.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    db     = DB_CONFIG["database"]

    tables      = get_tables(cursor)
    entities    = {}
    all_fks     = []
    diccionario = load_diccionario()
    dict_tablas = diccionario.get("tablas", {}) if diccionario else {}

    # Procesar cada tabla: extraer metadata completa y clasificar columnas
    for table in tables:
        # ANALYZE primero: refresca stats del optimizador para que get_row_count
        # obtenga un estimado valido del catalogo en lugar de un COUNT(*) costoso
        analyze_table(cursor, table)

        columns    = get_columns(cursor, table)
        indexes    = get_indexes(cursor, table)
        fks        = get_foreign_keys(cursor, db, table)
        row_count  = get_row_count(cursor, table)
        partitions = get_partitions(cursor, db, table)
        numeric, categorical, date_cols = classify_columns(columns)

        entity = {
            "all_columns":         [c["name"] for c in columns],
            "numeric_columns":     numeric,
            "date_columns":        date_cols,
            "categorical_columns": categorical,
            "indexes":             indexes,
            "partitions":          partitions,
            "row_count":           row_count,
        }

        # Enriquecer con descripciones del diccionario de datos si la tabla esta documentada
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

    # Agregar relaciones logicas del diccionario que no tienen FK formal en la BD.
    # Ejemplo: tablas que se vinculan por nombre de columna coincidente sin constraint.
    if diccionario:
        for rel in diccionario.get("relaciones", []):
            if rel.get("tipo") == "JOIN por nombre":
                pf = rel["desde"].split(".")
                pt = rel["hacia"].split(".")
                if len(pf) == 2 and len(pt) == 2:
                    # Solo agregar si no existe ya una FK equivalente
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

    schema = {
        "database":          db,
        "available_entities": entities,
        "relationships":      all_fks,
        "metadata": {
            "generated_by":     "schema_extractor/mysql",
            "diccionario_usado": diccionario is not None,
            "joins_comunes":     joins_comunes,
        }
    }

    # Escribir el JSON en la ruta del MCP; crear directorios si no existen
    os.makedirs(os.path.dirname(SCHEMA_PATH), exist_ok=True)
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"[MySQL] Schema guardado: {SCHEMA_PATH}")
    print(f"  Base de datos : {db}")
    print(f"  Tablas        : {len(entities)}")
    for name, info in entities.items():
        print(f"    {name}: {len(info['all_columns'])} cols, "
              f"{info['row_count']} filas, "
              f"{len(info.get('categorical_columns', []))} cols categoricas")
    print(f"  Relaciones    : {len(all_fks)}")

    cursor.close()
    conn.close()
    return schema


if __name__ == "__main__":
    extract_schema()
