"""
agents/MCP/tools/shared_tools.py

Tools MCP de uso transversal: no estan ligadas a un agente especifico del
sistema y pueden ser invocadas por cualquier consumidor MCP, incluyendo
clientes externos como Claude Desktop o Cursor.

Hoy ningun agente del pipeline las invoca directamente, pero estan
disponibles como utilidades de introspeccion del schema.
"""
from agents.MCP._mcp_instance import mcp, DESCRIPTIONS_DIR, load_schema


@mcp.tool()
def list_available_databases_server() -> list[str]:
    """
    Lista los identificadores de las bases de datos que tienen descripcion
    registrada en este servidor.

    El identificador retornado usa el mismo formato que espera
    get_database_description.

    Returns:
        Lista de strings con los identificadores disponibles. Lista vacia si
        todavia no se ha registrado ninguna descripcion.
    """
    if not DESCRIPTIONS_DIR.exists():
        return []
    result = []
    for md in sorted(DESCRIPTIONS_DIR.glob("*.md")):
        stem = md.stem
        for family in ("bird", "spider", "wikisql"):
            if stem.startswith(f"{family}_"):
                db_id = stem[len(family) + 1:]
                result.append(f"{family}:{db_id}")
                break
        else:
            result.append(stem)
    return result


@mcp.tool()
def get_table_relationships_server(table: str, db_name: str,
                                    backend: str = "mysql") -> dict:
    """
    Devuelve todas las relaciones (foreign keys) en las que participa una tabla.

    Util para que un agente conozca a que otras tablas se conecta una tabla
    dada y mediante que columnas, sin necesidad de inspeccionar todo el schema.

    Args:
        table: nombre de la tabla a consultar.
        db_name: identificador de la base de datos.

    Returns:
        Dict con:
          - exists: bool, si la tabla existe en el schema.
          - outgoing: lista de relaciones donde la tabla es origen
                      (sus FKs apuntan a otras tablas).
          - incoming: lista de relaciones donde la tabla es destino
                      (otras tablas apuntan a ella).
          Cada relacion incluye from_table, from_column, to_table, to_column,
          join_hint.
    """
    schema = load_schema(db_name, backend)
    if table not in schema.get("available_entities", {}):
        return {"exists": False, "outgoing": [], "incoming": []}
    outgoing, incoming = [], []
    for rel in schema.get("relationships", []):
        item = {
            "from_table": rel["from_table"],
            "from_column": rel["from_column"],
            "to_table": rel["to_table"],
            "to_column": rel["to_column"],
            "join_hint": f"{rel['from_table']}.{rel['from_column']} = "
                         f"{rel['to_table']}.{rel['to_column']}",
        }
        if rel["from_table"] == table:
            outgoing.append(item)
        elif rel["to_table"] == table:
            incoming.append(item)
    return {"exists": True, "outgoing": outgoing, "incoming": incoming}


@mcp.tool()
def get_pk_server(table: str, db_name: str,
                  backend: str = "mysql") -> dict:
    """
    Devuelve la(s) columna(s) que conforman la clave primaria de una tabla.

    Args:
        table: nombre de la tabla.
        db_name: identificador de la base de datos.

    Returns:
        Dict con:
          - exists: bool, si la tabla existe en el schema.
          - columns: lista de columnas que componen la PK (puede ser
                     compuesta). Lista vacia si la tabla no tiene PK
                     declarada en el schema.
    """
    schema = load_schema(db_name, backend)
    entities = schema.get("available_entities", {})
    if table not in entities:
        return {"exists": False, "columns": []}
    indexes = entities[table].get("indexes") or {}
    pk_info = indexes.get("PRIMARY") or {}
    return {"exists": True, "columns": list(pk_info.get("columns") or [])}


@mcp.tool()
def classify_table_role_server(table: str, db_name: str,
                                backend: str = "mysql") -> dict:
    """
    Clasifica el rol estructural de una tabla en el schema basandose en
    heuristicas de FKs, numero de columnas y filas.

    Categorias posibles:
      - junction: tabla puente entre dos o mas entidades (mayoria de columnas
                  son FKs + PK).
      - main_entity: entidad principal con muchas columnas propias y pocas
                     FKs salientes.
      - lookup: tabla pequeña de referencia/catalogo (pocas filas).
      - entity: clasificacion por defecto cuando no encaja en las anteriores.

    Args:
        table: nombre de la tabla a clasificar.
        db_name: identificador de la base de datos.

    Returns:
        Dict con:
          - exists: bool, si la tabla existe en el schema.
          - role: una de {junction, main_entity, lookup, entity}.
          - reasons: lista de strings con la justificacion (numero de FKs,
                     columnas, filas).
    """
    schema = load_schema(db_name, backend)
    entities = schema.get("available_entities", {})
    if table not in entities:
        return {"exists": False, "role": None, "reasons": []}
    info = entities[table]
    cols = info.get("all_columns") or []
    n_cols = len(cols)
    n_rows = info.get("row_count") or 0
    outgoing = sum(1 for r in schema.get("relationships", [])
                   if r.get("from_table") == table)
    incoming = sum(1 for r in schema.get("relationships", [])
                   if r.get("to_table") == table)

    reasons = [
        f"columnas={n_cols}",
        f"filas={n_rows}",
        f"fks_salientes={outgoing}",
        f"fks_entrantes={incoming}",
    ]

    if outgoing >= 2:
        role = "junction"
    elif outgoing == 0 and 0 < n_rows <= 20 and n_cols <= 5:
        role = "lookup"
    elif outgoing <= 1 and n_cols >= 6:
        role = "main_entity"
    else:
        role = "entity"

    return {"exists": True, "role": role, "reasons": reasons}
