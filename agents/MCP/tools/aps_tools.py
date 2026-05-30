"""
agents/MCP/tools/aps_tools.py

Tools MCP que consume el Agente de Proximidad Semantica (APS).

  - search_tables          : busqueda vectorial de tablas en ChromaDB
  - search_columns         : busqueda vectorial de columnas en ChromaDB
  - find_column_for_concept: verifica compatibilidad semantica de un filtro
  - find_join_path_server  : BFS sobre grafo de FKs (mantenido para tests)
"""
import sys

import networkx as nx

from agents.MCP._mcp_instance import (
    mcp, ROOT, resolve_vector_db_path, build_graph,
)


@mcp.tool()
def search_tables_server(query: str, db_name: str, n: int = 5,
                         backend: str = "mysql") -> list:
    """
    Busqueda vectorial de tablas en ChromaDB para una query del usuario.
    Retorna lista de tablas ordenadas por similitud con toda su metadata.
    """
    import chromadb
    from chromadb.config import Settings
    vector_db_path = resolve_vector_db_path(db_name, backend)
    if not vector_db_path.exists():
        return []
    try:
        client = chromadb.PersistentClient(
            path=str(vector_db_path), settings=Settings(anonymized_telemetry=False))
        col = client.get_collection("tables")
    except Exception:
        return []
    try:
        from utils.embeddings import get_embedding
        from config import EMBEDDING_MODEL
    except Exception:
        return []
    emb = get_embedding(query, EMBEDDING_MODEL, mode="query")
    from agents.MCP._mcp_instance import load_schema
    schema = load_schema(db_name, backend)
    results = col.query(query_embeddings=[emb], n_results=min(n, col.count()))
    tables = []
    if results and results["ids"] and results["ids"][0]:
        for i, tid in enumerate(results["ids"][0]):
            name = tid.replace("table:", "")
            dist = results["distances"][0][i] if results["distances"] else 0
            sim = max(0, 1 - dist)
            info = schema.get("available_entities", {}).get(name, {})
            tables.append({
                "table_name": name, "similarity": round(sim, 3),
                "all_columns": info.get("all_columns", []),
                "numeric_columns": info.get("numeric_columns", []),
                "categorical_columns": info.get("categorical_columns", []),
                "categorical_semantic": info.get("categorical_semantic", {}),
                "indexes": info.get("indexes", {}),
                "partitions": info.get("partitions"),
                "row_count": info.get("row_count", 0),
            })
    return tables


@mcp.tool()
def search_columns_server(query: str, db_name: str, tables: list[str] | None = None,
                          n: int = 10, backend: str = "mysql") -> list:
    """
    Busqueda vectorial de columnas en ChromaDB, opcionalmente filtrada
    por un subconjunto de tablas.
    """
    import chromadb
    from chromadb.config import Settings
    vector_db_path = resolve_vector_db_path(db_name, backend)
    if not vector_db_path.exists():
        return []
    try:
        client = chromadb.PersistentClient(
            path=str(vector_db_path), settings=Settings(anonymized_telemetry=False))
        col = client.get_collection("columns")
    except Exception:
        return []
    try:
        from utils.embeddings import get_embedding
        from config import EMBEDDING_MODEL
    except Exception:
        return []
    emb = get_embedding(query, EMBEDDING_MODEL, mode="query")
    where = {"table_name": {"$in": tables}} if tables else None
    results = col.query(query_embeddings=[emb], n_results=min(n, col.count()), where=where)
    columns = []
    seen = set()
    if results and results["ids"] and results["ids"][0]:
        for i, cid in enumerate(results["ids"][0]):
            if cid in seen:
                continue
            seen.add(cid)
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else 0
            sim = max(0, 1 - dist)
            columns.append({
                "table_name": meta.get("table_name", ""),
                "column_name": meta.get("column_name", ""),
                "column_type": meta.get("column_type", ""),
                "similarity": round(sim, 3),
            })
    return columns


@mcp.tool()
def find_join_path_server(from_table: str, to_table: str, db_name: str,
                   max_hops: int = 3, backend: str = "mysql") -> dict:
    """
    Encuentra el camino mas corto de JOINs entre dos tablas usando las
    relaciones (foreign keys) declaradas en el schema.

    Resuelve el caso de tablas puente: cuando dos tablas no tienen relacion
    directa pero estan conectadas a traves de una o mas tablas intermedias
    (por ejemplo, students y courses conectadas via enrollments).

    Args:
        from_table: tabla origen del recorrido.
        to_table: tabla destino del recorrido.
        db_name: identificador de la base de datos.
        max_hops: maximo numero de saltos permitidos. 1 = relacion directa.

    Returns:
        Dict con:
          - found: bool, si existe camino dentro de max_hops.
          - path: lista ordenada de tablas a recorrer (incluyendo origen y destino).
          - joins: lista ordenada de condiciones JOIN concretas como dicts
                   con from_table, from_column, to_table, to_column, join_hint.
          - hops: numero de saltos del camino (longitud de joins).
        Si no hay camino o las tablas no existen, found=False y los demas
        campos estan vacios.
    """
    g = build_graph(db_name, backend)
    empty = {"found": False, "path": [], "joins": [], "hops": 0}
    if from_table not in g or to_table not in g:
        return empty
    if from_table == to_table:
        return {"found": True, "path": [from_table], "joins": [], "hops": 0}
    try:
        path = nx.shortest_path(g, from_table, to_table)
    except nx.NetworkXNoPath:
        return empty
    hops = len(path) - 1
    if hops > max_hops:
        return empty
    joins = []
    for a, b in zip(path, path[1:]):
        rel = g[a][b]["rel"]
        if rel["from_table"] == a:
            from_t, from_c = rel["from_table"], rel["from_column"]
            to_t, to_c = rel["to_table"], rel["to_column"]
        else:
            from_t, from_c = rel["to_table"], rel["to_column"]
            to_t, to_c = rel["from_table"], rel["from_column"]
        joins.append({
            "from_table": from_t,
            "from_column": from_c,
            "to_table": to_t,
            "to_column": to_c,
            "join_hint": f"{from_t}.{from_c} = {to_t}.{to_c}",
        })
    return {"found": True, "path": path, "joins": joins, "hops": hops}


@mcp.tool()
def find_column_for_concept_server(concept: str, db_name: str,
                            tables: list[str] | None = None,
                            threshold: float = 0.6,
                            top_n: int = 5,
                            backend: str = "mysql") -> dict:
    """
    Busca columnas semanticamente compatibles con un concepto del usuario.

    Calcula el embedding del concepto y lo compara contra los embeddings de
    las descripciones de columna ya indexadas por el APS en ChromaDB.
    Devuelve las columnas mas similares y un flag de compatibilidad basado
    en el umbral provisto. Es deterministico: no usa LLM.

    Util para que el APS decida si un filtro del usuario tiene una columna
    razonable donde caer en el schema, sin recurrir a un LLM con prompt
    hardcodeado.

    Args:
        concept: termino del usuario a verificar (ej. valor de filtro).
        db_name: identificador de la base de datos.
        tables: si se pasa, restringe la busqueda a columnas de esas tablas.
        threshold: similitud coseno minima para considerar compatible.
        top_n: maximo numero de columnas a retornar.

    Returns:
        Dict con:
          - is_compatible: bool, True si hay al menos una columna con
                           similitud >= threshold.
          - best_similarity: similitud de la mejor coincidencia (0.0 si vacio).
          - matches: lista de matches ordenada por similitud descendente,
                     cada uno con table, column y similarity.
    """
    import chromadb
    from chromadb.config import Settings

    vector_db_path = resolve_vector_db_path(db_name, backend)
    if not vector_db_path.exists():
        return {"is_compatible": False, "best_similarity": 0.0, "matches": []}

    try:
        client = chromadb.PersistentClient(
            path=str(vector_db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection("columns")
    except Exception:
        return {"is_compatible": False, "best_similarity": 0.0, "matches": []}

    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from utils.embeddings import get_embedding
        try:
            from config import EMBEDDING_MODEL
        except Exception:
            EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
        emb = get_embedding(concept, EMBEDDING_MODEL, mode="query")
    except Exception:
        return {"is_compatible": False, "best_similarity": 0.0, "matches": []}

    where = None
    if tables:
        where = {"table_name": {"$in": list(tables)}}

    try:
        n = min(max(top_n, 1), max(collection.count(), 1))
        results = collection.query(
            query_embeddings=[emb], n_results=n, where=where,
        )
    except Exception:
        return {"is_compatible": False, "best_similarity": 0.0, "matches": []}

    matches = []
    if results and results.get("ids") and results["ids"][0]:
        for i, _ in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            dist = results["distances"][0][i] if results.get("distances") else 1.0
            sim = max(0.0, 1.0 - dist)
            if sim >= threshold:
                matches.append({
                    "table": meta.get("table_name", ""),
                    "column": meta.get("column_name", ""),
                    "similarity": round(sim, 3),
                })

    matches.sort(key=lambda m: m["similarity"], reverse=True)
    return {
        "is_compatible": bool(matches),
        "best_similarity": matches[0]["similarity"] if matches else 0.0,
        "matches": matches,
    }
