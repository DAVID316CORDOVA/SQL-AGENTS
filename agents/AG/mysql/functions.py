"""
agents/AG/mysql/functions.py

FUNCTIONS del AG-MySQL (helpers determinísticos, no van al prompt del LLM).

  - find_join_path_skill     : BFS sobre el grafo de FKs entre dos tablas
  - find_joins_among_tables  : todas las relaciones FK directas entre tablas
"""

import re


# ────────────────────────────────────────────────────────────────────
# Helpers de JOIN (responsabilidad migrada del APS al AG)
# ────────────────────────────────────────────────────────────────────

def find_join_path_skill(from_table: str, to_table: str, max_hops: int = 3) -> dict:
    """
    BFS sobre el grafo de FKs del schema.json para encontrar la ruta
    mas corta entre dos tablas (resuelve casos de tabla puente).
    Lee el mismo schema.json que find_joins_among_tables — sin MCP server.
    """
    if not from_table or not to_table:
        return {"found": False, "error": "from_table y to_table son obligatorios"}
    try:
        from agents.MCP._mcp_instance import load_schema
        from config import ACTIVE_DATASET
        schema = load_schema(ACTIVE_DATASET, backend="mysql")
        rels = schema.get("relationships", [])

        # Construir grafo bidireccional {tabla: [(vecino, rel)]}
        graph = {}
        for rel in rels:
            ft, tt = rel["from_table"].lower(), rel["to_table"].lower()
            graph.setdefault(ft, []).append((tt, rel))
            graph.setdefault(tt, []).append((ft, rel))

        src, dst = from_table.lower(), to_table.lower()
        if src not in graph or dst not in graph:
            return {"found": False, "path": [], "joins": [], "hops": 0}
        if src == dst:
            return {"found": True, "path": [from_table], "joins": [], "hops": 0}

        # BFS
        from collections import deque
        queue = deque([[src]])
        visited = {src}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == dst:
                if len(path) - 1 > max_hops:
                    return {"found": False, "path": [], "joins": [], "hops": 0}
                joins = []
                for a, b in zip(path, path[1:]):
                    for neighbor, rel in graph.get(a, []):
                        if neighbor == b:
                            ft = rel["from_table"].lower()
                            joins.append({
                                "from_table":  rel["from_table"] if ft == a else rel["to_table"],
                                "from_column": rel["from_column"] if ft == a else rel["to_column"],
                                "to_table":    rel["to_table"] if ft == a else rel["from_table"],
                                "to_column":   rel["to_column"] if ft == a else rel["from_column"],
                                "join_hint":   rel["join_hint"],
                            })
                            break
                return {"found": True, "path": path, "joins": joins, "hops": len(joins)}
            for neighbor, _ in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return {"found": False, "path": [], "joins": [], "hops": 0}
    except Exception as exc:
        return {"found": False, "error": f"{type(exc).__name__}: {exc}"}


def find_joins_among_tables(tables: list) -> dict:
    """
    Devuelve todas las relaciones FK DIRECTAS que existen entre las
    tablas indicadas, leyendo el schema.json del MCP.

    Util cuando el AG ya tiene un conjunto de tablas relevantes y solo
    necesita saber como unirlas (sin necesidad de BFS).
    """
    if not tables or len(tables) < 2:
        return {"joins": []}
    try:
        from agents.MCP._mcp_instance import load_schema
        from config import ACTIVE_DATASET
        schema = load_schema(ACTIVE_DATASET, backend="mysql")
        wanted = {t.lower() for t in tables}
        joins = []
        for rel in schema.get("relationships", []):
            ft = (rel.get("from_table") or "").lower()
            tt = (rel.get("to_table") or "").lower()
            if ft in wanted and tt in wanted:
                joins.append({
                    "from_table":  rel["from_table"],
                    "from_column": rel["from_column"],
                    "to_table":    rel["to_table"],
                    "to_column":   rel["to_column"],
                    "join_hint":   (
                        f"JOIN {rel['to_table']} ON {rel['from_table']}."
                        f"{rel['from_column']} = {rel['to_table']}."
                        f"{rel['to_column']}"
                    ),
                })
        return {"joins": joins, "n_joins": len(joins)}
    except Exception as exc:
        return {"joins": [], "error": f"{type(exc).__name__}: {exc}"}
