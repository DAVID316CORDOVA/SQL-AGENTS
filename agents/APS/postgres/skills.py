"""
agents/APS/postgres/skills.py

Skills del APS para PostgreSQL.
  - format_identifier : escapa nombres de tabla/columna con comillas dobles PostgreSQL
  - get_join_hint     : formatea un JOIN hint con sintaxis PostgreSQL
"""

PG_RESERVED = {
    "order", "group", "select", "from", "where", "join", "table",
    "column", "index", "by", "having", "limit", "offset",
    "insert", "update", "delete", "create", "drop", "alter",
    "values", "set", "into", "exists", "in", "not", "and", "or",
    "between", "like", "null", "is", "case", "when", "then", "else",
    "end", "distinct", "count", "sum", "avg", "min", "max",
    "user", "session", "natural", "full", "cross", "inner", "outer",
}


def format_identifier(name: str) -> str:
    """Escapa un nombre de tabla o columna con comillas dobles si es reservado en PostgreSQL."""
    if name.lower() in PG_RESERVED or "-" in name or " " in name:
        return f'"{name}"'
    return name


def get_join_hint(from_table: str, from_col: str, to_table: str, to_col: str) -> str:
    """Devuelve el hint de JOIN con sintaxis PostgreSQL (comillas dobles para nombres reservados)."""
    ft = format_identifier(from_table)
    fc = format_identifier(from_col)
    tt = format_identifier(to_table)
    tc = format_identifier(to_col)
    return f"JOIN {tt} ON {ft}.{fc} = {tt}.{tc}"
