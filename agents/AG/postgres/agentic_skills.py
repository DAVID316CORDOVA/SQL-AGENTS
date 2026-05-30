"""
agents/AG/postgres/agentic_skills.py

AGENTIC SKILLS del AG-PostgreSQL (LLM-invocables via tool calling).

  - validate_sql_safety : verifica que no haya DML/DDL, SELECT *, COUNT(*)
  - fix_reserved_words  : agrega comillas dobles a aliases con palabras
                          reservadas PostgreSQL
"""

import re

POSTGRES_RESERVED_WORDS = [
    'user', 'order', 'group', 'limit', 'offset', 'end', 'start',
    'check', 'table', 'column', 'constraint', 'index', 'position',
    'between', 'case', 'when', 'then', 'else', 'select', 'from',
    'where', 'and', 'or', 'not', 'in', 'like', 'is', 'null', 'as',
    'by', 'asc', 'desc', 'join', 'on', 'into', 'values', 'set',
    'all', 'any', 'some', 'exists', 'window', 'partition', 'over',
    'rows', 'range', 'current', 'following', 'preceding', 'unbounded',
    'intersect', 'except', 'union', 'with', 'recursive',
    'rank', 'dense_rank', 'row_number', 'ntile', 'lag', 'lead',
    'first_value', 'last_value', 'percent_rank', 'cume_dist',
    'name', 'type', 'status', 'level', 'date', 'time', 'timestamp',
    'year', 'month', 'day', 'value', 'values', 'default', 'session',
]


def validate_sql_safety(sql: str) -> dict:
    """Verifica que el SQL no contenga operaciones peligrosas."""
    if not sql:
        return {"valid": False, "reason": "SQL vacio", "violations": [], "warnings": []}
    u = sql.upper()
    violations = []
    warnings = []
    for w in ["DROP ", "DELETE ", "UPDATE ", "INSERT ", "TRUNCATE ", "ALTER ", "CREATE "]:
        if w in u:
            violations.append(f"Contiene {w.strip()} (operacion DML/DDL prohibida)")
    if "SELECT *" in u:
        violations.append("SELECT * prohibido — especificar columnas")
    if "COUNT(*)" in u:
        violations.append("COUNT(*) prohibido — usar COUNT(columna_pk)")
    if violations:
        return {"valid": False, "reason": violations[0], "violations": violations, "warnings": warnings}
    return {"valid": True, "reason": None, "violations": [], "warnings": warnings}


def fix_reserved_words(sql: str) -> dict:
    """Agrega comillas dobles a aliases que coinciden con palabras reservadas PostgreSQL."""
    original = sql
    changes = []
    for w in POSTGRES_RESERVED_WORDS:
        pattern_inline = rf'(\s+AS\s+)({w})(\s*[,\)\n])'
        pattern_eol = rf'(\s+AS\s+)({w})(\s*)$'
        new_inline = re.sub(pattern_inline, r'\1"\2"\3', sql, flags=re.IGNORECASE)
        new_eol = re.sub(pattern_eol, r'\1"\2"\3', new_inline, flags=re.IGNORECASE)
        if new_eol != sql:
            changes.append(f'Comillas dobles agregadas a alias "{w}"')
            sql = new_eol
    return {"sql": sql, "original": original, "changes": changes}
