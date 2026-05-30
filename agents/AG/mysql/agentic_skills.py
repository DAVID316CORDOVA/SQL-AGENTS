"""
agents/AG/mysql/agentic_skills.py

AGENTIC SKILLS del AG-MySQL (LLM-invocables via tool calling).

Estos van al prompt como tools disponibles. El LLM decide cuando
invocarlos durante el agentic loop.

  - validate_sql_safety : verifica que no haya DML/DDL, SELECT *, COUNT(*)
  - fix_reserved_words  : agrega backticks a aliases con palabras reservadas
"""

import re

MYSQL_RESERVED_WORDS = [
    'rank', 'dense_rank', 'row_number', 'ntile', 'percent_rank',
    'lead', 'lag', 'first_value', 'last_value',
    'select', 'from', 'where', 'join', 'on', 'and', 'or', 'not', 'in',
    'between', 'like', 'is', 'null', 'as', 'by', 'asc', 'desc', 'limit',
    'case', 'when', 'then', 'else', 'end',
    'index', 'key', 'unique',
    'over', 'partition', 'window', 'rows', 'range',
    'date', 'time', 'timestamp', 'year', 'month', 'day',
    'count', 'sum', 'avg', 'min', 'max', 'group', 'order',
    'first', 'last', 'values', 'value', 'set', 'if',
    'status', 'type', 'level', 'name',
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
    """Agrega backticks a aliases que coinciden con palabras reservadas MySQL."""
    original = sql
    changes = []
    for w in MYSQL_RESERVED_WORDS:
        pattern_inline = rf'(\s+AS\s+)({w})(\s*[,\)\n])'
        pattern_eol = rf'(\s+AS\s+)({w})(\s*)$'
        new_inline = re.sub(pattern_inline, rf'\1`\2`\3', sql, flags=re.IGNORECASE)
        new_eol = re.sub(pattern_eol, rf'\1`\2`\3', new_inline, flags=re.IGNORECASE)
        if new_eol != sql:
            changes.append(f"Backtick agregado a alias `{w}`")
            sql = new_eol
    return {"sql": sql, "original": original, "changes": changes}
