"""
agents/AE/skills.py

Skills del AE - Agente Explicador (compartido MySQL/PostgreSQL).
  - format_sql_readable : formatea SQL con saltos de linea para mostrar al usuario
"""

import re


def format_sql_readable(sql: str) -> dict:
    """Formatea el SQL con saltos de linea para mejor legibilidad."""
    keywords = [
        "SELECT", "FROM", "WHERE", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN",
        "JOIN", "ON", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "WITH",
    ]
    result = sql.strip()
    for kw in keywords:
        result = re.sub(
            rf'\b({kw})\b',
            rf'\n{kw}',
            result,
            flags=re.IGNORECASE
        )
    return {"sql_formatted": result.strip()}
