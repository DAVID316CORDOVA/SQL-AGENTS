"""
agents/AV/postgres/tools.py

TOOLS del AV-PostgreSQL (LLM-invocables via agentic loop).
"""

import re


def check_syntax_rules(sql: str) -> dict:
    """Verifica reglas de sintaxis prohibidas en PostgreSQL."""
    violations = []
    warnings = []
    u = sql.upper()

    if "SELECT *" in u:
        violations.append("SELECT * prohibido — especificar columnas")
    if "COUNT(*)" in u:
        violations.append("COUNT(*) prohibido — usar COUNT(columna_pk)")
    for dml in ["DROP ", "DELETE ", "UPDATE ", "INSERT ", "TRUNCATE ", "ALTER "]:
        if dml in u:
            violations.append(f"{dml.strip()} prohibido — solo SELECT")

    if "GROUP_CONCAT" in u:
        violations.append("GROUP_CONCAT no existe en PostgreSQL — usar STRING_AGG(col, ',')")

    if re.search(r'\bAUTO_INCREMENT\b', sql, re.IGNORECASE):
        violations.append("AUTO_INCREMENT no existe en PostgreSQL — usar SERIAL o GENERATED ALWAYS AS IDENTITY")

    join_pattern = re.findall(r'\bJOIN\b(?!\s+USING)(?!\s+\w+\s+ON)', sql, re.IGNORECASE)
    if join_pattern:
        warnings.append("JOIN sin clausula ON detectado — verificar sintaxis")

    if re.search(r'\bWHERE\b[^;]*\b(ROW_NUMBER|RANK|DENSE_RANK)\b\s*\(', sql, re.IGNORECASE):
        violations.append("Window function en clausula WHERE — usar CTE o subquery")

    return {"violations": violations, "warnings": warnings, "valid": len(violations) == 0}


def check_semantic_patterns(sql: str, query: str) -> dict:
    """Detecta patrones SQL semanticamente problematicos para PostgreSQL."""
    patterns = []
    suggestions = []
    u = sql.upper()

    if "LIMIT" in u and "PARTITION BY" not in u:
        lower_q = query.lower()
        if any(w in lower_q for w in ["por cada", "de cada", "por grupo", "por categoria"]):
            patterns.append("LIMIT global detectado en consulta por grupos — considerar PARTITION BY con ROW_NUMBER")

    if re.search(r'EXTRACT\s*\(\s*(YEAR|MONTH|DAY)\s+FROM', sql, re.IGNORECASE):
        suggestions.append("EXTRACT en WHERE rompe indices PostgreSQL — usar rangos: WHERE col >= 'X' AND col < 'Y'")

    if "HAVING" in u and not re.search(r'HAVING\s+\w*\s*(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql, re.IGNORECASE):
        patterns.append("HAVING usado para filtros no-agregados — mover a WHERE")

    if re.search(r'\bLIKE\b', sql, re.IGNORECASE) and 'ILIKE' not in u:
        suggestions.append("Considerar ILIKE en lugar de LIKE para busquedas case-insensitive en PostgreSQL")

    return {"patterns": patterns, "suggestions": suggestions}


def check_query_efficiency(sql: str) -> dict:
    """
    Analisis estatico de calidad del SQL sin conectarse a la base de datos.

    Detecta patrones que indican consultas incorrectas o ineficientes:
      - Producto cartesiano: multiples tablas en FROM sin clausula JOIN ON
      - Exceso de JOINs: mas de 4 JOINs para un esquema tipicamente pequeño
      - FULL OUTER JOIN cuando LEFT/RIGHT JOIN seria suficiente
      - DISTINCT redundante cuando GROUP BY ya garantiza unicidad
      - Cadenas OR sobre la misma columna (mejor usar IN)
      - Subqueries anidadas en mas de 2 niveles (preferir CTE)
    """
    issues = []
    suggestions = []
    u = sql.upper()

    # 1. Producto cartesiano
    from_match = re.search(
        r'\bFROM\b(.*?)(?:\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|$)',
        sql, re.IGNORECASE | re.DOTALL
    )
    if from_match:
        from_clause = from_match.group(1)
        comma_tables = len(re.findall(r',', from_clause.split('(')[0]))
        join_count_in_from = len(re.findall(r'\bJOIN\b', from_clause, re.IGNORECASE))
        if comma_tables > 0 and join_count_in_from == 0:
            issues.append(
                "Producto cartesiano detectado: multiples tablas en FROM sin clausula JOIN ON — "
                "reemplazar comas por JOIN ... ON ..."
            )

    # 2. Exceso de JOINs
    join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
    if join_count > 4:
        issues.append(
            f"Exceso de JOINs ({join_count}): revisar si todas las tablas son necesarias "
            "o si el esquema sugiere una ruta mas directa"
        )

    # 3. FULL OUTER JOIN
    if re.search(r'\bFULL\s+(OUTER\s+)?JOIN\b', sql, re.IGNORECASE):
        suggestions.append(
            "FULL OUTER JOIN detectado: verificar si LEFT JOIN o RIGHT JOIN es suficiente "
            "segun la logica de negocio — FULL OUTER devuelve filas sin par en ambos lados"
        )

    # 4. DISTINCT redundante con GROUP BY
    if re.search(r'\bSELECT\s+DISTINCT\b', sql, re.IGNORECASE) and 'GROUP BY' in u:
        suggestions.append(
            "DISTINCT redundante: GROUP BY ya garantiza unicidad de grupos — "
            "eliminar DISTINCT para evitar ordenacion adicional innecesaria"
        )

    # 5. Cadena OR sobre la misma columna -> usar IN
    or_same_col = re.findall(
        r'(\b\w+\b)\s*=\s*(?:\'[^\']*\'|"[^"]*"|\w+)\s+OR\s+\1\s*=',
        sql, re.IGNORECASE
    )
    if or_same_col:
        suggestions.append(
            f"Multiples condiciones OR sobre '{or_same_col[0]}': "
            "reemplazar con IN (...) para mayor claridad"
        )

    # 6. Subqueries profundamente anidadas
    nesting = sql.upper().count('(SELECT')
    if nesting > 2:
        suggestions.append(
            f"Subquery anidada a {nesting} niveles: considerar CTE (WITH ... AS) "
            "para mejorar legibilidad"
        )

    return {
        "executed": True,
        "static_analysis": True,
        "issues": issues,
        "suggestions": suggestions,
        "needs_optimization": len(issues) > 0,
        "join_count": join_count,
        "has_full_scan": False,
        "analyze_executed": False,
        "actual_time_ms": None,
    }


def check_query_performance(sql: str, table_row_counts: dict = None) -> dict:
    """
    Heuristicas de rendimiento sin ejecutar la query ni conectarse a la BD.
    Version PostgreSQL: considera ILIKE, EXTRACT, LIKE sobre texto y ::cast.
    """
    warnings = []
    risk_level = "low"
    u = sql.upper()

    # 1. Leading wildcard en LIKE / ILIKE
    if re.search(r"(LIKE|ILIKE)\s+'%[^']+", sql, re.IGNORECASE):
        warnings.append(
            "Leading wildcard LIKE/ILIKE '%...' impide uso de indice B-tree — "
            "considerar indice GIN/pg_trgm para busquedas de texto"
        )
        risk_level = "medium"

    # 2. Funcion o EXTRACT sobre columna en WHERE
    func_where = re.search(
        r'\bWHERE\b[^;]*\b(EXTRACT|DATE_TRUNC|LOWER|UPPER|LENGTH|ABS)\s*[\(\s]',
        sql, re.IGNORECASE | re.DOTALL
    )
    if func_where:
        warnings.append(
            f"{func_where.group(1).upper()} aplicada en WHERE — "
            "el planificador de PostgreSQL no puede usar el indice; usar rango de fechas"
        )
        risk_level = "medium"

    # 3. Subquery correlacionada en WHERE
    if re.search(r'\bWHERE\b[^;]*\(SELECT\b', sql, re.IGNORECASE | re.DOTALL):
        warnings.append(
            "Subquery en WHERE puede ser correlacionada — "
            "se ejecutaria una vez por cada fila de la tabla exterior"
        )
        if risk_level == "low":
            risk_level = "medium"

    # 4. Tabla grande sin WHERE
    if table_row_counts:
        large = [t for t, n in table_row_counts.items() if isinstance(n, int) and n > 10000]
        if large and 'WHERE' not in u:
            warnings.append(
                f"Tablas con >10k filas ({', '.join(large)}) sin WHERE — "
                "Seq Scan esperado; considerar filtro o LIMIT"
            )
            risk_level = "high"

    # 5. Multiples JOINs sin LIMIT
    join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
    if join_count >= 3 and 'LIMIT' not in u:
        warnings.append(
            f"{join_count} JOINs sin LIMIT — resultado puede ser muy grande"
        )
        if risk_level == "low":
            risk_level = "medium"

    return {
        "risk_level": risk_level,
        "slow_query_predicted": risk_level in ("medium", "high"),
        "warnings": warnings,
        "join_count": join_count,
        "note": "Analisis heuristico sin ejecucion. Para tiempos exactos usar EXPLAIN ANALYZE (trabajo futuro).",
    }
