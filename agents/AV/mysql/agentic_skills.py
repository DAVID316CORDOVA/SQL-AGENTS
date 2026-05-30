"""
agents/AV/mysql/agentic_skills.py

AGENTIC SKILLS del AV-MySQL (LLM-invocables via agentic loop).

Estos son skills que el LLM puede invocar durante _run_agent_loop.
Sus descriptions van al prompt como tools disponibles. El LLM decide
cuando llamarlas basandose en el contexto de la consulta.

  - check_syntax_rules      : reglas de sintaxis MySQL prohibidas
  - check_semantic_patterns : patrones SQL problematicos
  - check_query_efficiency  : analisis estatico de calidad del SQL
                              (producto cartesiano, exceso de JOINs,
                              FULL OUTER, DISTINCT redundante, etc.)
  - check_query_performance : heuristicas de rendimiento sin ejecutar
                              (wildcards, funciones en WHERE, subqueries
                              correlacionadas, tablas grandes sin filtro)
"""

import re


def check_syntax_rules(sql: str) -> dict:
    """Verifica reglas de sintaxis prohibidas en MySQL."""
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

    # Contar JOINs y ONs: si hay mas JOINs que ONs hay al menos uno sin ON.
    # La regex anterior usaba un lookahead que solo detectaba una palabra entre
    # JOIN y ON, generando falsos positivos con tablas de nombre largo o aliases.
    n_joins = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
    n_ons   = len(re.findall(r'\bON\b',   sql, re.IGNORECASE))
    if n_joins > n_ons:
        warnings.append("JOIN sin clausula ON detectado — verificar sintaxis")

    if re.search(r'\bWHERE\b[^;]*\b(ROW_NUMBER|RANK|DENSE_RANK)\b\s*\(', sql, re.IGNORECASE):
        violations.append("Window function en clausula WHERE — usar CTE o subquery")

    return {"violations": violations, "warnings": warnings, "valid": len(violations) == 0}


def check_semantic_patterns(sql: str, query: str) -> dict:
    """Detecta patrones SQL semanticamente problematicos para MySQL."""
    patterns = []
    suggestions = []
    u = sql.upper()

    if "LIMIT" in u and "PARTITION BY" not in u:
        lower_q = query.lower()
        if any(w in lower_q for w in ["por cada", "de cada", "por grupo", "por categoria"]):
            patterns.append("LIMIT global detectado en consulta por grupos — considerar PARTITION BY con ROW_NUMBER")

    if re.search(r'YEAR\s*\(|MONTH\s*\(|DAY\s*\(', sql, re.IGNORECASE):
        suggestions.append("Funciones de fecha en WHERE rompen indices MySQL — usar rangos: WHERE col BETWEEN 'X' AND 'Y'")

    if "HAVING" in u and not re.search(r'HAVING\s+\w*\s*(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql, re.IGNORECASE):
        patterns.append("HAVING usado para filtros no-agregados — mover a WHERE")

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

    # 1. Producto cartesiano: tablas separadas por coma en FROM sin ningun JOIN
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

    # 3. FULL OUTER JOIN (raramente correcto en OLTP)
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
            "reemplazar con IN (...) para mayor claridad y posible optimizacion del plan"
        )

    # 6. Subqueries profundamente anidadas (mas de 2 niveles)
    nesting = sql.upper().count('(SELECT')
    if nesting > 2:
        suggestions.append(
            f"Subquery anidada a {nesting} niveles: considerar CTE (WITH ... AS) "
            "para mejorar legibilidad y facilitar optimizacion"
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

    Predice si una consulta sera lenta basandose en patrones conocidos:
      - Wildcard inicial en LIKE '%...' → no puede usar indice
      - Funcion sobre columna en WHERE  → rompe el indice de esa columna
      - Subquery correlacionada en WHERE → se ejecuta una vez por fila exterior
      - Tabla grande (>10k filas segun APS) sin clausula WHERE → full scan
      - Multiples JOINs sin LIMIT sobre tablas con muchas filas → resultado enorme

    No sustituye a EXPLAIN ANALYZE (trabajo futuro); complementa el analisis
    estatico con estimaciones de riesgo de rendimiento.
    """
    warnings = []
    risk_level = "low"  # low | medium | high
    u = sql.upper()

    # 1. Leading wildcard: LIKE '%texto' — no puede aprovechar B-tree index
    if re.search(r"LIKE\s+'%[^']+", sql, re.IGNORECASE):
        warnings.append(
            "Leading wildcard LIKE '%...' impide uso de indice B-tree — "
            "probable full table scan"
        )
        risk_level = "medium"

    # 2. Funcion sobre columna en WHERE — el optimizador no puede usar el indice
    func_where = re.search(
        r'\bWHERE\b[^;]*\b(YEAR|MONTH|DAY|LOWER|UPPER|LENGTH|ABS|DATE)\s*\(',
        sql, re.IGNORECASE | re.DOTALL
    )
    if func_where:
        warnings.append(
            f"{func_where.group(1).upper()}() aplicada en WHERE sobre columna — "
            "el optimizador no puede usar el indice; usar rango de fechas o valor directo"
        )
        risk_level = "medium"

    # 3. Subquery correlacionada en WHERE — se ejecuta N veces (una por fila externa)
    if re.search(r'\bWHERE\b[^;]*\(SELECT\b', sql, re.IGNORECASE | re.DOTALL):
        warnings.append(
            "Subquery en clausula WHERE puede ser correlacionada — "
            "se ejecutaria una vez por cada fila de la tabla exterior (O(n) queries)"
        )
        if risk_level == "low":
            risk_level = "medium"

    # 4. Tabla grande sin WHERE — full scan sobre tabla con muchas filas
    if table_row_counts:
        large = [t for t, n in table_row_counts.items() if isinstance(n, int) and n > 10000]
        if large and 'WHERE' not in u:
            warnings.append(
                f"Tablas con >10k filas ({', '.join(large)}) sin clausula WHERE — "
                "full table scan esperado; considerar agregar filtro o LIMIT"
            )
            risk_level = "high"

    # 5. Multiples JOINs sin LIMIT — el resultado puede crecer exponencialmente
    join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
    if join_count >= 3 and 'LIMIT' not in u:
        warnings.append(
            f"{join_count} JOINs sin LIMIT — el resultado puede ser muy grande "
            "si las tablas tienen muchas filas; agregar LIMIT o revisar si todos los JOINs son necesarios"
        )
        if risk_level == "low":
            risk_level = "medium"

    return {
        "risk_level": risk_level,           # low | medium | high
        "slow_query_predicted": risk_level in ("medium", "high"),
        "warnings": warnings,
        "join_count": join_count,
        "note": "Analisis heuristico sin ejecucion. Para tiempos exactos usar EXPLAIN ANALYZE (trabajo futuro).",
    }
