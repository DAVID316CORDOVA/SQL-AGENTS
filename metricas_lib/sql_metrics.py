# -*- coding: utf-8 -*-
"""
metricas_lib/sql_metrics.py

Metricas para evaluacion de SQL generado por el agente AG.

Las salidas SQL son texto, asi que en principio se les puede aplicar
ROUGE-L directamente. Pero ROUGE-L sin normalizacion penaliza diferencias
de formato (mayusculas, whitespace, comillas, aliases) que NO afectan la
semantica de la query. Esta diferencia ensucia la comparacion entre el
generated_sql del AG y el expected_sql del dataset.

Por eso este modulo expone:

  - normalize_sql()  : normaliza una cadena SQL para comparacion lexical:
                       lowercase, whitespace colapsado, operadores espaciados,
                       comillas unificadas, RESOLUCION DE ALIASES de tabla
                       (T1.name == singer.name == s.name) y
                       NORMALIZACION DE ORDEN DE JOIN (FROM t1 JOIN t2 ==
                       FROM t2 JOIN t1 para INNER JOIN conmutativo).

  - rouge_l_sql()    : ROUGE-L F1 entre dos cadenas SQL APLICANDO la
                       normalizacion arriba. Determinista, sin LLM-juez.

Justificacion academica: Spider y BIRD aplican normalizacion ortografica
similar antes de cualquier metrica lexical sobre SQL. Este modulo sigue
esa practica estandar. La resolucion de aliases es necesaria porque los
modelos LLM usan alias distintos (T1, s, singer) para la misma tabla,
lo cual penalizaria injustamente a SQLs semanticamente equivalentes.
La normalizacion de JOIN es necesaria porque INNER JOIN es conmutativo:
FROM concert JOIN stadium == FROM stadium JOIN concert.
"""

import re
from metricas_lib.deepeval_metrics import rouge_l_score

# ── Palabras reservadas SQL — nunca se tratan como nombre de tabla o alias ──
_SQL_KEYWORDS = frozenset({
    'select', 'from', 'where', 'join', 'on', 'group', 'order',
    'having', 'limit', 'and', 'or', 'not', 'in', 'between',
    'like', 'null', 'is', 'by', 'asc', 'desc', 'distinct',
    'inner', 'outer', 'left', 'right', 'full', 'cross', 'natural',
    'case', 'when', 'then', 'else', 'end', 'union', 'all', 'any',
    'exists', 'count', 'sum', 'avg', 'max', 'min', 'as', 'into',
    'set', 'update', 'delete', 'insert', 'values', 'with', 'top',
    'cast', 'convert', 'double', 'int', 'varchar', 'char', 'float',
    'using', 'offset', 'fetch', 'over', 'partition', 'row_number',
    'rank', 'dense_rank', 'coalesce', 'ifnull', 'isnull', 'nullif',
    'intersect', 'except', 'minus',
})


# ===========================================================================
# CODIGO ORIGINAL — conservado como referencia, reemplazado por la version
# mejorada que agrega normalizacion de orden de JOIN.
# ===========================================================================

def _resolve_aliases_v1(sql: str) -> str:
    """Version original de resolucion de aliases — usada por Fase 1.
    Sin normalizacion de orden de JOIN.
    """
    s = sql
    alias_to_table: dict[str, str] = {}
    for m in re.finditer(r'\b(\w+)\s+as\s+(\w+)\b', s):
        table, alias = m.group(1), m.group(2)
        if table not in _SQL_KEYWORDS and alias not in _SQL_KEYWORDS:
            alias_to_table[alias] = table
    for m in re.finditer(r'(?:from|join)\s+(\w+)\s+(\w+)\b(?!\s*\.)', s):
        table, alias = m.group(1), m.group(2)
        if table not in _SQL_KEYWORDS and alias not in _SQL_KEYWORDS:
            alias_to_table.setdefault(alias, table)
    for m in re.finditer(r'\)\s+(\w+)\b', s):
        alias = m.group(1)
        if alias not in _SQL_KEYWORDS:
            alias_to_table[alias] = 'subquery'
    if alias_to_table:
        def replace_ref(m: re.Match) -> str:
            prefix = m.group(1)
            col    = m.group(2)
            if prefix in alias_to_table:
                return f"{alias_to_table[prefix]}.{col}"
            return m.group(0)
        s = re.sub(r'\b(\w+)\.(\w+)\b', replace_ref, s)
    s = re.sub(r'\s+as\s+\w+\b', '', s)
    for alias in alias_to_table:
        s = re.sub(
            r'((?:from|join)\s+\w+)\s+' + re.escape(alias) + r'\b(?!\s*\.)',
            r'\1', s,
        )
    for alias, mapped in alias_to_table.items():
        if mapped == 'subquery':
            s = re.sub(r'\)\s+' + re.escape(alias) + r'\b', ')', s)
    s = re.sub(r'\s+,', ',', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_sql_v1(sql: str) -> str:
    """Normalizacion ORIGINAL — Fase 1 (sin correccion de orden de JOINs).
    Usada en los experimentos de Fase 1 (phase_1).
    No corrige el orden de FROM t1 JOIN t2, por lo que SQLs con el mismo
    significado pero distinto orden de JOIN obtienen ROUGE-L < 1.0.
    """
    if not sql:
        return ""
    s = sql.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([=<>])\s*", r" \1 ", s)
    s = re.sub(r"!\s*=", "!=", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace('"', "'")
    s = _resolve_aliases_v1(s)
    return s


# ===========================================================================
# VERSION MEJORADA — agrega normalizacion de orden de JOIN (v2)
# ===========================================================================

def _resolve_aliases(sql: str) -> str:
    """
    Resuelve aliases de tabla en el SQL para una comparacion lexical justa.

    Problema que resuelve: dos SQLs semanticamente identicos pueden obtener
    ROUGE-L < 1.0 si usan aliases distintos para las mismas tablas.
    Ejemplo: "singer AS T1, T1.age" vs "singer AS s, s.age" → tokens diferentes
    pero mismo significado. Esta funcion los unifica: ambos quedan como "singer.age".

    Tres tipos de alias que se detectan y eliminan:
      1a. Explicito:  tabla AS alias  →  alias_to_table[alias] = tabla
      1b. Implicito:  FROM tabla alias (sin AS)  →  igual que 1a
      1c. Subquery:   (...) alias  →  alias_to_table[alias] = 'subquery'

    Recibe SQL ya en lowercase (producido por normalize_sql antes de llamar).
    """
    s = sql

    alias_to_table: dict[str, str] = {}

    # Paso 1a: alias explicito "tabla AS alias" — el mas comun en SQLs generados por LLMs
    for m in re.finditer(r'\b(\w+)\s+as\s+(\w+)\b', s):
        table, alias = m.group(1), m.group(2)
        if table not in _SQL_KEYWORDS and alias not in _SQL_KEYWORDS:
            alias_to_table[alias] = table

    # Paso 1b: alias implicito "FROM tabla alias" o "JOIN tabla alias"
    # El alias no va seguido de '.' para no confundirlo con "tabla.columna"
    for m in re.finditer(r'(?:from|join)\s+(\w+)\s+(\w+)\b(?!\s*\.)', s):
        table, alias = m.group(1), m.group(2)
        if table not in _SQL_KEYWORDS and alias not in _SQL_KEYWORDS:
            alias_to_table.setdefault(alias, table)   # el explicito tiene prioridad

    # Paso 1c: alias de subquery "(...) alias" — normalizar al centinela 'subquery'
    # para que todos los alias de subquery produzcan el mismo token
    for m in re.finditer(r'\)\s+(\w+)\b', s):
        alias = m.group(1)
        if alias not in _SQL_KEYWORDS:
            alias_to_table[alias] = 'subquery'

    # Paso 2: reemplazar "alias.columna" por "tabla.columna" en todo el SQL
    if alias_to_table:
        def replace_ref(m: re.Match) -> str:
            prefix = m.group(1)
            col    = m.group(2)
            if prefix in alias_to_table:
                return f"{alias_to_table[prefix]}.{col}"
            return m.group(0)

        s = re.sub(r'\b(\w+)\.(\w+)\b', replace_ref, s)

    # Paso 3a: eliminar "AS alias" en cualquier posicion
    s = re.sub(r'\s+as\s+\w+\b', '', s)

    # Paso 3b: eliminar alias implicito de tabla "FROM tabla alias" → "FROM tabla"
    for alias in alias_to_table:
        s = re.sub(
            r'((?:from|join)\s+\w+)\s+' + re.escape(alias) + r'\b(?!\s*\.)',
            r'\1',
            s,
        )

    # Paso 3c: eliminar alias de subquery ") alias" → ")"
    for alias, mapped in alias_to_table.items():
        if mapped == 'subquery':
            s = re.sub(r'\)\s+' + re.escape(alias) + r'\b', ')', s)

    # Limpiar artefactos de formato
    s = re.sub(r'\s+,', ',', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _normalize_join_order(sql: str) -> str:
    """
    Canonicaliza el orden de tablas en FROM...JOIN para que INNER JOINs sean comparables
    sin importar el orden en que el LLM los escribio.

    Motivacion: INNER JOIN es conmutativo matematicamente pero no lexicalmente.
    El gold SQL puede tener "FROM concert JOIN stadium" y el generado tener
    "FROM stadium JOIN concert". Son equivalentes pero ROUGE-L los penaliza
    (baja de ~0.84 a ~0.68). Esta funcion los normaliza a orden alfabetico.

    Estrategia de canonicalizacion:
      1. Para cada par FROM t1 JOIN t2: si t1 > t2 (alfabetico), intercambiar.
         Resultado canonico: FROM menor JOIN mayor
      2. Para cada ON c1 = c2: si c1 > c2, intercambiar.
         Resultado canonico: ON menor = mayor

    Solo aplica a JOINs simples (dos tablas, una condicion ON).
    Recibe SQL ya en lowercase y con aliases resueltos.
    """
    s = sql  # copia local

    # ── Paso A: normalizar orden de las dos tablas en FROM t1 JOIN t2 ─────
    #    Patron: "from <tabla1> join <tabla2>"
    #    Restriccion: solo captura JOINs simples (sin INNER/LEFT/etc. explícito
    #    ya que en la normalizacion previa 'inner' se habria conservado como keyword)
    def _swap_from(m: re.Match) -> str:
        t1 = m.group(1)   # primera tabla (en FROM)
        t2 = m.group(2)   # segunda tabla (despues de JOIN)
        if t1 > t2:       # t1 va despues alfabeticamente → intercambiar
            return f'from {t2} join {t1}'
        return m.group(0) # ya esta en orden correcto, no tocar

    # aplicar el swap sobre todos los pares FROM...JOIN del SQL
    s = re.sub(r'from\s+(\w+)\s+join\s+(\w+)', _swap_from, s)

    # ── Paso B: normalizar orden dentro de cada condicion ON c1 = c2 ──────
    #    Patron: "on <expr1> = <expr2>"   (solo '=', no '<' ni '>')
    #    Expresion puede ser tabla.columna o solo columna
    def _swap_on(m: re.Match) -> str:
        c1 = m.group(1)   # lado izquierdo de la igualdad
        c2 = m.group(2)   # lado derecho de la igualdad
        if c1 > c2:       # c1 va despues alfabeticamente → intercambiar
            return f'on {c2} = {c1}'
        return m.group(0) # ya esta en orden correcto, no tocar

    # aplicar solo al primer ON despues de cada JOIN (evita WHERE y HAVING)
    s = re.sub(r'\bon\s+(\S+)\s* = \s*(\S+)', _swap_on, s)

    return s


def normalize_sql(sql: str) -> str:
    """
    Normaliza una cadena SQL para que la comparacion lexical mida contenido
    semantico y no diferencias de formato irrelevantes. Aplica en orden:

      Paso 1 — lowercase            :  SELECT  →  select
      Paso 2 — whitespace colapsado :  newlines, tabs, multi-spaces  →  1 espacio
      Paso 3 — espacios operadores  :  id=1  →  id = 1 ,  x<3  →  x < 3
      Paso 4 — rejuntar !=          :  !  =  →  !=  (roto por el paso anterior)
      Paso 5 — colapsar espacios    :  limpia dobles espacios del paso 3
      Paso 6 — comillas unificadas  :  "France"  →  'France'
      Paso 7 — resolver aliases     :  T1.name / s.name  →  singer.name
      Paso 8 — normalizar JOIN      :  FROM stadium JOIN concert
                                       →  FROM concert JOIN stadium  (alfabetico)
                                       ON stadium.id = concert.id
                                       →  ON concert.id = stadium.id

    Args:
        sql: cadena SQL cruda (puede tener saltos de linea, mayusculas, etc.)

    Returns:
        cadena SQL normalizada en una sola linea, comparable con ROUGE-L.
    """
    if not sql:                                      # guard: string vacio → devolver vacio
        return ""
    s = sql.lower().strip()                          # paso 1: todo a minusculas
    s = re.sub(r"\s+", " ", s)                       # paso 2: cualquier whitespace → 1 espacio
    s = re.sub(r"\s*([=<>])\s*", r" \1 ", s)         # paso 3: espaciar operadores = < >
    s = re.sub(r"!\s*=", "!=", s)                    # paso 4: rejuntar != separado por paso 3
    s = re.sub(r"\s+", " ", s).strip()               # paso 5: colapsar espacios extra del paso 3
    s = s.replace('"', "'")                          # paso 6: unificar comillas dobles → simples
    s = _resolve_aliases(s)                           # paso 7: T1.name → singer.name, etc.
    s = _normalize_join_order(s)                      # paso 8: FROM t1 JOIN t2 → orden canonico
    return s


def rouge_l_sql_v1(generated_sql: str, expected_sql: str) -> float:
    """ROUGE-L con normalize_sql_v1 — version ORIGINAL usada en Fase 1.
    No corrige el orden de JOINs. Usada historicamente en phase_1/main.py.
    """
    if not generated_sql or not expected_sql:
        return 0.0
    return rouge_l_score(
        actual_output=normalize_sql_v1(generated_sql),
        expected_output=normalize_sql_v1(expected_sql),
    )


def rouge_l_sql_v2(generated_sql: str, expected_sql: str) -> float:
    """ROUGE-L con normalize_sql (v2) — version CORREGIDA usada en Fase 1.2.
    Incluye normalizacion de orden de JOINs (INNER JOIN es conmutativo).
    FROM t1 JOIN t2  ==  FROM t2 JOIN t1 despues de normalizacion.
    """
    if not generated_sql or not expected_sql:
        return 0.0
    return rouge_l_score(
        actual_output=normalize_sql(generated_sql),
        expected_output=normalize_sql(expected_sql),
    )


# Alias por compatibilidad — rouge_l_sql apunta a la version actual (v2)
def rouge_l_sql(generated_sql: str, expected_sql: str) -> float:
    """ROUGE-L F1 entre dos SQLs con normalizacion completa (v2, version actual).
    Alias de rouge_l_sql_v2. Usada por defecto en phase_1.2 y produccion.
    """
    return rouge_l_sql_v2(generated_sql, expected_sql)
