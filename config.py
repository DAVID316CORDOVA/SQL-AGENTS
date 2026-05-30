############

"""
config.py

Configuracion global del sistema NL->SQL Multi-Agente.
TODOS los parametros configurables estan aqui.

Ubicacion: config.py (raiz)
"""

import os

# ============================================================
# PROVEEDOR LLM - "openai" (default) o "anthropic"
# ============================================================
LLM_PROVIDER = "openai"

# ============================================================
# API KEYS
# ============================================================
# Las claves se leen de variables de entorno (cargadas desde .env por
# python-dotenv en cada entry point). NO commitear claves hardcodeadas.
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Configurar en environment para que todos los agentes la lean.
# Solo sobreescribe si no hay ya una clave valida en el entorno.
if LLM_PROVIDER == "openai":
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# ============================================================
# MODELOS POR AGENTE (centralizado)
# Se usan los modelos del proveedor activo.
#
# OpenAI:    gpt-4o-mini (rapido/barato), gpt-4o (potente)
# Anthropic: claude-haiku-4-5 (rapido),   claude-sonnet-4-6 / claude-opus-4-6 (potente)
# ============================================================
if LLM_PROVIDER == "openai":
    AGENT_MODELS = {
        "AR": "gpt-4o-mini",
        "APS": "gpt-4o-mini",
        "AG": "gpt-4o",
        "AV": "gpt-4o",
        "AE": "gpt-4o",
        "AS": "gpt-4o",
        "classifier": "gpt-4o-mini",
        "enricher": "gpt-4o-mini",
    }
    DEFAULT_MODEL = "gpt-4o"
else:
    AGENT_MODELS = {
        "AR": "claude-haiku-4-5",
        "APS": "claude-haiku-4-5",
        "AG": "claude-sonnet-4-6",
        "AV": "claude-sonnet-4-6",
        "AE": "claude-sonnet-4-6",
        "AS": "claude-sonnet-4-6",
        "classifier": "claude-haiku-4-5",
        "enricher": "claude-haiku-4-5",
    }
    DEFAULT_MODEL = "claude-opus-4-6"

# ============================================================
# OVERRIDES CON GANADORES OPTUNA (auto-load desde experiments/winners/best_hyperparameters.json)
# ============================================================
# Si existe best_hyperparameters.json, sobrescribe AGENT_MODELS con los ganadores Optuna y
# expone AGENT_TEMPERATURES por agente. Si no existe (o falla la carga), se
# usan los defaults hardcodeados arriba.
#
# El JSON v2.0 incluye una clave "lookup" con variantes por (backend, dataset),
# consumida via get_agent_model() / get_agent_temperature() / get_aps_config().
AGENT_TEMPERATURES = {}            # role -> float (vista plana, retrocompat)
_AGENT_LOOKUP: dict = {}           # role -> list[variant dict]; cargado del JSON
_AGENT_TUNING_DATASETS: set = set()  # datasets sobre los que se tuneo

try:
    import json as _json
    from pathlib import Path as _Path
    _BEST_HYPERPARAMS_PATH = _Path(__file__).resolve().parent / "experiments" / "winners" / "best_hyperparameters.json"
    if _BEST_HYPERPARAMS_PATH.exists():
        _w = _json.loads(_BEST_HYPERPARAMS_PATH.read_text(encoding="utf-8"))
        if LLM_PROVIDER == "openai":
            # Vista plana (retrocompat con codigo viejo que lee AGENT_MODELS["AG"])
            for k, v in _w.get("agent_models", {}).items():
                if k in AGENT_MODELS:
                    AGENT_MODELS[k] = v
            AGENT_TEMPERATURES = dict(_w.get("agent_temperatures", {}))
            # Vista detallada por (backend, dataset)
            _AGENT_LOOKUP = dict(_w.get("lookup", {}))
            for variants in _AGENT_LOOKUP.values():
                for variant in variants:
                    if variant.get("dataset"):
                        _AGENT_TUNING_DATASETS.add(variant["dataset"])
except Exception:
    # No bloquear el startup si best_hyperparameters.json esta corrupto o ausente
    pass


def _resolve_agent_variant(role: str, backend: str | None = None,
                            dataset: str | None = None) -> dict:
    """
    Resuelve los hiperparametros optimos para un agente segun backend y dataset.

    El JSON de ganadores Optuna almacena multiples variantes por agente (una por
    cada combinacion backend/dataset tuneada). Esta funcion aplica una estrategia
    de fallback en cascada para garantizar que siempre se devuelva algo util:

      1. Match exacto (backend X, dataset Y) → ganador especifico del experimento
      2. Solo backend → ganador del backend, independiente del dataset
      3. Solo dataset → ganador del dataset, independiente del backend
      4. Cualquier variante → la de mayor combined_score
      5. {} → el agente no fue tuneado, usar defaults hardcodeados

    El fallback garantiza que la app funcione con datasets no tuneados, usando
    los hiperparametros del backend correcto como mejor aproximacion.
    """
    variants = _AGENT_LOOKUP.get(role, [])
    if not variants:
        return {}

    backend = (backend or "").lower() or None

    # Iterar sobre las variantes disponibles en orden de especificidad
    if backend and dataset:
        for v in variants:
            if v.get("backend") == backend and v.get("dataset") == dataset:
                return v
    if backend:
        for v in variants:
            if v.get("backend") == backend:
                return v
    if dataset:
        for v in variants:
            if v.get("dataset") == dataset:
                return v
    # Fallback final: la variante con mejor combined_score de todas las disponibles
    return max(variants, key=lambda v: float(v.get("combined_score", 0)))


def get_agent_model(role: str, backend: str | None = None,
                     dataset: str | None = None) -> str:
    """
    Resuelve el modelo a usar para un agente, opcionalmente discriminando
    por backend y dataset.

    Si no hay informacion en el lookup, cae al AGENT_MODELS plano (que ya
    lleva los defaults o el ganador agregado).
    """
    variant = _resolve_agent_variant(role, backend, dataset or ACTIVE_DATASET)
    if "model" in variant:
        return variant["model"]
    return AGENT_MODELS.get(role, DEFAULT_MODEL)


def get_agent_temperature(role: str, backend: str | None = None,
                            dataset: str | None = None,
                            default: float = 0.1) -> float:
    """Idem get_agent_model pero para la temperatura."""
    variant = _resolve_agent_variant(role, backend, dataset or ACTIVE_DATASET)
    if "temperature" in variant:
        return float(variant["temperature"])
    if role in AGENT_TEMPERATURES:
        return float(AGENT_TEMPERATURES[role])
    return default


def get_aps_config(backend: str = "mysql", dataset: str | None = None) -> dict:
    """
    Devuelve la configuracion de retrieval del APS para (backend, dataset):
      similarity_metric, embedding_model, top_n_tables, top_n_columns.

    Si no hay match, retorna defaults razonables.
    """
    variant = _resolve_agent_variant("APS", backend, dataset or ACTIVE_DATASET)
    return {
        "similarity_metric": variant.get("similarity_metric", "cosine"),
        "embedding_model":   variant.get("embedding_model", EMBEDDING_MODEL),
        "top_n_tables":      int(variant.get("top_n_tables", 5)),
        "top_n_columns":     int(variant.get("top_n_columns", 10)),
    }


def is_dataset_tuned(dataset: str | None = None) -> bool:
    """
    Retorna True si el dataset indicado tiene hyperparams tuneados.
    Util para que el orquestador advierta cuando se ejecuta sobre un
    dataset NO tuneado (los hyperparams se aplican igual, pero sin
    garantia empirica de generalizacion).
    """
    return (dataset or ACTIVE_DATASET) in _AGENT_TUNING_DATASETS

# Modelo de embeddings local (sentence-transformers, soporte espanol)
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
 
# ============================================================
# DATASET ACTIVO
# Un solo parametro controla todos los paths y la BD.
#
# Opciones:
#   "demo_db"               -> BD propia (escuela)  - MySQL demo_db
#   "spider:<db_id>"        -> Spider dataset, BD especifica
#                              Ej: "spider:museum_visit"
#   "wikisql"               -> WikiSQL dataset
#
# Para experimentos, cambiar ACTIVE_DATASET y reiniciar.
# ============================================================
# Se puede sobreescribir via env var: ACTIVE_DATASET=spider:museum_visit python main.py
ACTIVE_DATASET = os.environ.get("ACTIVE_DATASET", "demo_db")

# Resolucion automatica de paths segun dataset.
# Metadatas (DDL+diccionario) -> agents/MCP/metadata/<backend>/<family>/<db>/
# Embeddings APS (Chroma)     -> agents/APS/<backend>/chroma/<family>/<db>/
# Cada backend tiene su propia copia para permitir divergencia futura
# (ej. agregar/modificar tablas en uno sin afectar al otro).
_METADATA_BASE = "agents/MCP/metadata"
_APS_CHROMA_BASE_MYSQL = "agents/APS/mysql/chroma"
_APS_CHROMA_BASE_PG    = "agents/APS/postgres/chroma"

if ACTIVE_DATASET.startswith("spider:"):
    _spider_db_id    = ACTIVE_DATASET.split(":", 1)[1]
    SCHEMA_PATH      = f"{_METADATA_BASE}/mysql/spider/{_spider_db_id}/schema.json"
    DICCIONARIO_PATH = f"{_METADATA_BASE}/mysql/spider/{_spider_db_id}/diccionario_datos.json"
    PG_SCHEMA_PATH   = f"{_METADATA_BASE}/postgres/spider/{_spider_db_id}/schema.json"
    PG_DICCIONARIO_PATH = f"{_METADATA_BASE}/postgres/spider/{_spider_db_id}/diccionario_datos.json"
    VECTOR_DB_PATH    = f"{_APS_CHROMA_BASE_MYSQL}/spider/{_spider_db_id}"
    PG_VECTOR_DB_PATH = f"{_APS_CHROMA_BASE_PG}/spider/{_spider_db_id}"
    ACTIVE_DATABASE  = f"spider_{_spider_db_id}"

elif ACTIVE_DATASET.startswith("bird:"):
    _bird_db_id      = ACTIVE_DATASET.split(":", 1)[1]
    SCHEMA_PATH      = f"{_METADATA_BASE}/mysql/bird/{_bird_db_id}/schema.json"
    DICCIONARIO_PATH = f"{_METADATA_BASE}/mysql/bird/{_bird_db_id}/diccionario_datos.json"
    PG_SCHEMA_PATH   = f"{_METADATA_BASE}/postgres/bird/{_bird_db_id}/schema.json"
    PG_DICCIONARIO_PATH = f"{_METADATA_BASE}/postgres/bird/{_bird_db_id}/diccionario_datos.json"
    VECTOR_DB_PATH    = f"{_APS_CHROMA_BASE_MYSQL}/bird/{_bird_db_id}"
    PG_VECTOR_DB_PATH = f"{_APS_CHROMA_BASE_PG}/bird/{_bird_db_id}"
    ACTIVE_DATABASE  = f"bird_{_bird_db_id}"

elif ACTIVE_DATASET == "wikisql":
    SCHEMA_PATH      = f"{_METADATA_BASE}/mysql/wikisql/schema.json"
    DICCIONARIO_PATH = f"{_METADATA_BASE}/mysql/wikisql/diccionario_datos.json"
    PG_SCHEMA_PATH   = f"{_METADATA_BASE}/postgres/wikisql/schema.json"
    PG_DICCIONARIO_PATH = f"{_METADATA_BASE}/postgres/wikisql/diccionario_datos.json"
    VECTOR_DB_PATH    = f"{_APS_CHROMA_BASE_MYSQL}/wikisql"
    PG_VECTOR_DB_PATH = f"{_APS_CHROMA_BASE_PG}/wikisql"
    ACTIVE_DATABASE  = "wikisql_db"

else:
    # demo_db (default)
    SCHEMA_PATH      = f"{_METADATA_BASE}/mysql/demo_db/schema.json"
    DICCIONARIO_PATH = f"{_METADATA_BASE}/mysql/demo_db/diccionario_datos.json"
    PG_SCHEMA_PATH   = f"{_METADATA_BASE}/postgres/demo_db/schema.json"
    PG_DICCIONARIO_PATH = f"{_METADATA_BASE}/postgres/demo_db/diccionario_datos.json"
    VECTOR_DB_PATH    = f"{_APS_CHROMA_BASE_MYSQL}/demo_db"
    PG_VECTOR_DB_PATH = f"{_APS_CHROMA_BASE_PG}/demo_db"
    ACTIVE_DATABASE  = ACTIVE_DATASET
SUMMARIES_PATH = "summaries"

# Tablas de sistema que APS NO debe indexar ni buscar
EXCLUDED_TABLES = ["audit_log"]

# ============================================================
# BASE DE DATOS
# DB_CONFIG y PG_CONFIG se resuelven automaticamente desde ACTIVE_DATASET.
# Tambien se pueden cambiar manualmente si se necesita otro host/puerto.
# ============================================================
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123",
    "database": ACTIVE_DATABASE
}

PG_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "user": "postgres",
    "password": "123",
    "dbname": ACTIVE_DATABASE
}

# ============================================================
# REINTENTOS Y LIMITES
# ============================================================
MAX_RETRIES = 3
MAX_CONVERSATION_HISTORY = 10
ORCHESTRATOR_SUMMARY_MAX_TOKENS = 500
SUSTAINER_MAX_TOKENS  = 300
EXPLAINER_MAX_TOKENS  = 8000
 
# ============================================================
# PONDERACIONES WACS
# ============================================================
CONFIDENCE_WEIGHTS = {
    "AR": 0.10,
    "APS": 0.20,
    "AG": 0.30,
    "AV": 0.40,
}
 
# Umbral del AR para derivar is_valid_query.
# Si confidence_score > AR_VALID_QUERY_THRESHOLD → is_valid_query = True
# Leido por agents/AR/refiner_agent.py
CONFIDENCE_THRESHOLDS = {
    "AR":               0.60,  # P(es query valida) > 0.60 → pasa al pipeline
    "APS_TABLES":       0.55,  # Similitud top-1 TABLA minima (bajado de 0.60 para reducir falsos rechazos borderline)
    "APS_COLUMNS":      0.50,  # Similitud top-1 COLUMNA minima: hard floor del vector search
    "APS_GRAY_ZONE_HIGH": 0.80,  # Por encima: pasa directo sin LLM; por debajo: LLM verifica answerability
    # Flujo APS: TABLES < 0.55 → rechaza | COLUMNS < 0.50 → rechaza
    # 0.50 ≤ COLUMNS < 0.80 → zona gris: LLM verifica si la query es respondible con las columnas halladas
    # COLUMNS ≥ 0.80 → pasa directo (columna claramente compatible)
}

# APS — hiperparametros de recuperacion vectorial
# Cuantas tablas y columnas devuelve ChromaDB por query.
# Optuna puede sobreescribirlos pasando el valor al __init__; estos son los defaults de produccion.
APS_TOP_N_TABLES  = 5
APS_TOP_N_COLUMNS = 10


# ============================================================
# VALIDADOR (AV)
# ============================================================
LIMIT_IS_SUGGESTION_ONLY = True
DEFAULT_LIMIT = 100
 
# ============================================================
# AV — Decision EXPLAIN vs EXPLAIN ANALYZE (per-backend)
# ============================================================
# Antes de validar el plan de un SQL, el AV consulta el row_count
# (estimado, desde el catalogo) de TODAS las tablas referenciadas en el
# FROM/JOINs y aplica la regla:
#   - max(row_count) <= EXPLAIN_ANALYZE_ROW_THRESHOLD_<BACKEND>
#       -> EXPLAIN ANALYZE (ejecuta la consulta y mide tiempos reales)
#   - en caso contrario
#       -> EXPLAIN simple (solo plan, sin ejecutar)
#
# Si EXPLAIN ANALYZE muestra tiempo > PERFORMANCE_THRESHOLD_SECONDS_<BACKEND>,
# el AV genera retroalimentacion al AG sugiriendo optimizar; el ciclo
# AG <-> AV se repite hasta MAX_RETRIES iteraciones. Si tras esas
# iteraciones el tiempo sigue alto, el sistema retorna la consulta
# igualmente, anotando el tiempo medido para que AS y AE lo comuniquen
# al usuario.
EXPLAIN_ANALYZE_ROW_THRESHOLD_MYSQL    = 200_000
EXPLAIN_ANALYZE_ROW_THRESHOLD_POSTGRES = 100_000
PERFORMANCE_THRESHOLD_SECONDS_MYSQL    = 10
PERFORMANCE_THRESHOLD_SECONDS_POSTGRES = 10

# ============================================================
# CDC SCHEMA EXTRACTOR — Muestreo de valores categoricos
# ============================================================
# Decision adaptativa segun el tamano de la tabla, para evitar el costo
# prohibitivo de COUNT(DISTINCT col) sobre tablas grandes.
#
# Flujo:
#   1. ANALYZE TABLE / ANALYZE para refrescar estadisticas del optimizador.
#   2. Leer el row_count ESTIMADO del catalogo (information_schema.TABLES
#      en MySQL, pg_class.reltuples en PostgreSQL). Lectura de microsegundos.
#      Solo si el catalogo devuelve NULL/0 se cae al COUNT(*) exacto.
#   3. Si row_count <= SCHEMA_FULL_SCAN_ROW_THRESHOLD_<BACKEND>:
#        - Cardinalidad EXACTA via COUNT(DISTINCT col) sobre la tabla.
#   4. Si row_count > SCHEMA_FULL_SCAN_ROW_THRESHOLD_<BACKEND>:
#        - Cardinalidad APROXIMADA via COUNT(DISTINCT col) sobre una muestra
#          de SCHEMA_SAMPLE_ROW_LIMIT filas.
#   5. Si la cardinalidad <= SCHEMA_MAX_CARDINALITY, la columna se
#      considera categorica y se muestrean sus valores top.
SCHEMA_FULL_SCAN_ROW_THRESHOLD_MYSQL    = 100_000
SCHEMA_FULL_SCAN_ROW_THRESHOLD_POSTGRES = 100_000
SCHEMA_SAMPLE_ROW_LIMIT                 = 10_000   # tamano de la muestra (compartido)
SCHEMA_MAX_CARDINALITY                  = 10       # umbral semantico (compartido)

ROWS_THRESHOLD_WARNING = 100
ROWS_THRESHOLD_PRODUCTION = 10_000
JOIN_THRESHOLD = 4
 
# ============================================================
# MEMORIA
# ============================================================
MEMORY_SHORT_TERM_ENABLED = True
MEMORY_LONG_TERM_ENABLED = True
MEMORY_LONG_TERM_THRESHOLD = 0.89
# STM similarity para new_sql_query: sinonimos ~0.989, filtros distintos ~0.908
MEMORY_SIMILARITY_THRESHOLD = 0.93
# LTM threshold para new_sql_query: alto para evitar falsos positivos con filtros extra
MEMORY_CONTINUATION_THRESHOLD = 0.97
# LTM threshold para continuation: casi exacto (evita "lima+arte" != "lima+arte+gpa")
MEMORY_CONTINUATION_LTM_THRESHOLD = 0.99
# STM/LTM threshold para context_switch: la nueva query puede estar reformulada
MEMORY_CONTEXT_SWITCH_THRESHOLD = 0.97

# Raiz de la memoria a largo plazo (Chroma persistente).
# Cada usuario+backend+dataset tiene su propio cliente persistente:
#   memoria_chroma/<backend>/<usuario>/<dataset>/chroma.sqlite3
MEMORY_CHROMA_BASE = "memoria_chroma"

# ============================================================
# CONTEXT SWITCH
# ============================================================
CONTEXT_SWITCH_DB_PATH = "context_switch_db"
CONTEXT_SWITCH_COLLECTION = "context_switch"
CONTEXT_SWITCH_THRESHOLD = 0.80

# Umbral minimo de similitud para considerar que un concepto de filtro
# tiene columna compatible en el schema. Por debajo = columna faltante.
APS_FILTER_CONCEPT_THRESHOLD = 0.80

# ============================================================
# AGENT SKILLS (tool calling via Anthropic Claude)
# ============================================================
# Numero maximo de iteraciones del agentic loop por agente.
# Cada iteracion = 1 llamada LLM + N ejecuciones de skills.
# Aumentar si hay consultas muy complejas que requieren mas skills.
MAX_SKILL_ITERATIONS = 8

# Skills registradas por agente (para referencia y auditoria)
AGENT_SKILLS = {
    "AR": [],
    "APS": ["search_tables", "search_columns"],
    "AG": ["validate_sql_safety", "fix_reserved_words"],
    "AV": ["check_syntax_rules", "check_semantic_patterns", "check_query_efficiency", "check_query_performance"],
    "AE": ["format_sql_readable"],
    "AS": ["get_agent_reasoning"],
}