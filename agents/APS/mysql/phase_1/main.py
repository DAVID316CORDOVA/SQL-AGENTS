# -*- coding: utf-8 -*-
"""
agents/APS/mysql/phase_1/main.py

Evaluacion del APS (MySQL) — Fase 1: comparacion de metricas de similitud.

APS hace retrieval semantico contra ChromaDB (vector store de tablas y
columnas). La similitud usada por HNSW se elige al crear la coleccion.
ChromaDB soporta {"cosine", "ip", "l2"}.

Variamos solo la similarity. Todo lo demas (embeddings, top-K, LLM del
APS) queda fijo.

Metricas (sin LLM-juez):
  - precision@5 sobre tablas
  - recall sobre tablas
  - F1 sobre tablas
  - mismo trio sobre columnas
  - combined = (F1_tables + F1_columns) / 2
"""

import os
import sys
import json
import shutil
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

os.environ.setdefault("ACTIVE_DATASET", "spider:concert_singer")

# FIX para segfault Windows: pre-cargar sentence-transformers ANTES de
# que ChromaDB se inicialice. ChromaDB carga su propio pyarrow al
# instanciar PersistentClient. Si despues sentence_transformers intenta
# cargar datasets->pyarrow, hay DLL conflict y segfault 0xC0000005.
# Pre-cargando aca, sentence_transformers carga pyarrow primero y
# ChromaDB reusa lo cargado.
from utils.embeddings import _load as _preload_st_model
_preload_st_model("intfloat/multilingual-e5-base")

from metricas_lib import precision_at_k, f1_retrieval
from metricas_lib import recall_retrieval as recall

# ─────────────────────────────────────────────────────────────────────
# Configuracion
# ─────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")

# Donde se guarda el ganador (lo lee schema_matcher_agent en runtime)
BEST_SIMILARITY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "best_similarity.json"
)

TOP_K_TABLES  = 2   # forzar competencia: el APS debe priorizar las tablas más relevantes
TOP_K_COLUMNS = 5   # columnas: mantener top-5 (21 columnas en el schema)
TOP_K = TOP_K_TABLES  # alias para compatibilidad con _evaluate_one
SIMILARITIES = ["cosine", "ip", "l2"]


# ─────────────────────────────────────────────────────────────────────
# Helper: cambiar similarity y rebuildar indice
# ─────────────────────────────────────────────────────────────────────

def _set_similarity_in_file(sim: str) -> None:
    """Escribe best_similarity.json con la similarity a usar."""
    os.makedirs(os.path.dirname(BEST_SIMILARITY_PATH), exist_ok=True)
    with open(BEST_SIMILARITY_PATH, "w", encoding="utf-8") as f:
        json.dump({"similarity": sim}, f, indent=2)


def _resolve_paths() -> tuple[str, str]:
    """Resuelve schema_path y vector_db_path desde config segun ACTIVE_DATASET."""
    from config import SCHEMA_PATH, VECTOR_DB_PATH
    return os.path.join(ROOT, SCHEMA_PATH), os.path.join(ROOT, VECTOR_DB_PATH)


def _wipe_vector_db():
    """Borra el indice ChromaDB para forzar reindex con nueva similarity.

    En Windows ChromaDB no libera los file handles inmediatamente al
    eliminar el objeto. Forzamos GC y reintentamos si la primera vez falla.
    """
    import gc, time
    _, vector_db_path = _resolve_paths()
    if not os.path.exists(vector_db_path):
        return

    # Forzar liberacion de handles
    gc.collect()
    time.sleep(0.5)

    for attempt in range(5):
        try:
            shutil.rmtree(vector_db_path)
            return
        except PermissionError:
            gc.collect()
            time.sleep(1.0)
    # Ultimo intento sin try (deja el error subir)
    shutil.rmtree(vector_db_path)


# ─────────────────────────────────────────────────────────────────────
# Carga del dataset
# ─────────────────────────────────────────────────────────────────────

def load_samples(dataset_path: str = DATASET_PATH) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]


# ─────────────────────────────────────────────────────────────────────
# Evaluacion de UNA query
# ─────────────────────────────────────────────────────────────────────

def _evaluate_one(db_name: str, sample: dict, top_k: int = TOP_K_TABLES) -> dict:
    """
    Ejecuta APS sobre refined_query llamando las funciones MCP directamente
    (sin subprocess). En produccion Docker el MCP corre en contenedor separado;
    en evaluacion single-process las llamamos como funciones Python normales.
    """
    from agents.MCP.tools.aps_tools import search_tables_server, search_columns_server

    refined_query  = sample["refined_query"]
    expected_tables  = set(sample["expected_tables"])
    expected_columns = set(sample["expected_columns"])

    t0 = time.time()
    try:
        retrieved_tables  = search_tables_server(refined_query, db_name, n=TOP_K_TABLES,  backend="mysql")
        retrieved_columns = search_columns_server(refined_query, db_name, n=TOP_K_COLUMNS, backend="mysql")
        elapsed = time.time() - t0
    except Exception as e:
        return {
            "id": sample["id"],
            "refined_query":   refined_query,
            "expected_tables":  list(expected_tables),
            "expected_columns": list(expected_columns),
            "retrieved_tables":  [],
            "retrieved_columns": [],
            "P_tables": 0.0, "R_tables": 0.0, "F1_tables": 0.0,
            "P_cols":   0.0, "R_cols":   0.0, "F1_cols":   0.0,
            "time_s":   0.0,
            "error":    str(e),
        }

    # Convertir resultados a sets/listas comparables
    # _search_tables retorna lista de dicts {"table_name", "similarity"}
    retrieved_table_names = [t.get("table_name", "") for t in retrieved_tables]
    # _search_columns retorna lista de dicts {"table_name", "column_name", "similarity"}
    retrieved_column_keys = [
        f"{c.get('table_name', '')}.{c.get('column_name', '')}"
        for c in retrieved_columns
    ]

    # Precision@N variable: N = numero de items esperados.
    # APS sigue retornando top-K=5 (en produccion), pero medimos precision
    # con N especifico de la pregunta para que sea comparable
    # independientemente del tamano del schema.
    n_tables  = max(1, len(expected_tables))
    n_columns = max(1, len(expected_columns))

    P_t  = precision_at_k(retrieved_table_names, expected_tables, k=n_tables)
    R_t  = recall(retrieved_table_names[:TOP_K_TABLES], expected_tables)
    F1_t = f1_retrieval(retrieved_table_names, expected_tables, k=n_tables)

    P_c  = precision_at_k(retrieved_column_keys, expected_columns, k=n_columns)
    R_c  = recall(retrieved_column_keys[:TOP_K_COLUMNS], expected_columns)
    F1_c = f1_retrieval(retrieved_column_keys, expected_columns, k=n_columns)

    return {
        "id":                sample["id"],
        "refined_query":     refined_query,
        "expected_tables":   list(expected_tables),
        "expected_columns":  list(expected_columns),
        "retrieved_tables":  retrieved_table_names,
        "retrieved_columns": retrieved_column_keys,
        "P_tables": round(P_t, 3),  "R_tables": round(R_t, 3),  "F1_tables": round(F1_t, 3),
        "P_cols":   round(P_c, 3),  "R_cols":   round(R_c, 3),  "F1_cols":   round(F1_c, 3),
        "time_s":   round(elapsed, 2),
        "error":    None,
    }


# ─────────────────────────────────────────────────────────────────────
# Evaluacion completa para UNA similarity
# ─────────────────────────────────────────────────────────────────────

def evaluate_aps(similarity: str = "cosine",
                 samples: list = None,
                 verbose: bool = False) -> dict:
    """
    Reindexa con la similarity dada y evalua APS sobre todas las queries.

    Devuelve metricas agregadas + per_sample.
    """
    # 1. Setear similarity en archivo y borrar indice viejo
    _set_similarity_in_file(similarity)
    _wipe_vector_db()

    # 2. Instanciar APS con paths del dataset activo (Spider/BIRD/demo_db)
    from agents.APS.mysql.schema_matcher_agent import MysqlSchemaMatcherAgent
    schema_path, vector_db_path = _resolve_paths()

    # Primera instancia: construye el indice (escribe a disco)
    _builder = MysqlSchemaMatcherAgent(
        schema_path=schema_path,
        vector_db_path=vector_db_path,
    )
    # Liberar el client (forzar flush a disco)
    del _builder

    # Segunda instancia: reabre el ChromaDB (workaround bug Windows
    # "Nothing found on disk" cuando se queryea el indice recien escrito).
    agent = MysqlSchemaMatcherAgent(
        schema_path=schema_path,
        vector_db_path=vector_db_path,
    )

    # Liberar handles — la busqueda la hacen las funciones MCP directamente.
    import gc as _gc
    try:
        del agent.tables_col, agent.columns_col, agent.chroma
        _gc.collect()
    except Exception:
        pass
    del agent

    try:
        from config import ACTIVE_DATASET
        db_name = ACTIVE_DATASET
    except ImportError:
        db_name = "spider:concert_singer"

    if samples is None:
        samples = load_samples()

    per_sample = []
    F1_tables_list = []
    F1_cols_list   = []

    for sample in samples:
        result = _evaluate_one(db_name, sample, top_k=TOP_K)
        per_sample.append(result)
        F1_tables_list.append(result["F1_tables"])
        F1_cols_list.append(result["F1_cols"])

        if verbose:
            print(f"  [{result['id']:>2}] F1_t={result['F1_tables']:.3f}  "
                  f"F1_c={result['F1_cols']:.3f}  t={result['time_s']:.2f}s")

    n = len(per_sample)
    avg = lambda xs: round(sum(xs) / n, 3) if n else 0.0

    F1_tables_avg = avg(F1_tables_list)
    F1_cols_avg   = avg(F1_cols_list)
    P_tables_avg  = avg([r["P_tables"] for r in per_sample])
    R_tables_avg  = avg([r["R_tables"] for r in per_sample])
    P_cols_avg    = avg([r["P_cols"]   for r in per_sample])
    R_cols_avg    = avg([r["R_cols"]   for r in per_sample])

    combined = round((F1_tables_avg + F1_cols_avg) / 2, 3)

    avg_time = round(
        sum(r["time_s"] for r in per_sample) / n, 2
    ) if n else 0.0

    result = {
        "similarity":     similarity,
        "n_samples":      n,
        "F1_tables":      F1_tables_avg,
        "P_tables":       P_tables_avg,
        "R_tables":       R_tables_avg,
        "F1_columns":     F1_cols_avg,
        "P_columns":      P_cols_avg,
        "R_columns":      R_cols_avg,
        "combined_score": combined,
        "avg_time_s":     avg_time,
        "per_sample":     per_sample,
    }
    import gc; gc.collect()
    return result


def save_results(metrics: dict, tag: str = "single") -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{tag}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
