"""
orchestrator/memory_functions.py

Funciones de busqueda en los dos niveles de memoria del sistema NL→SQL.

El sistema usa dos capas de memoria complementarias:

  STM (Short-Term Memory): cache en RAM, valido durante la sesion.
    - Acceso instantaneo por clave exacta (O(1)).
    - Si no hay match exacto, calcula similitud coseno con embeddings
      de las queries almacenadas. Umbral configurable (default 0.93).
    - Evita re-ejecutar el pipeline para preguntas identicas o casi identicas.

  LTM (Long-Term Memory): persistencia en disco via ChromaDB.
    - Sobrevive entre sesiones del mismo usuario.
    - Usa embeddings vectoriales para busqueda semantica.
    - Umbral mas alto que STM (0.97-0.99) para evitar falsos positivos entre
      queries similares pero con filtros distintos.

Estas funciones son invocadas por OrchestratorAgent._route() antes de
activar el pipeline, y por memory_store_node al final del pipeline exitoso.
"""

import numpy as np


def _cosine_similarity(a, b) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def check_short_term_memory(question: str, stm: dict,
                             threshold: float = 0.93,
                             exact_only: bool = False) -> dict:
    """Busca la pregunta en el cache de sesion (RAM). Retorna hit+resultado o miss."""
    if not stm:
        return {"hit": False}

    key = question.lower().strip()

    # Paso 1: match exacto por clave normalizada (sin costo computacional)
    if key in stm:
        entry = stm[key]
        cached_result = entry.get("result", {}) if isinstance(entry.get("result"), dict) else {}
        return {
            "hit": True,
            "cached_result": cached_result,
            "similarity": 1.0,
            "source": "short_term",
            "original_query": entry.get("query", question),
        }

    if exact_only:
        # En modo continuation el orquestador usa exact_only=True para evitar
        # que una query con filtros distintos se confunda con una version mas corta
        # almacenada (ej: "alumnos de lima que estudian arte" != "alumnos de lima")
        return {"hit": False}

    # Paso 2: similitud coseno contra todos los embeddings del STM.
    # Solo se llega aqui si exact_only=False (modo new_sql_query).
    from utils.embeddings import get_embedding
    try:
        from config import EMBEDDING_MODEL
    except ImportError:
        EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    query_emb = get_embedding(question, EMBEDDING_MODEL)
    if query_emb is None:
        return {"hit": False}

    # Buscar la entrada con mayor similitud coseno entre todas las del cache
    best_sim, best_entry = 0.0, None
    for k, entry in stm.items():
        emb = entry.get("embedding")
        if emb is None:
            continue
        sim = _cosine_similarity(query_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_entry = entry

    stored_key = best_entry.get("query", "").lower() if best_entry else ""

    # Proteccion contra falsos positivos en queries de seguimiento conversacional.
    # Si la nueva query contiene la almacenada como subcadena (o viceversa) y la
    # diferencia de longitud supera el 30% de tokens, la nueva tiene filtros extra
    # que hacen semanticamente diferente el resultado → no es un hit valido.
    # Ejemplo: "alumnos de lima" (STM) vs "alumnos de lima que estudian arte" (nueva)
    # tienen similitud ~0.96 pero no son equivalentes.
    if stored_key and (stored_key in key or key in stored_key):
        len_stored = len(stored_key.split())
        len_new    = len(key.split())
        longer, shorter = max(len_stored, len_new), min(len_stored, len_new)
        is_substring_relation = shorter > 0 and (longer - shorter) / shorter > 0.30
    else:
        is_substring_relation = False

    if best_sim >= threshold and best_entry and not is_substring_relation:
        cached_result = best_entry.get("result", {}) if isinstance(best_entry.get("result"), dict) else {}
        return {
            "hit": True,
            "cached_result": cached_result,
            "similarity": round(best_sim, 4),
            "source": "short_term",
            "original_query": best_entry.get("query", question),
        }

    return {"hit": False, "similarity": round(best_sim, 4)}


def check_long_term_memory(question: str, ltm, threshold: float = 0.93) -> dict:
    """
    Busca en ChromaDB (disco) consultas exitosas de sesiones anteriores del mismo usuario.

    ltm es el objeto LongTermMemory inyectado desde api.py. Delega la busqueda
    vectorial a ltm.search() que internamente consulta ChromaDB con embeddings.
    """
    if ltm is None:
        return {"hit": False, "similarity": 0.0, "original_query": ""}

    try:
        result = ltm.search(question, threshold=threshold)
        if result and result.get("hit"):
            return {
                "hit": True,
                "cached_result": result.get("cached_result", {}),
                "similarity": round(result.get("similarity", 0), 4),
                "source": "long_term",
                "original_query": result.get("original_query", ""),
            }
        return {
            "hit": False,
            "similarity": round(result.get("similarity", 0) if result else 0, 4),
            "original_query": result.get("original_query", "") if result else "",
        }
    except Exception:
        return {"hit": False, "similarity": 0.0, "original_query": ""}
