"""
state.py

Estado compartido del grafo LangGraph.
Incluye campos de memoria a corto plazo que se inyectan
desde main.py en cada invocacion del grafo.

Flujo de memoria:
  1. main.py mantiene short_term_cache (dict en RAM)
  2. Antes de invocar el grafo, inyecta el cache en GraphState
  3. El nodo memory_check busca en corto plazo (RAM) y largo plazo (ChromaDB)
  4. Si hay hit → salta a END
  5. Si no → ejecuta AR→AE
  6. Despues, main.py guarda resultado en corto plazo (RAM)
  7. Al cerrar sesion, main.py manda todo el corto plazo a largo plazo (ChromaDB)

Ubicacion: orchestrator/state.py
"""

from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
import operator


def _merge_dicts(a: Dict, b: Dict) -> Dict:
    """
    Fusiona dos dicts para reducers de LangGraph.
    - Si ambos tienen la misma clave y los valores son listas → extend (acumula).
    - En otro caso → overwrite (comportamiento normal, util para agent_times).
    Esto permite que skills_trace acumule skills en reintentos (AG iter 1 + iter 2),
    mientras agent_times simplemente actualiza con el ultimo valor.
    """
    merged = dict(a) if a else {}
    if b:
        for k, v in b.items():
            if k in merged and isinstance(merged[k], list) and isinstance(v, list):
                merged[k] = merged[k] + v
            else:
                merged[k] = v
    return merged


class GraphState(TypedDict, total=False):
    # Input
    user_input: str
    original_user_input: str
    username: str

    # Resultados de agentes
    ar_result: Dict[str, Any]
    aps_result: Dict[str, Any]
    ag_result: Dict[str, Any]
    av_result: Dict[str, Any]
    ae_result: Dict[str, Any]

    # SQL
    final_sql: str
    current_sql: str

    # Razonamiento (se concatena)
    all_reasoning: Annotated[List[str], operator.add]

    # Confianza
    overall_confidence: float

    # Iteraciones
    iteration_count: int
    iteration_history: Annotated[List[Dict[str, Any]], operator.add]

    # Feedback
    current_feedback: Optional[str]

    # Control
    error: Optional[str]
    is_complete: bool

    # ============================================================
    # MEMORIA CORTO PLAZO (inyectada por main.py via GraphState)
    # ============================================================
    # Cache de sesion: {query_lower: {"query", "embedding", "result"}}
    memory_short_term: Dict[str, Any]

    # Objeto LongTermMemory (inyectado por main.py para buscar en ChromaDB)
    memory_long_term: Any

    # Resultado del chequeo
    memory_cache_hit: bool
    memory_cached_result: Dict[str, Any]
    memory_similarity: float
    memory_original_query: str
    memory_source: str  # "short_term" o "long_term"
    ltm_miss_similarity: float
    ltm_miss_closest: str
    ltm_threshold: float

    # ============================================================
    # TOOL REGISTRY (control de acceso por agente)
    # ============================================================
    tool_registry: Any

    # ============================================================
    # INTENCION Y CONTEXTO
    # ============================================================
    user_intent: str  # "sustentacion", "conversacional", "sql_query", "rejected"
    last_result_for_as: Dict[str, Any]  # Resultado anterior para sustentador

    # ============================================================
    # MOTOR DE BASE DE DATOS
    # ============================================================
    db_type: str  # "mysql" o "postgres"

    # ============================================================
    # ORQUESTADOR - campos gestionados por OrchestratorAgent
    # ============================================================
    conversation_history: List[Dict[str, Any]]  # historial de la sesion
    context_switched: bool       # True si el usuario cambio de tema
    is_out_of_scope: bool        # True si la pregunta fue rechazada
    rejection_reason: str        # motivo del rechazo si aplica

    # ============================================================
    # TIMING POR AGENTE
    # ============================================================
    agent_times: Annotated[Dict[str, float], _merge_dicts]

    # ============================================================
    # SKILLS TRACE — registro de skills invocadas por cada agente
    # Estructura: {"AR": [...], "AG": [...], "AV": [...], ...}
    # Cada entrada: {"skill": str, "args": dict, "response": any, "iteration": int}
    # Se usa en evaluacion del orquestador para verificar uso correcto de skills.
    # ============================================================
    skills_trace: Annotated[Dict[str, Any], _merge_dicts]