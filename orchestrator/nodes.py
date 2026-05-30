"""
orchestrator/nodes.py

Implementa cada nodo del grafo LangGraph y las funciones de decision (aristas condicionales).

Cada funcion *_node recibe el estado completo del grafo (GraphState) y retorna
un diccionario con los campos que desea actualizar. LangGraph fusiona esas
actualizaciones con el estado global usando los reducers definidos en state.py.

Dos modos de ejecucion por nodo:
  - Modo HTTP  (Docker): el agente corre en su propio contenedor. El nodo
    envía el payload via _http() al endpoint /invoke del contenedor.
  - Modo local (monolito): el nodo instancia el agente Python directamente.
    La variable de entorno AGENT_<ROL>_URL determina que modo usar.

Funciones de decision (should_continue_*):
  Retornan un string que LangGraph usa para elegir la siguiente arista.
  Son funciones puras sin efectos secundarios: solo leen el estado.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests as _requests
from orchestrator.state import GraphState

try:
    from config import MAX_RETRIES, CONFIDENCE_WEIGHTS, CONFIDENCE_THRESHOLDS
except ImportError:
    MAX_RETRIES = 3
    CONFIDENCE_WEIGHTS = {"AR": 0.10, "APS": 0.20, "AG": 0.30, "AV": 0.40}
    CONFIDENCE_THRESHOLDS = {"AR": 0.80}

# ================================================================
# MODO HTTP — activo cuando las variables de entorno están definidas
# ================================================================
_AGENT_URLS = {
    "AR":  os.environ.get("AGENT_AR_URL",  ""),
    "APS": os.environ.get("AGENT_APS_URL", ""),
    "AG":  os.environ.get("AGENT_AG_URL",  ""),
    "AV":  os.environ.get("AGENT_AV_URL",  ""),
    "AE":  os.environ.get("AGENT_AE_URL",  ""),
    "AS":  os.environ.get("AGENT_AS_URL",  ""),
}


def _collect_skills(agent, role: str, iteration: int = 0) -> tuple:
    """Monkey-patcha _execute_skill para registrar skills invocadas. Restaurar con _orig en finally."""
    import json as _json
    records = []
    _orig = agent._execute_skill

    def _tracker(skill_name: str, args: dict) -> str:
        result_str = _orig(skill_name, args)
        try:
            response = _json.loads(result_str)
        except Exception:
            response = result_str
        records.append({
            "skill":     skill_name,
            "args":      args,
            "response":  response,
            "iteration": iteration,
        })
        return result_str

    agent._execute_skill = _tracker
    return records, _orig   # devuelve records (lista viva) y original para restaurar


def _sanitize_payload(obj):
    """Elimina recursivamente objetos no serializables a JSON — ToolRegistry, embeddings numpy, objetos LTM."""
    import json as _json
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            try:
                _json.dumps(v)
                clean[k] = v
            except (TypeError, ValueError):
                sanitized = _sanitize_payload(v)
                if sanitized is not None:
                    clean[k] = sanitized
        return clean
    if isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            try:
                _json.dumps(item)
                result.append(item)
            except (TypeError, ValueError):
                sanitized = _sanitize_payload(item)
                if sanitized is not None:
                    result.append(sanitized)
        return result
    # Objeto no serializable: descartar
    return None


def _http(agent: str, payload: dict) -> dict:
    url = _AGENT_URLS[agent]
    safe_payload = _sanitize_payload(payload)
    resp = _requests.post(f"{url}/invoke", json=safe_payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


# ================================================================
# NODO ORQUESTADOR (entry point del grafo)
# ================================================================

_orchestrator_agent = None

def _get_orchestrator():
    global _orchestrator_agent
    if _orchestrator_agent is None:
        from orchestrator.orchestrator_agent import OrchestratorAgent
        _orchestrator_agent = OrchestratorAgent()
    return _orchestrator_agent


def orchestrator_node(state: GraphState) -> dict:
    """
    Primer nodo del grafo. Clasifica el intent, verifica memoria y enriquece la query.

    En una sola llamada LLM determina:
    - Si la pregunta es nueva, de seguimiento (continuation) o cambio de tema.
    - Si la respuesta ya esta en STM o LTM (cache hit → no se activa el pipeline).
    - La query enriquecida con contexto conversacional para los agentes siguientes.
    """
    t0 = time.time()
    user_input = state.get("user_input", "")
    db_type = state.get("db_type", "mysql")
    short_term_cache = state.get("memory_short_term", {})
    long_term_memory = state.get("memory_long_term")
    conversation_history = state.get("conversation_history", [])

    try:
        agent = _get_orchestrator()
        result = agent._route(
            user_input=user_input,
            short_term_cache=short_term_cache,
            long_term_memory=long_term_memory,
            conversation_history=conversation_history,
            db_type=db_type
        )
    except Exception as exc:
        result = {
            "intent": "sql_query",
            "enriched_query": user_input,
            "cached_sql": None,
            "context_switched": False,
            "rejection_reason": None,
            "reasoning": f"Fallback: {exc}"
        }

    intent = result.get("intent", "sql_query")
    enriched = result.get("enriched_query", user_input)
    cached_sql = result.get("cached_sql")
    context_switched = result.get("context_switched", False)
    rejection_reason = result.get("rejection_reason")

    memory_cache_hit = result.get("memory_cache_hit", False)
    memory_cached_result = result.get("memory_cached_result") or {}
    memory_source = result.get("memory_source", "")
    memory_similarity = result.get("memory_similarity", 0)
    memory_original_query = result.get("memory_original_query", "")
    ltm_miss_similarity = result.get("ltm_miss_similarity", 0)
    ltm_miss_closest = result.get("ltm_miss_closest", "")
    ltm_threshold = result.get("ltm_threshold", 0)

    updates = {
        "user_intent": intent,
        "user_input": enriched,
        "context_switched": context_switched,
        "is_out_of_scope": intent == "rejected",
        "rejection_reason": rejection_reason or "",
        "db_type": db_type,
        "memory_cache_hit": memory_cache_hit,
        "memory_source": memory_source,
        "memory_similarity": memory_similarity,
        "memory_original_query": memory_original_query,
        "ltm_miss_similarity": ltm_miss_similarity,
        "ltm_miss_closest": ltm_miss_closest,
        "ltm_threshold": ltm_threshold,
        "all_reasoning": [
            f"[ORCHESTRATOR] Intent: {intent} | DB: {db_type.upper()} | "
            f"Contexto: {'cambiado' if context_switched else 'continuo'} | "
            f"Memoria: {'hit (' + memory_source + ')' if memory_cache_hit else 'miss'} | "
            f"{result.get('reasoning', '')}"
        ]
    }

    if memory_cache_hit and memory_cached_result:
        # Propagar todos los campos del resultado cacheado al estado del grafo
        # para que main.py pueda mostrar el resultado igual que uno fresco.
        for key in ("ar_result", "aps_result", "ag_result", "av_result", "ae_result",
                    "overall_confidence", "iteration_count"):
            if key in memory_cached_result:
                updates[key] = memory_cached_result[key]
        updates["final_sql"] = memory_cached_result.get("final_sql", "")
        updates["memory_cached_result"] = memory_cached_result

    updates["agent_times"] = {"orchestrator": round(time.time() - t0, 2)}
    return updates


def should_continue_after_orchestrator(state: GraphState) -> str:
    # Funcion de decision: retorna el nombre del siguiente nodo segun el intent clasificado.
    # "sustentacion" → el usuario pregunta por que se genero un SQL determinado
    # "conversacional" → saludo, small talk, no es una query de BD
    # cache hit → la respuesta ya esta en memoria, no re-ejecutar el pipeline
    # "rejected/clarification_needed" → el orquestador no puede responder, terminar
    # cualquier otro caso → activar el pipeline comenzando por AR
    intent = state.get("user_intent", "sql_query")
    if intent == "sustentacion":
        return "AS"
    if intent == "conversacional":
        return "end"
    if state.get("memory_cache_hit"):
        return "end"
    if intent in ("rejected", "clarification_needed"):
        return "end"
    return "AR"


# ================================================================
# NODOS DE AGENTES
# ================================================================

def ar_node(state: GraphState) -> dict:
    # AR - Agente Refinador: primer filtro del pipeline.
    # Valida que la pregunta sea una consulta de BD y la refina para los agentes siguientes.
    t0 = time.time()
    if _AGENT_URLS["AR"]:
        result = _http("AR", {"user_input": state.get("user_input", ""),
                               "db_type": state.get("db_type", "mysql"),
                               "active_dataset": os.environ.get("ACTIVE_DATASET", "demo_db")})
        result["agent_times"] = {"AR": round(time.time() - t0, 2)}
        return result
    from agents.AR.refiner_agent import RefinerAgent
    user_input = state.get("user_input", "")
    db_type    = state.get("db_type", "mysql")

    registry = state.get("tool_registry")
    if registry:
        try:
            registry.execute("AR", "check_forbidden_keywords")
        except PermissionError:
            return {"ar_result": {"is_valid_query": False, "confidence_score": 0,
                    "reasoning": "Permiso denegado"}, "all_reasoning": ["[AR] Permiso denegado"]}

    agent = RefinerAgent()
    # Usar la temperatura ganadora del experimento Optuna (almacenada en config)
    from config import AGENT_TEMPERATURES
    temp = AGENT_TEMPERATURES.get("AR", 0.1)
    # Instalar el tracker de skills para registrar que skills invoco el LLM
    # (usado despues en evaluacion del pipeline)
    skill_records, _orig_ar = _collect_skills(agent, "AR")
    try:
        result = agent.process(user_input, temperature=temp)
    finally:
        # Restaurar siempre el _execute_skill original aunque falle el proceso
        agent._execute_skill = _orig_ar
    # Resetear contadores de iteracion al inicio de cada query nueva
    return {
        "ar_result": result,
        "all_reasoning": [f"[AR-Refiner]\n{result.get('reasoning', '')}"],
        "iteration_count": 0, "iteration_history": [],
        "current_feedback": None, "current_sql": "", "final_sql": "",
        "skills_trace": {"AR": skill_records},
        "agent_times": {"AR": round(time.time() - t0, 2)}
    }


_aps_cache = {}

def _get_aps_agent(db_type):
    # Leer paths desde config para soportar ACTIVE_DATASET (Spider, WikiSQL, etc.)
    # Usar paths especificos del backend para separar schemas y indices ChromaDB
    try:
        if (db_type or "mysql").lower() == "postgres":
            from config import PG_SCHEMA_PATH as SCHEMA_PATH, PG_VECTOR_DB_PATH as VECTOR_DB_PATH
        else:
            from config import SCHEMA_PATH, VECTOR_DB_PATH
    except ImportError:
        if (db_type or "mysql").lower() == "postgres":
            SCHEMA_PATH    = "agents/MCP/metadata/postgres/demo_db/schema.json"
            VECTOR_DB_PATH = "agents/APS/postgres/chroma/demo_db"
        else:
            SCHEMA_PATH    = "agents/MCP/metadata/mysql/demo_db/schema.json"
            VECTOR_DB_PATH = "agents/APS/mysql/chroma/demo_db"

    # Cache por combinacion unica de (db_type, schema) para soportar multiples datasets
    cache_key = f"{db_type}::{SCHEMA_PATH}"
    if cache_key not in _aps_cache:
        from agents.APS import SchemaMatcherAgent
        _aps_cache[cache_key] = SchemaMatcherAgent(
            schema_path=SCHEMA_PATH,
            vector_db_path=VECTOR_DB_PATH,
            db_type=db_type,
        )
    return _aps_cache[cache_key]


def aps_node(state: GraphState) -> dict:
    # APS - Schema Matcher: busca en ChromaDB las tablas/columnas relevantes para la pregunta.
    # Retorna el schema reducido (solo las entidades necesarias) que AG usara para generar SQL.
    t0 = time.time()
    if _AGENT_URLS["APS"]:
        result = _http("APS", {"ar_result": state.get("ar_result", {}),
                                "db_type": state.get("db_type", "mysql"),
                                "last_result_for_as": state.get("last_result_for_as"),
                                "active_dataset": os.environ.get("ACTIVE_DATASET", "demo_db")})
        result["agent_times"] = {"APS": round(time.time() - t0, 2)}
        return result
    ar_result = state.get("ar_result", {})
    db_type   = state.get("db_type", "mysql")

    # Cortocircuito: si AR ya rechazo la query, no tiene sentido buscar schema
    if not ar_result.get("is_valid_query"):
        return {
            "aps_result": {"success": False, "agent": "APS-SchemaMatcher",
                           "confidence_score": 0.0, "reasoning": "AR rechazo"},
            "all_reasoning": ["[APS]\nNo ejecutado - consulta rechazada"]
        }

    # El agente APS se cachea por combinacion (db_type, schema_path) para evitar
    # recargar ChromaDB en cada query de la misma sesion
    agent = _get_aps_agent(db_type)
    skill_records, _orig_aps = _collect_skills(agent, "APS")
    try:
        result = agent.process(ar_result)
    finally:
        agent._execute_skill = _orig_aps
    # APS decide por si solo: si ChromaDB no encuentra tablas → success=False.
    # No hay umbrales ni bypasses adicionales — el agente ya maneja eso.

    return {
        "aps_result": result,
        "all_reasoning": [f"[APS-SchemaMatcher]\n{result.get('reasoning', '')}"],
        "skills_trace": {"APS": skill_records},
        "agent_times": {"APS": round(time.time() - t0, 2)}
    }



def ag_node(state: GraphState) -> dict:
    # AG - Generador de SQL: produce la query SELECT a partir del schema reducido de APS.
    # En modo correccion (feedback != None) recibe el SQL anterior y las instrucciones de AV.
    t0 = time.time()
    if _AGENT_URLS["AG"]:
        result = _http("AG", {"aps_result": state.get("aps_result", {}),
                               "db_type": state.get("db_type", "mysql"),
                               "feedback": state.get("current_feedback"),
                               "ar_result": state.get("ar_result", {}),
                               "user_input": state.get("user_input", ""),
                               "current_sql": state.get("current_sql", ""),
                               "active_dataset": os.environ.get("ACTIVE_DATASET", "demo_db")})
        result["agent_times"] = {"AG": round(time.time() - t0, 2)}
        return result
    db_type = state.get("db_type", "mysql")
    # Seleccionar el generador correcto segun el motor de BD activo
    if db_type == "postgres":
        from agents.AG.postgres.sql_generator_agent import SQLGeneratorAgent
    else:
        from agents.AG.mysql.sql_generator_agent import SQLGeneratorAgent
    aps_result = state.get("aps_result", {})
    if not aps_result.get("success"):
        return {
            "ag_result": {"success": False, "agent": "AG-SQLGenerator",
                           "confidence_score": 0.0, "reasoning": "APS fallo", "sql": ""},
            "current_sql": "", "final_sql": "",
            "all_reasoning": ["[AG]\nNo ejecutado - APS fallo"]
        }

    # En modo cache hit del orquestador, APS puede no haber llenado input_query.
    # Se garantiza aqui para que AG siempre tenga la pregunta disponible.
    if not aps_result.get("input_query"):
        ar = state.get("ar_result", {})
        aps_result["input_query"] = (
            ar.get("refined_query") or
            state.get("user_input", "")
        )

    # Garantizar que original_intent existe para que AG pueda usar el tipo de accion
    # (COUNT, MAX, AVG, etc.) en la generacion del SQL
    if not aps_result.get("original_intent"):
        ar = state.get("ar_result", {})
        aps_result["original_intent"] = ar.get("intent", {})

    feedback     = state.get("current_feedback")
    previous_sql = state.get("current_sql", "")
    iteration    = state.get("iteration_count", 0)

    registry = state.get("tool_registry")
    if registry:
        try:
            registry.execute("AG", "validate_sql_safety")
        except PermissionError as e:
            return {
                "ag_result": {"success": False, "agent": "AG-SQLGenerator",
                               "confidence_score": 0.0,
                               "reasoning": "Permiso denegado: validate_sql_safety", "sql": ""},
                "current_sql": "", "all_reasoning": ["[AG] Permiso denegado: validate_sql_safety"]
            }

    agent = SQLGeneratorAgent()

    # Buscar el ultimo SQL exitoso en el historial para encadenamiento conversacional.
    # Se recorre el historial en orden inverso buscando el mensaje mas reciente
    # que contiene "SQL generado:" — ese es el SQL base para la query de seguimiento.
    last_successful_sql = None
    conv_history = state.get("conversation_history", [])
    for h in reversed(conv_history[-20:]):
        if h.get("role") == "system" and "SQL generado:" in h.get("content", ""):
            sql_part = h["content"].split("SQL generado:")[-1].strip()
            if sql_part:
                last_successful_sql = sql_part
                break

    skill_records, _orig_ag = _collect_skills(agent, "AG", iteration)
    try:
        # Modo correccion: AV encontro errores en la iteracion anterior
        if feedback and previous_sql:
            result = agent.process(aps_result, feedback_from_av=feedback, previous_sql=previous_sql,
                                   last_successful_sql=last_successful_sql)
        else:
            result = agent.process(aps_result, last_successful_sql=last_successful_sql)
    finally:
        agent._execute_skill = _orig_ag

    sql = result.get("sql", "")
    if result.get("is_correction"):
        r = f"[AG - CORRECCION #{iteration+1}]\nSQL: {sql}\n{result.get('reasoning','')}"
    else:
        r = f"[AG-SQLGenerator]\nSQL: {sql}\n{result.get('reasoning','')}"
    return {"ag_result": result, "current_sql": sql, "all_reasoning": [r],
            "skills_trace": {"AG": skill_records},
            "agent_times": {"AG": round(time.time() - t0, 2)}}


def av_node(state: GraphState) -> dict:
    t0 = time.time()
    if _AGENT_URLS["AV"]:
        result = _http("AV", {"ag_result": state.get("ag_result", {}),
                               "aps_result": state.get("aps_result", {}),
                               "db_type": state.get("db_type", "mysql"),
                               "iteration_count": state.get("iteration_count", 0),
                               "ar_result": state.get("ar_result", {}),
                               "active_dataset": os.environ.get("ACTIVE_DATASET", "demo_db")})
        result["agent_times"] = {"AV": round(time.time() - t0, 2)}
        # ── Sincronizar estado igual que la rama local ──────────────────
        # La respuesta HTTP de AV no incluye iteration_count ni current_feedback.
        # Si no los seteamos aquí, el contador nunca sube y el orquestador
        # reintenta indefinidamente (siempre iteration=0 < MAX_RETRIES).
        iteration = state.get("iteration_count", 0) + 1
        result["iteration_count"] = iteration
        av_inner = result.get("av_result", {})
        if av_inner.get("needs_correction"):
            result["current_feedback"] = av_inner.get("feedback_for_ag")
        else:
            result["current_feedback"] = None
            # Fijar final_sql solo cuando AV valida OK
            current_sql = state.get("current_sql", "")
            if current_sql:
                result["final_sql"] = current_sql
        return result
    db_type = state.get("db_type", "mysql")
    if db_type == "postgres":
        from agents.AV.postgres.sql_validator_agent import SQLValidatorAgent
    else:
        from agents.AV.mysql.sql_validator_agent import SQLValidatorAgent
    ag_result = state.get("ag_result", {})
    ar_result = state.get("ar_result", {})
    aps_result = state.get("aps_result", {})
    current_sql = state.get("current_sql", "")
    iteration = state.get("iteration_count", 0) + 1

    # Verificar permisos via Tool Registry
    registry = state.get("tool_registry")
    if registry:
        try:
            registry.execute("AV", "check_syntax_rules")
        except PermissionError:
            return {"av_result": {"success": False, "is_valid": False, "needs_correction": True,
                    "confidence_score": 0.0, "reasoning": "Permiso denegado",
                    "errors": ["AV sin permisos"], "warnings": [], "suggestions": []},
                    "iteration_count": iteration,
                    "current_feedback": "AV no tiene permisos para validar SQL.",
                    "all_reasoning": ["[AV] Permiso denegado"]}

    if not ag_result.get("success") or not current_sql:
        return {
            "av_result": {"success": False, "agent": "AV-SQLValidator",
                           "is_valid": False, "needs_correction": False,
                           "confidence_score": 0.0, "reasoning": "Sin SQL",
                           "issues": [], "errors": [], "warnings": [], "suggestions": []},
            "final_sql": "", "iteration_count": iteration,
            "all_reasoning": ["[AV]\nNo ejecutado - sin SQL"]
        }
    agent = SQLValidatorAgent()
    # AV usa ReAct interno — trackear skills
    skill_records, _orig_av = _collect_skills(agent, "AV", iteration)
    try:
        result = agent.process(ag_result, ar_result, aps_result)
    finally:
        agent._execute_skill = _orig_av
    entry = {"iteration": iteration, "sql": current_sql,
             "issues": result.get("errors", []),
             "suggestions": result.get("suggestions", []),
             "feedback": result.get("feedback_for_ag"),
             "confidence": result.get("confidence_score", 0),
             "was_corrected": result.get("needs_correction", False)}
    resp = {"av_result": result, "iteration_count": iteration,
            "iteration_history": [entry],
            "skills_trace": {"AV": skill_records},
            "all_reasoning": [f"[AV - Intento #{iteration}]\n{result.get('reasoning','')}"]}
    if result.get("needs_correction"):
        resp["current_feedback"] = result.get("feedback_for_ag")
    else:
        resp["current_feedback"] = None
        resp["final_sql"] = current_sql
    resp["agent_times"] = {"AV": round(time.time() - t0, 2)}
    return resp


def ae_node(state: GraphState) -> dict:
    t0 = time.time()
    if _AGENT_URLS["AE"]:
        result = _http("AE", {"ar_result": state.get("ar_result", {}),
                               "aps_result": state.get("aps_result", {}),
                               "ag_result": state.get("ag_result", {}),
                               "av_result": state.get("av_result", {}),
                               "final_sql": state.get("final_sql", ""),
                               "db_type": state.get("db_type", "mysql"),
                               "iteration_count": state.get("iteration_count", 0),
                               "conversation_history": state.get("conversation_history", []),
                               "active_dataset": os.environ.get("ACTIVE_DATASET", "demo_db")})
        result["agent_times"] = {"AE": round(time.time() - t0, 2)}
        return result
    aps_result = state.get("aps_result", {})
    av_result = state.get("av_result", {})
    iteration_count = state.get("iteration_count", 0)

    arc = state.get("ar_result", {}).get("confidence_score", 0)
    apsc = aps_result.get("confidence_score", 0)
    agc = state.get("ag_result", {}).get("confidence_score", 0)
    avc = av_result.get("confidence_score", 0) if av_result else 0

    ar_valid = state.get("ar_result", {}).get("is_valid_query")
    aps_success = aps_result.get("success", False)
    oe4_success = state.get("ag_result", {}).get("success", False)

    # Calcular confianza global
    # Para SQL exitoso: promedio ponderado normal
    # Para rechazos: % de confianza de que el rechazo es correcto
    if not ar_valid:
        # AR rechaza: la confianza es que tan seguro esta de que NO es query BD
        # Si arc = 0 (muy seguro que no es), rejection_confidence = 100%
        overall = 1.0 - arc
    elif not aps_success or aps_result.get("below_threshold"):
        # APS rechaza: la confianza es que tan seguro esta de que no hay tablas
        # Si apsc = 0.20 (baja similitud), rejection_confidence = 80%
        overall = 1.0 - apsc
    elif not oe4_success:
        overall = 0.0
    else:
        overall = (arc * CONFIDENCE_WEIGHTS["AR"] +
                   apsc * CONFIDENCE_WEIGHTS["APS"] +
                   agc * CONFIDENCE_WEIGHTS["AG"] +
                   avc * CONFIDENCE_WEIGHTS["AV"])

    final_sql = state.get("final_sql") or state.get("current_sql", "")
    full = dict(state)
    full["overall_confidence"] = round(overall, 3)
    full["final_sql"] = final_sql

    # --- Helper: extraer ultimo SQL exitoso del historial (para chain_rejection) ---
    def _last_successful_sql_from_history():
        for h in reversed((state.get("conversation_history") or [])[-20:]):
            if h.get("role") == "system" and "SQL generado:" in h.get("content", ""):
                sql_part = h["content"].split("SQL generado:")[-1].strip()
                if sql_part:
                    return sql_part
        return ""

    needs_fix = av_result.get("needs_correction", False) if av_result else False

    # Detectar si es un chain_rejection: APS falla DESPUES de un SQL exitoso previo
    _chain_rejection = (not aps_success or aps_result.get("below_threshold", False)) and bool(_last_successful_sql_from_history())
    if _chain_rejection:
        full["chain_rejection"] = True
        full["last_useful_sql"] = _last_successful_sql_from_history()

    # Detectar escenario para el label del reasoning
    if not ar_valid:
        _ae_scenario = "AR_FAILURE"
    elif not aps_success or aps_result.get("below_threshold", False):
        _ae_scenario = "CHAIN_REJECTION" if _chain_rejection else "APS_FAILURE"
    elif av_result.get("unfixable") or (needs_fix and iteration_count >= MAX_RETRIES):
        _ae_scenario = "AG_AV_EXHAUSTED"
    else:
        _ae_scenario = "SUCCESS"

    # --- Verificar permisos via Tool Registry (aplica a todos los escenarios) ---
    registry = state.get("tool_registry")
    if registry:
        try:
            registry.execute("AE", "format_sql_readable")
        except PermissionError:
            return {"ae_result": {"explanation": {"final_reasoning": "Explicacion no disponible."}},
                    "final_sql": final_sql, "overall_confidence": round(overall, 3),
                    "is_complete": True,
                    "all_reasoning": ["[AE] Permiso denegado"],
                    "agent_times": {"AE": round(time.time() - t0, 2)}}

    # --- AE LLM: llamar para TODOS los escenarios (SUCCESS y fallos) ---
    # El LLM genera la explicacion adecuada segun el estado del pipeline.
    from agents.AE.explainer_agent import ExplainerAgent
    agent = ExplainerAgent()
    from config import AGENT_TEMPERATURES
    temp = AGENT_TEMPERATURES.get("AE", 0.3)
    # AE usa ReAct interno — trackear skills
    skill_records, _orig_ae = _collect_skills(agent, "AE")
    try:
        result = agent.process(full, temperature=temp)
    finally:
        agent._execute_skill = _orig_ae
    return {"ae_result": result, "final_sql": final_sql,
            "overall_confidence": round(overall, 3), "is_complete": True,
            "skills_trace": {"AE": skill_records},
            "all_reasoning": ["[AE]\nExplicacion generada"],
            "agent_times": {"AE": round(time.time() - t0, 2)}}


def as_node(state: GraphState) -> dict:
    """
    AS - Sustentador. Se ejecuta DENTRO del grafo cuando
    user_intent == "sustentacion".
    """
    if _AGENT_URLS["AS"]:
        result = _http("AS", {"user_input": state.get("user_input", ""),
                               "last_result_for_as": state.get("last_result_for_as"),
                               "active_dataset": os.environ.get("ACTIVE_DATASET", "demo_db")})
        return result
    from agents.AS.sustainer_agent import SustainerAgent

    user_input = state.get("user_input", "")
    last_result = state.get("last_result_for_as", {})

    # Verificar permisos via Tool Registry
    registry = state.get("tool_registry")
    if registry:
        try:
            registry.execute("AS", "get_agent_reasoning")
        except PermissionError:
            return {
                "ae_result": {"explanation": {"final_reasoning": "Sustentador no disponible (permiso denegado)."}},
                "is_complete": True,
                "all_reasoning": ["[AS] Permiso denegado"]
            }

    if not last_result or not last_result.get("final_sql"):
        return {
            "ae_result": {"explanation": {"final_reasoning": "No hay una consulta anterior para explicar."}},
            "is_complete": True,
            "skills_trace": {"AS": []},
            "all_reasoning": ["[AS] Sin resultado anterior"]
        }

    try:
        agent = SustainerAgent()
        from config import AGENT_TEMPERATURES
        temp = AGENT_TEMPERATURES.get("AS", 0.3)
        # AS usa ReAct interno — trackear skills
        skill_records, _orig_as = _collect_skills(agent, "AS")
        try:
            answer = agent.process(last_result, user_input, temperature=temp)
        finally:
            agent._execute_skill = _orig_as
        return {
            "ae_result": {"explanation": {"final_reasoning": answer}},
            "final_sql": last_result.get("final_sql", ""),
            "is_complete": True,
            "skills_trace": {"AS": skill_records},
            "all_reasoning": [f"[AS]\n{answer}"]
        }
    except Exception as e:
        return {
            "ae_result": {"explanation": {"final_reasoning": f"Error en sustentador: {e}"}},
            "is_complete": True,
            "skills_trace": {"AS": []},
            "all_reasoning": [f"[AS] Error: {e}"]
        }


# ================================================================
# NODO DE ESCRITURA EN MEMORIA CORTO PLAZO
# ================================================================

def memory_store_node(state: GraphState) -> dict:
    """
    Guarda el resultado del pipeline en la memoria de corto plazo (STM).

    Se ejecuta despues de AE para cada query que llego hasta el final del pipeline
    (con o sin SQL exitoso). Solo guarda si la query fue valida (ar_result.is_valid_query)
    y si hay un final_sql no vacio.

    LangGraph no comparte dicts por referencia entre nodos: el nodo debe retornar
    la STM actualizada como parte del dict de actualizaciones, y LangGraph se encarga
    de propagarla al estado global.
    """
    try:
        from config import MEMORY_SHORT_TERM_ENABLED, EMBEDDING_MODEL
        if not MEMORY_SHORT_TERM_ENABLED:
            return {}

        ar = state.get("ar_result") or {}
        if not ar.get("is_valid_query"):
            return {}

        final_sql = state.get("final_sql", "")
        if not final_sql:
            return {}

        enriched_query = state.get("user_input", "")
        original_query = state.get("original_user_input", enriched_query)

        from utils.embeddings import get_embedding
        import uuid
        from datetime import datetime as _dt

        cache_key = enriched_query.lower().strip()
        emb = get_embedding(enriched_query, EMBEDDING_MODEL)

        cached_result = {k: state.get(k) for k in (
            "ar_result", "aps_result", "ag_result", "av_result", "ae_result",
            "final_sql", "overall_confidence", "iteration_count",
            "context_switched", "memory_cache_hit", "memory_source", "agent_times"
        )}

        stm = dict(state.get("memory_short_term") or {})
        stm[cache_key] = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": _dt.now().isoformat(),
            "query": enriched_query,
            "original_query": original_query,
            "embedding": emb,
            "result": cached_result,
        }
        return {"memory_short_term": stm}
    except Exception:
        pass
    return {}


# ================================================================
# FUNCIONES DE DECISION
# ================================================================

def should_continue_after_ar(state: GraphState) -> str:
    # Si AR determino que la pregunta no es una consulta de BD (confidence bajo),
    # ir directo a AE para que explique el rechazo al usuario.
    ar = state.get("ar_result", {})
    if not ar.get("is_valid_query"):
        return "AE"
    return "APS"


def should_continue_after_aps(state: GraphState) -> str:
    # Si APS no encontro tablas con similitud suficiente, no hay schema para generar SQL.
    # AE explicara que la BD no tiene informacion para esa pregunta.
    aps = state.get("aps_result", {})
    if not aps.get("success"):
        return "AE"
    return "AG"


def should_retry_or_continue(state: GraphState) -> str:
    av        = state.get("av_result", {})
    iteration = state.get("iteration_count", 0)

    # SQL validado correctamente: continuar a AE para generar la explicacion
    if not av.get("needs_correction"):
        return "AE"

    # Error irrecuperable: el schema no tiene la columna necesaria.
    # Reintentar no servira porque AG no puede inventar columnas que no existen.
    if av.get("unfixable"):
        return "AE"

    # AG ya reporto en su reasoning que omitio un filtro por limitacion del schema.
    # Si AV reclama ese filtro, es un conflicto insolucionable → terminar.
    ag = state.get("ag_result", {})
    ag_reasoning = (ag.get("reasoning") or "").lower()
    schema_limit_phrases = [
        "no existe columna", "no hay columna", "se omiti",
        "columna faltante", "no es posible filtrar"
    ]
    if any(p in ag_reasoning for p in schema_limit_phrases):
        return "AE"

    # Hay errores corregibles y aun quedan intentos: volver a AG con el feedback de AV
    if iteration < MAX_RETRIES:
        return "retry"

    # Se agotaron los reintentos: AE mostrara el ultimo SQL aunque tenga errores
    return "AE"