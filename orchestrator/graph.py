"""
orchestrator/graph.py

Define el grafo de ejecucion del sistema SQL-Agents usando LangGraph.

El grafo es un DAG (grafo dirigido aciclico) con aristas condicionales que
permiten cortocircuitar el pipeline segun el estado del sistema:

  Flujo principal (query SQL nueva):
    orchestrator → AR → APS → AG → AV ──(error, retry)──► AG
                                        └──(valido)──► AE → memory_store → END

  Flujo de sustentacion (usuario pregunta "por que?"):
    orchestrator → AS → END

  Flujos de terminacion anticipada:
    orchestrator → END  (conversacional, cache hit STM/LTM, rejected)
    AR → AE → memory_store → END  (query invalida: no es pregunta de BD)
    APS → AE → memory_store → END (schema sin tablas relevantes)

El nodo memory_store guarda el resultado en la memoria de corto plazo (STM)
para que consultas similares en la misma sesion no re-ejecuten el pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from orchestrator.state import GraphState
from orchestrator.nodes import (
    orchestrator_node,
    ar_node, aps_node, ag_node, av_node, ae_node, as_node,
    memory_store_node,
    should_continue_after_orchestrator,
    should_continue_after_ar,
    should_continue_after_aps,
    should_retry_or_continue
)

try:
    from registry import setup_registry
    _registry = setup_registry()
except ImportError:
    _registry = None


def create_graph():
    # Se crea un grafo nuevo en cada invocacion para garantizar estado limpio.
    # LangGraph compila el grafo a un ejecutor optimizado con el metodo .compile().
    workflow = StateGraph(GraphState)

    # Registrar cada agente como un nodo del grafo
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("AR",           ar_node)
    workflow.add_node("APS",          aps_node)
    workflow.add_node("AG",           ag_node)
    workflow.add_node("AV",           av_node)
    workflow.add_node("AE",           ae_node)
    workflow.add_node("AS",           as_node)
    workflow.add_node("memory_store", memory_store_node)

    # El orquestador es siempre el primer nodo en ejecutarse
    workflow.set_entry_point("orchestrator")

    # El orquestador clasifica el intent y decide el siguiente paso:
    # "AR"  → pipeline SQL (nuevo o continuacion)
    # "AS"  → sustentacion (el usuario pregunta "por que?" sobre la query anterior)
    # "end" → conversacional, cache hit o rechazado directamente
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue_after_orchestrator,
        {"AR": "AR", "AS": "AS", "end": END}
    )

    # AR decide si la pregunta es una query de BD valida
    # "APS" → si, continuar con busqueda en schema
    # "AE"  → no, ir directo al explicador que informa el rechazo
    workflow.add_conditional_edges(
        "AR", should_continue_after_ar,
        {"APS": "APS", "AE": "AE"}
    )

    # APS decide si encontro tablas relevantes para la pregunta
    # "AG" → si, generar el SQL
    # "AE" → no, informar al usuario que no hay schema para esa pregunta
    workflow.add_conditional_edges(
        "APS", should_continue_after_aps,
        {"AG": "AG", "AE": "AE"}
    )

    # AG siempre pasa a AV para validacion (arista fija, sin condicion)
    workflow.add_edge("AG", "AV")

    # AV decide si el SQL es valido o necesita correccion
    # "retry" → volver a AG con feedback (ciclo de correccion)
    # "AE"    → SQL valido o error irrecuperable, ir al explicador
    workflow.add_conditional_edges(
        "AV", should_retry_or_continue,
        {"retry": "AG", "AE": "AE"}
    )

    # AE siempre guarda en memoria y termina
    workflow.add_edge("AE", "memory_store")
    workflow.add_edge("memory_store", END)
    # AS (sustentador) termina directamente sin guardar en memoria
    workflow.add_edge("AS", END)

    return workflow.compile()


_REGISTRY_DEFAULT = object()  # Centinela para distinguir "no pasado" de None


def run_query(user_input, memory_short_term=None, memory_long_term=None,
              user_intent="otro", last_result_for_as=None, db_type="mysql",
              conversation_history=None, tool_registry=_REGISTRY_DEFAULT):
    # Crear y compilar el grafo para esta invocacion
    graph = create_graph()

    # tool_registry=None deshabilita el control de permisos (util en evaluacion E2E).
    # Si no se pasa ningun valor, usar el registry global configurado en startup.
    reg = _registry if tool_registry is _REGISTRY_DEFAULT else tool_registry

    # Estado inicial del grafo: todos los campos del GraphState con sus valores por defecto.
    # LangGraph propaga este estado a cada nodo y acumula las actualizaciones que cada
    # nodo retorna (ver state.py para la logica de merge por campo).
    initial_state = {
        "user_input":           user_input,
        "original_user_input":  user_input,   # preservar el input crudo antes de enrichment
        "all_reasoning":        [],
        "iteration_history":    [],
        "iteration_count":      0,
        "current_feedback":     None,
        "current_sql":          "",
        "final_sql":            "",
        "is_complete":          False,
        "memory_short_term":    memory_short_term or {},
        "memory_long_term":     memory_long_term,
        "memory_cache_hit":     False,
        "tool_registry":        reg,
        "user_intent":          user_intent,
        "last_result_for_as":   last_result_for_as or {},  # resultado anterior para AS
        "db_type":              db_type,
        "conversation_history": conversation_history or [],
        "context_switched":     False,
        "is_out_of_scope":      False,
        "rejection_reason":     "",
        "agent_times":          {},
    }

    return graph.invoke(initial_state)


def get_registry():
    return _registry
