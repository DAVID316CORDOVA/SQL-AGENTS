"""
orchestrator/orchestrator_agent.py

Orquestador del sistema NL-to-SQL.

Dos niveles de responsabilidad:

  _route()   — routing interno (llamado por orchestrator_node dentro del grafo).
               Clasifica intent, verifica memoria, enriquece query. Una sola LLM call.

  process()  — nivel sesion (llamado desde main.py / api.py).
               Llama el grafo completo y gestiona todo el estado de sesion:
               STM, historial, sustentador, context_switch.

  close_session()  — flush STM → LTM al cerrar sesion.
  reset_context()  — reinicio de contexto (nuevo tema).
"""

import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import get_client, get_model, get_client_for_model, call_llm
from orchestrator.prompt import ORCHESTRATOR_PROMPT
from orchestrator.memory_functions import check_short_term_memory, check_long_term_memory

try:
    from config import (
        MEMORY_SIMILARITY_THRESHOLD, MEMORY_CONTINUATION_THRESHOLD,
        MEMORY_CONTINUATION_LTM_THRESHOLD, MEMORY_CONTEXT_SWITCH_THRESHOLD,
        MEMORY_SHORT_TERM_ENABLED, MEMORY_LONG_TERM_ENABLED,
        MAX_CONVERSATION_HISTORY,
    )
except ImportError:
    MEMORY_SIMILARITY_THRESHOLD = 0.93
    MEMORY_CONTINUATION_THRESHOLD = 0.97
    MEMORY_CONTINUATION_LTM_THRESHOLD = 0.99
    MEMORY_CONTEXT_SWITCH_THRESHOLD = 0.97
    MEMORY_SHORT_TERM_ENABLED = True
    MEMORY_LONG_TERM_ENABLED = True
    MAX_CONVERSATION_HISTORY = 20


class OrchestratorAgent:

    def __init__(self, db_type: str = "mysql",
                 long_term_memory=None, sustainer=None,
                 model: str = None, temperature: float = None):
        self.model = model or get_model("classifier")
        self.client = get_client_for_model(self.model)
        self._agent_label = "Orchestrator"
        # Temperatura: parametro > config > 0.0
        if temperature is not None:
            self.temperature = temperature
        else:
            try:
                from config import AGENT_TEMPERATURES
                self.temperature = float(AGENT_TEMPERATURES.get("classifier", 0.0))
            except Exception:
                self.temperature = 0.0

        # Configuracion de sesion
        self.db_type = db_type
        self.ltm = long_term_memory
        self.sustainer = sustainer

        # Estado de sesion — propiedad del orquestador
        self.stm: dict = {}
        self.conversation_history: list = []
        self._last_successful_result: dict = {}

    # ------------------------------------------------------------------
    # API publica — nivel sesion
    # ------------------------------------------------------------------

    def process(self, user_input: str) -> dict:
        """
        Punto de entrada de nivel sesion (llamado desde main.py y api.py).
        Invoca el grafo completo y gestiona el estado de sesion: STM, historial
        conversacional, contexto del sustentador y sincronizacion LTM→STM.
        """
        from orchestrator.graph import run_query

        result = run_query(
            user_input,
            memory_short_term=self.stm,
            memory_long_term=self.ltm,
            last_result_for_as=self._last_successful_result,
            db_type=self.db_type,
            conversation_history=self.conversation_history,
        )

        enriched_query = result.get("user_input", user_input)
        intent = result.get("user_intent", result.get("intent", "sql_query"))

        # Para hits de LTM: guardar en el historial la forma exacta almacenada en ChromaDB.
        # El input crudo y la forma almacenada difieren ligeramente (~2% de similitud),
        # lo que podria hacer que la siguiente query de seguimiento no supere el umbral 0.99.
        # Usando la forma almacenada, el enriquecimiento del LLM produce similitud ~1.0.
        if result.get("memory_cache_hit") and result.get("memory_source") == "long_term":
            _stored = (result.get("memory_original_query") or "").strip()
            if _stored:
                enriched_query = _stored

        # Cambio de contexto: el usuario inicio un tema nuevo.
        # Limpiar historial y memoria de sustentacion para empezar desde cero.
        if result.get("context_switched"):
            self.conversation_history.clear()
            if self.sustainer:
                self.sustainer.reset_context()
            self._last_successful_result = {}

        # Cache hit: aplanar el resultado para que el caller (api.py) lo use igual
        # que un resultado fresco del pipeline, sin conocer que vino de cache.
        if result.get("memory_cache_hit"):
            cached = result.get("memory_cached_result") or {}
            result = {
                **cached,
                "memory_cache_hit": True,
                "memory_source": result.get("memory_source"),
                "memory_similarity": result.get("memory_similarity", 0),
                "memory_original_query": result.get("memory_original_query", ""),
                "context_switched": result.get("context_switched", False),
                "user_intent": intent,
                "user_input": user_input,
                "enriched_query": enriched_query,
            }

        final_sql = result.get("final_sql", "")

        # Notificar al sustentador del nuevo SQL exitoso para que pueda responder
        # preguntas del tipo "por que se uso ese JOIN?" en el siguiente turno
        if final_sql and self.sustainer:
            self.sustainer.add_to_context(enriched_query, final_sql)

        # Actualizar historial: la clarificacion pendiente se registra con rol especial
        # para que el orquestador la detecte en el proximo turno y la resuelva.
        if intent == "clarification_needed":
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "system", "content": f"Clarificacion pendiente: \"{enriched_query}\""})
            cap = MAX_CONVERSATION_HISTORY * 2
            if len(self.conversation_history) > cap:
                # Mantener solo los mensajes mas recientes para no inflar el contexto
                self.conversation_history = self.conversation_history[-cap:]

        # Para queries SQL (nuevas, continuation, context_switch):
        # agregar al historial un resumen legible del resultado para que el LLM
        # pueda enriquecer correctamente la siguiente query de seguimiento.
        if intent not in ("conversacional", "sustentacion", "clarification_needed"):
            ar_valid = (result.get("ar_result") or {}).get("is_valid_query", False)
            aps_ok   = (result.get("aps_result") or {}).get("success", True)
            aps_low  = (result.get("aps_result") or {}).get("below_threshold", False)

            if final_sql:
                summary = f"Pregunta: \"{enriched_query}\" -> SQL generado: {final_sql[:80]}"
            elif not ar_valid:
                summary = f"Pregunta: \"{user_input}\" -> Rechazada (no es consulta BD)"
            elif not aps_ok or aps_low:
                summary = f"Pregunta: \"{user_input}\" -> Rechazada (sin tablas relevantes)"
            else:
                summary = f"Pregunta: \"{user_input}\" -> Sin resultado"

            self.conversation_history.append({"role": "user",   "content": enriched_query})
            self.conversation_history.append({"role": "system", "content": summary})

            cap = MAX_CONVERSATION_HISTORY * 2
            if len(self.conversation_history) > cap:
                self.conversation_history = self.conversation_history[-cap:]

        # Sincronizar la STM: memory_store_node actualiza el estado del grafo pero
        # no el dict de sesion de este objeto. Sin este sync, close_session() flushearia
        # un dict vacio a LTM perdiendo todo lo generado en la sesion.
        new_stm = result.get("memory_short_term")
        if isinstance(new_stm, dict) and new_stm:
            self.stm.update(new_stm)

        # Poblar STM desde LTM para que consultas repetidas en la misma sesion
        # sean instantaneas (match exacto en STM, sin volver a consultar ChromaDB).
        # Los hits de LTM no pasan por memory_store_node, por eso se agregan aqui.
        if (MEMORY_SHORT_TERM_ENABLED and final_sql
                and result.get("memory_source") == "long_term"):
            _stm_key = enriched_query.lower().strip()
            if _stm_key and _stm_key not in self.stm:
                # Excluir campos de metadata de cache para no contaminar la entrada STM
                _stm_entry = {k: v for k, v in result.items()
                              if k not in ("memory_cache_hit", "memory_source",
                                           "memory_similarity", "memory_original_query",
                                           "context_switched", "user_intent",
                                           "user_input", "enriched_query")}
                self.stm[_stm_key] = {
                    "query": enriched_query,
                    "original_query": user_input,
                    "result": _stm_entry,
                }

        # Actualizar el ultimo resultado exitoso para que AS pueda acceder a el
        # en la siguiente pregunta de sustentacion
        if final_sql:
            self._last_successful_result = result

        return result

    def reset_context(self):
        """Reinicia el contexto conversacional (comando 'nuevo tema')."""
        self.conversation_history.clear()
        self._last_successful_result = {}
        if self.sustainer:
            self.sustainer.reset_context()

    def close_session(self) -> int:
        """
        Persiste la STM en LTM al cerrar la sesion (flush STM → ChromaDB).

        Itera sobre todas las queries exitosas de la sesion y las guarda en
        ChromaDB con sus embeddings para que esten disponibles en sesiones futuras.
        Retorna el numero de queries guardadas.
        """
        if not self.ltm or not getattr(self.ltm, "enabled", False) or not self.stm:
            return 0
        count = 0
        # Cada entrada en la STM corresponde a una query exitosa de la sesion
        for _, entry in self.stm.items():
            result_data = entry.get("result")
            query = entry.get("query", "")
            if result_data and query:
                self.ltm.store(query, result_data,
                               original_query=entry.get("original_query"))
                count += 1
        return count

    # ------------------------------------------------------------------
    # Routing interno — llamado por orchestrator_node dentro del grafo
    # ------------------------------------------------------------------

    def _route(self, user_input: str, short_term_cache: dict = None,
               long_term_memory=None, conversation_history: list = None,
               db_type: str = "mysql") -> dict:
        """
        Nucleo de la logica de routing. Llamado exclusivamente por orchestrator_node.

        Ejecuta en dos etapas:
          Etapa 0 - Match exacto STM (sin LLM, 0ms): si la query es identica a una
                    almacenada, retorna el resultado cacheado inmediatamente.
          Etapa 1 - Clasificacion LLM: una llamada al LLM clasifica el intent
                    (new_sql_query, continuation, context_switch, conversacional, etc.)
                    y enriquece la query con contexto conversacional.
          Etapa 2 - Busqueda en memoria: segun el decision del LLM, busca en STM/LTM
                    con umbrales calibrados por tipo de decision.
        """
        stm     = short_term_cache or {}
        ltm     = long_term_memory
        history = conversation_history or []
        db      = db_type or self.db_type

        # Estructura de respuesta por defecto: se usa cuando ningun camino especial aplica
        fallback = {
            "intent": "sql_query",
            "enriched_query": user_input,
            "cached_sql": None,
            "context_switched": False,
            "rejection_reason": None,
            "reasoning": "fallback",
            "memory_cache_hit": False,
            "memory_cached_result": None,
            "memory_source": None,
            "ltm_miss_similarity": 0,
            "ltm_miss_closest": "",
            "ltm_threshold": MEMORY_SIMILARITY_THRESHOLD,
        }

        def _dbg(msg):
            try:
                with open("orch_debug.log", "a", encoding="utf-8") as _f:
                    _f.write(msg + "\n")
            except Exception:
                pass

        # ── Match EXACTO STM (0ms, sin LLM) ──────────────────────────
        _dbg(f"INPUT={user_input!r} STM_KEYS={list(stm.keys())}")
        stm_exact = self._check_stm(user_input, stm, exact_only=True)
        _dbg(f"EXACT_HIT={stm_exact['hit']}")
        if stm_exact["hit"]:
            return {
                **fallback,
                "memory_cache_hit": True,
                "memory_cached_result": stm_exact["cached_result"],
                "memory_source": "short_term",
                "memory_similarity": 1.0,
                "memory_original_query": stm_exact["original_query"],
                "reasoning": "STM exact hit",
            }

        # ── Afirmacion de clarificacion pendiente (sin LLM) ─────────
        _pending = self._get_pending_clarification(history)
        if _pending:
            _aff = {"si", "sí", "ok", "dale", "correcto", "claro",
                    "exacto", "eso", "afirmativo", "yes", "ya", "listo"}
            _qs  = ("cuantos", "cuántos", "cuales", "cuáles", "dame", "lista",
                    "quienes", "quiénes", "cual", "cuál", "que ", "qué ", "promedio")
            _u = user_input.lower().strip().rstrip("?.!,;").strip()
            _p = _pending.lower().lstrip("¿").strip()
            if _u in _aff and any(_p.startswith(w) for w in _qs):
                return {
                    **fallback,
                    "intent": "sql_query",
                    "enriched_query": _pending.strip("¿?").strip(),
                    "reasoning": "Usuario confirmo la consulta pendiente de clarificacion",
                }

        # ── ETAPA 1: Ruteo LLM ───────────────────────────────────────
        recent  = self._get_last_successful_questions(history)
        routing = self._route_with_llm(user_input, recent, db, history=history)

        decision      = routing.get("decision", "new_sql_query")
        enriched_query = (routing.get("enriched_query") or user_input).strip() or user_input
        reasoning     = routing.get("reasoning", "")

        if decision == "context_switch":
            embedded = enriched_query
            stm_hit  = self._check_stm(embedded, stm, threshold=MEMORY_CONTEXT_SWITCH_THRESHOLD)
            if stm_hit["hit"]:
                return {**fallback, "enriched_query": embedded, "context_switched": True,
                        "intent": "sql_query", "reasoning": reasoning,
                        "memory_cache_hit": True,
                        "memory_cached_result": stm_hit["cached_result"],
                        "memory_source": "short_term",
                        "memory_similarity": stm_hit["similarity"],
                        "memory_original_query": stm_hit["original_query"]}
            ltm_hit = self._check_ltm(embedded, ltm, threshold=MEMORY_CONTEXT_SWITCH_THRESHOLD)
            if ltm_hit["hit"]:
                return {**fallback, "enriched_query": embedded, "context_switched": True,
                        "intent": "sql_query", "reasoning": reasoning,
                        "memory_cache_hit": True,
                        "memory_cached_result": ltm_hit["cached_result"],
                        "memory_source": "long_term",
                        "memory_similarity": ltm_hit["similarity"],
                        "memory_original_query": ltm_hit["original_query"]}
            return {**fallback, "enriched_query": embedded,
                    "context_switched": True, "intent": "sql_query", "reasoning": reasoning}

        intent_map = {
            "sustentation": "sustentacion",
            "conversacional": "conversacional",
            "clarification_needed": "clarification_needed",
        }
        intent = intent_map.get(decision, "sql_query")

        if intent == "clarification_needed":
            return {"intent": "clarification_needed", "enriched_query": enriched_query,
                    "context_switched": False, "rejection_reason": None,
                    "reasoning": f"decision=clarification_needed | {reasoning}",
                    "memory_cache_hit": False, "memory_cached_result": None,
                    "memory_source": None, "cached_sql": None,
                    "ltm_miss_similarity": 0, "ltm_miss_closest": "",
                    "ltm_threshold": MEMORY_SIMILARITY_THRESHOLD}

        if intent in ("conversacional", "sustentacion"):
            return {"intent": intent, "enriched_query": enriched_query,
                    "context_switched": False, "rejection_reason": None,
                    "reasoning": f"decision={decision} | {reasoning}",
                    "memory_cache_hit": False, "memory_cached_result": None,
                    "memory_source": None, "cached_sql": None,
                    "ltm_miss_similarity": 0, "ltm_miss_closest": "",
                    "ltm_threshold": MEMORY_SIMILARITY_THRESHOLD}

        # ── ETAPA 2: Busqueda en memoria ─────────────────────────────
        ltm_miss_sim = 0
        ltm_miss_closest = ""
        _dbg(f"DECISION={decision!r} ENRICHED={enriched_query!r}")

        if decision == "continuation":
            # STM: solo exacto (similitud genera falsos positivos entre queries con
            # distintos filtros, ej: "de lima" vs "de lima que estudian filosofia")
            stm_result = self._check_stm(enriched_query, stm, exact_only=True)
            _dbg(f"CONT STM_EXACT={stm_result['hit']}")
            if stm_result["hit"]:
                return {**fallback, "enriched_query": enriched_query,
                        "memory_cache_hit": True,
                        "memory_cached_result": stm_result["cached_result"],
                        "memory_source": "short_term", "memory_similarity": 1.0,
                        "memory_original_query": stm_result["original_query"],
                        "reasoning": "STM exact hit (continuation)"}
            # LTM para continuation: solo si la query enriquecida coincide casi exactamente.
            # Umbral alto evita falsos positivos ("lima+arte" != "lima+arte+gpa").
            ltm_result = self._check_ltm(enriched_query, ltm, threshold=MEMORY_CONTINUATION_LTM_THRESHOLD)
            if ltm_result["hit"]:
                return {**fallback, "enriched_query": enriched_query,
                        "memory_cache_hit": True,
                        "memory_cached_result": ltm_result["cached_result"],
                        "memory_source": "long_term",
                        "memory_similarity": ltm_result["similarity"],
                        "memory_original_query": ltm_result["original_query"],
                        "reasoning": f"LTM exact hit (continuation) sim={ltm_result['similarity']:.3f}"}
        else:
            stm_result = self._check_stm(user_input, stm)
            _dbg(f"NEW_SQL STM_HIT={stm_result['hit']} SIM={stm_result.get('similarity',0):.3f}")
            if stm_result["hit"]:
                return {**fallback, "memory_cache_hit": True,
                        "memory_cached_result": stm_result["cached_result"],
                        "memory_source": "short_term",
                        "memory_similarity": stm_result["similarity"],
                        "memory_original_query": stm_result["original_query"],
                        "reasoning": f"STM hit sim={stm_result['similarity']:.3f}"}
            # LTM: umbral alto para evitar falsos positivos cuando la query tiene filtros extra
            ltm_result = self._check_ltm(user_input, ltm, threshold=MEMORY_CONTINUATION_THRESHOLD)
            ltm_miss_sim     = ltm_result.get("similarity", 0)
            ltm_miss_closest = ltm_result.get("original_query", "")
            if ltm_result["hit"]:
                return {**fallback, "memory_cache_hit": True,
                        "memory_cached_result": ltm_result["cached_result"],
                        "memory_source": "long_term",
                        "memory_similarity": ltm_result["similarity"],
                        "memory_original_query": ltm_result["original_query"],
                        "reasoning": f"LTM hit sim={ltm_result['similarity']:.3f}"}

        return {"intent": intent, "enriched_query": enriched_query,
                "context_switched": False, "rejection_reason": None,
                "reasoning": f"decision={decision} | {reasoning}",
                "memory_cache_hit": False, "memory_cached_result": None,
                "memory_source": None, "cached_sql": None,
                "ltm_miss_similarity": round(ltm_miss_sim, 4),
                "ltm_miss_closest": ltm_miss_closest,
                "ltm_threshold": MEMORY_SIMILARITY_THRESHOLD}

    # ------------------------------------------------------------------
    # LLM routing
    # ------------------------------------------------------------------

    def _route_with_llm(self, user_input: str, recent_successful: list,
                        db_type: str, history: list = None) -> dict:
        if recent_successful:
            *prev, current = recent_successful
            history_text = ""
            if prev:
                history_text += "  Historial previo:\n"
                history_text += "\n".join(f'    [{i+1}] "{q}"' for i, q in enumerate(prev))
                history_text += "\n"
            history_text += f'  ESTADO ACTUAL (base para continuation): "{current}"'
        else:
            history_text = "  (ninguna — primera pregunta de la sesion)"

        pending_clarif = self._get_pending_clarification(history or [])
        if pending_clarif:
            history_text += f'\n  [CLARIFICACION PENDIENTE] "{pending_clarif}"'

        system_prompt = ORCHESTRATOR_PROMPT.format(
            db_type=db_type.upper(), history_text=history_text)

        try:
            result = call_llm(
                self.client, self.model,
                messages=[{"role": "user", "content": f'Question: "{user_input}"'}],
                system=system_prompt,
                temperature=self.temperature, max_tokens=250,
            )
            raw = result["content"].strip()
            if raw.startswith("```"):
                import re
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                raw = m.group(0) if m else raw
            parsed   = json.loads(raw)
            decision = parsed.get("decision", "new_sql_query").lower().strip()
            if decision not in ("conversacional", "sustentation", "context_switch",
                                "continuation", "new_sql_query", "clarification_needed"):
                decision = "new_sql_query"
            reasoning = parsed.get("reasoning", "")

            # Guard inverso: continuation + enriched "X o Y" corto sin verbos
            # Solo sube a clarification_needed si el valor existente y el nuevo
            # aparecen en el MISMO contexto (misma palabra previa) → mismo campo.
            if decision == "continuation" and recent_successful:
                _eq_inv = (parsed.get("enriched_query") or "").strip()
                _eq_inv_lower = _eq_inv.lower()
                _last_q_inv = recent_successful[-1].lower()
                _state_filter_inv = any(fw in _last_q_inv for fw in (" que ", " con "))
                _action_inv = ("cuantos", "cuántos", "cuales", "cuáles", "dame",
                               "lista", "hay", "quienes", "quiénes")
                if (" o " in _eq_inv_lower
                        and len(_eq_inv.split()) <= 5
                        and not any(w in _eq_inv_lower for w in _action_inv)
                        and _state_filter_inv):
                    _parts_inv = _eq_inv_lower.split(" o ", 1)
                    if len(_parts_inv) == 2:
                        _va, _vb = _parts_inv[0].strip(), _parts_inv[1].strip()
                        _uin = user_input.lower()

                        def _word_before(text, val):
                            idx = text.find(val)
                            if idx > 0:
                                toks = text[:idx].strip().split()
                                return toks[-1] if toks else ""
                            return ""

                        _a_in_state = _va in _last_q_inv
                        _b_in_state = _vb in _last_q_inv
                        _same_ctx = False
                        if _a_in_state and not _b_in_state:
                            _same_ctx = _word_before(_last_q_inv, _va) == _word_before(_uin, _vb)
                        elif _b_in_state and not _a_in_state:
                            _same_ctx = _word_before(_last_q_inv, _vb) == _word_before(_uin, _va)
                        if _same_ctx:
                            decision = "clarification_needed"

            # Guardia clarification_needed: valida formato y que el estado tenga filtros.
            if decision == "clarification_needed":
                _eq = (parsed.get("enriched_query") or "").strip()
                _eq_lower = _eq.lower()
                # 1) Debe tener " o " entre las dos opciones
                _has_or = " o " in _eq_lower or " or " in _eq_lower
                # 2) Sin verbos de accion (el LLM no debe reconstruir la consulta entera)
                _action = ("cuantos", "cuántos", "cuales", "cuáles", "dame", "lista",
                           "quienes", "quiénes", "promedio", "maximo", "minimo")
                _has_action = any(w in _eq_lower for w in _action)
                # 3) Sin negaciones/placeholders inventados → señal de falso positivo
                _negation = (" no ", " sin ", "total", "otro lugar", "todos",
                             "sin restricci", "restricci")
                _has_neg = any(n in _eq_lower for n in _negation)
                # 4) El estado actual debe tener filtros reales (" que " o " con ")
                _last_q = (recent_successful[-1] if recent_successful else "").lower()
                _state_has_filter = any(fw in _last_q for fw in (" que ", " con "))
                # 5) Enriquecida corta: no debe ser una consulta completa reconstruida (max 8 palabras)
                _too_long = len(_eq.split()) > 8
                # 6) Cada parte alrededor de " o " debe tener max 3 palabras
                _parts_or = _eq_lower.replace("¿", "").replace("?", "").strip().split(" o ", 1)
                _parts_too_long = (len(_parts_or) == 2 and (
                    len(_parts_or[0].strip().split()) > 3 or
                    len(_parts_or[1].strip().split()) > 3))
                # 7) Los dos valores no deben ser de dominios claramente distintos
                # (curso vs ciudad → continuation, no clarification_needed)
                _cross_domain = False
                if len(_parts_or) == 2 and recent_successful:
                    _va7, _vb7 = _parts_or[0].strip(), _parts_or[1].strip()
                    _uin7 = user_input.lower()
                    def _wb(txt, v):
                        i = txt.find(v)
                        if i > 0:
                            t = txt[:i].strip().split()
                            return t[-1] if t else ""
                        return ""
                    _course_ctx = {"estudian", "estudia", "cursan", "cursa"}
                    _loc_ctx    = {"de", "del", "en", "desde"}
                    _a7_in = _va7 in _last_q
                    _b7_in = _vb7 in _last_q
                    if _a7_in and not _b7_in:
                        _cx_e, _cx_n = _wb(_last_q, _va7), _wb(_uin7, _vb7)
                    elif _b7_in and not _a7_in:
                        _cx_e, _cx_n = _wb(_last_q, _vb7), _wb(_uin7, _va7)
                    else:
                        _cx_e = _cx_n = ""
                    _cross_domain = (
                        (_cx_e in _course_ctx and _cx_n in _loc_ctx) or
                        (_cx_e in _loc_ctx and _cx_n in _course_ctx)
                    )
                if (not recent_successful or not _has_or
                        or _has_action or _has_neg or not _state_has_filter
                        or _too_long or _parts_too_long or _cross_domain):
                    decision = "continuation" if recent_successful else "new_sql_query"
                    parsed["enriched_query"] = user_input

            # Recuperacion de enriquecido para continuation cuando el LLM fallo.
            # Casos que disparan recuperacion:
            #   a) enriched vacio o igual al input crudo (LLM no enriqueció)
            #   b) enriched es un fragmento sin la entidad del estado base
            #      (LLM devolvio solo el nuevo filtro sin incluir el estado anterior)
            if decision == "continuation" and recent_successful:
                _ceq  = (parsed.get("enriched_query") or "").strip()
                _base = recent_successful[-1].strip("¿? ").strip()
                if not _ceq or _ceq.lower() == user_input.lower():
                    _raw   = user_input.strip()
                    _leads = sorted([
                        "y de esos cuantos", "de esos cuantos", "cuantos de esos",
                        "d esos cuantos",    "y de esos",       "de esos",
                        "d esos",            "de ellos cuantos", "cuantos de ellos",
                        "y cuantos",         "entonces dime cuantos", "entonces cuantos",
                        "dime cuantos",      "y dime cuantos",
                    ], key=len, reverse=True)
                    _filt = _raw
                    for _lead in _leads:
                        if _raw.lower().startswith(_lead):
                            _filt = _raw[len(_lead):].strip()
                            break
                    if _filt and _filt.lower() != _base.lower():
                        parsed["enriched_query"] = f"{_base} que {_filt}"
                    else:
                        parsed["enriched_query"] = _base

            # Post-processing: detectar concatenacion literal en vez de OR.
            # El LLM a veces ignora la regla OR y produce "base y de esos cuantos son de X"
            # en lugar de "base o de X". Lo detectamos y reconstruimos como OR.
            if decision == "continuation" and recent_successful:
                _eq_chk = (parsed.get("enriched_query") or "").strip()
                _bad_concat_pats = [
                    "y de esos cuantos son de", "y de esos cuántos son de",
                    "y de ellos cuantos son de", "y de ellas cuantos son de",
                    "y de esos cuantos hay en",  "y de esos cuántos hay en",
                ]
                if _eq_chk and any(p in _eq_chk.lower() for p in _bad_concat_pats):
                    _base_fix = recent_successful[-1].strip("¿? ").strip()
                    _fixed_or = self._reconstruct_as_or(_base_fix, _eq_chk)
                    if _fixed_or:
                        parsed["enriched_query"] = _fixed_or

            llm_enriched = (parsed.get("enriched_query") or "").strip().strip('"').strip("'")
            enriched = (llm_enriched
                        if llm_enriched and llm_enriched.lower() != user_input.lower()
                        else user_input)

            return {"decision": decision, "enriched_query": enriched, "reasoning": reasoning}

        except Exception as e:
            return {"decision": "new_sql_query", "enriched_query": user_input,
                    "reasoning": f"fallback: {e}"}

    # ------------------------------------------------------------------
    # Helpers de memoria / enriquecimiento
    # ------------------------------------------------------------------

    def _reconstruct_as_or(self, base: str, bad_eq: str) -> str:
        """
        Detecta "y de esos cuantos son de [ciudad]" en bad_eq y reconstruye
        el enriquecido como OR correcto usando el base del estado.

        Ejemplo:
          base   = "cuantos alumnos de berlin que estudian arte"
          bad_eq = "cuantos alumnos de berlin que estudian arte y de esos cuantos son de bogota"
          result = "cuantos alumnos de berlin o de bogota que estudian arte"
        """
        import re as _re
        # Extraer la nueva ciudad del patron "y de esos cuantos son de X"
        m = _re.search(
            r'y de esos (?:cuantos|cuántos) (?:son|hay) (?:de|en)\s+([a-záéíóúüñ\w]+)',
            bad_eq.lower()
        )
        if not m:
            return ""
        new_val = m.group(1).strip()
        # Stopwords que no son valores reales de dimension
        _stops = {"esos", "ellos", "ellas", "esas", "estos", "estas", "el", "la", "los", "las"}
        if new_val in _stops:
            return ""
        # Encontrar el primer "de [palabra]" en base que no sea stopword
        base_m = _re.search(
            r'(?:^|\s)de\s+(?!esos|ellos|ellas|esas|estos|estas)([a-záéíóúüñ\w]+)',
            base.lower()
        )
        if not base_m:
            return ""
        old_val = base_m.group(1).strip()
        if old_val == new_val:
            return ""
        # Reemplazar "de {old_val}" por "de {old_val} o de {new_val}" (primera ocurrencia)
        result = _re.sub(
            r'(?i)\bde\s+' + _re.escape(old_val) + r'\b',
            f'de {old_val} o de {new_val}',
            base,
            count=1
        )
        return result if result != base else ""

    def _check_stm(self, question: str, stm: dict,
                   exact_only: bool = False, threshold: float = None) -> dict:
        if not MEMORY_SHORT_TERM_ENABLED:
            return {"hit": False}
        return check_short_term_memory(
            question, stm,
            threshold=threshold if threshold is not None else MEMORY_SIMILARITY_THRESHOLD,
            exact_only=exact_only,
        )

    def _check_ltm(self, question: str, ltm, threshold: float = None) -> dict:
        if not MEMORY_LONG_TERM_ENABLED or not ltm:
            return {"hit": False}
        if not getattr(ltm, "enabled", False):
            return {"hit": False}
        t = threshold if threshold is not None else MEMORY_SIMILARITY_THRESHOLD
        return check_long_term_memory(question, ltm, threshold=t)

    def _get_last_successful_questions(self, history: list) -> list:
        # Recorre el historial en orden inverso buscando mensajes de sistema con
        # "SQL generado:" — esos corresponden a queries que terminaron exitosamente.
        # Extrae la pregunta enriquecida (no el SQL) para usarla como base del
        # contexto de enriquecimiento en _route_with_llm.
        recent = []
        for h in reversed(history[-20:]):
            if h.get("role") == "system" and "SQL generado:" in h.get("content", ""):
                content = h["content"]
                if "Pregunta:" in content:
                    q = content.split("Pregunta:")[1].split("->")[0].strip().strip('"')
                    recent.insert(0, q)
            if len(recent) >= 5:
                break
        return recent

    def _get_pending_clarification(self, history: list) -> str:
        # Busca si el turno anterior termino con una clarificacion pendiente.
        # Una clarificacion esta "resuelta" si despues de ella hay un mensaje "SQL generado:".
        # Si aun no se resolvio, retorna el texto de la clarificacion para que el
        # orquestador lo use como contexto al interpretar la respuesta del usuario.
        recent = history[-10:]
        clarif_idx = None
        clarif_value = ""
        for i, h in enumerate(recent):
            if h.get("role") == "system" and "Clarificacion pendiente:" in h.get("content", ""):
                clarif_idx = i
                clarif_value = h["content"].replace("Clarificacion pendiente:", "").strip().strip('"')
        if clarif_idx is None:
            return ""
        # Verificar si la clarificacion fue resuelta por una query exitosa posterior
        for h in recent[clarif_idx + 1:]:
            if h.get("role") == "system" and "SQL generado:" in h.get("content", ""):
                return ""
        return clarif_value

    def get_info(self):
        return {
            "name": "Orchestrator",
            "model": self.model,
            "agentes": ["AR", "APS", "AG", "AV", "AE", "AS"],
            "grafo": "orchestrator → AR → APS → AG → AV → AE → END",
        }
