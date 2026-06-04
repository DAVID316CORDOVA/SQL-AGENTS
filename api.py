"""
api.py

Backend FastAPI del sistema SQL-Agents. Expone el pipeline NL→SQL como API REST.

Gestiona sesiones por usuario: cada sesion mantiene su propio historial
conversacional, memoria de corto plazo (STM) y referencia a la memoria
de largo plazo (LTM en ChromaDB).

Rutas principales:
  POST /api/session       → crear sesion (usuario + db_type + dataset)
  POST /api/query         → procesar pregunta del usuario
  DELETE /api/session/:id → cerrar sesion y flush STM → LTM
  GET  /api/stats/:id     → estadisticas de la sesion (hits, rechazos, etc.)
  GET  /api/hints/:id     → consultas recientes de la LTM (sugerencias UI)
  POST /api/reindex       → re-indexar ChromaDB del APS para un dataset
  DELETE /api/memory/:id  → borrar LTM del usuario

El servidor no ejecuta SQL contra la BD: solo genera la query y la retorna
al cliente junto con la explicacion de AE y el indice de confianza WACS.
"""

import os
import sys
import uuid
import json
from datetime import datetime
from pathlib import Path

_BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_BASE_DIR))

# Cargar variables de entorno desde .env (API keys, configuración)
try:
    from dotenv import load_dotenv
    load_dotenv(_BASE_DIR / ".env")
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="SQL-Agents")
app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")

# APS y embeddings se inicializan lazy en la primera consulta

# Sesiones en memoria: session_id -> dict con estado
sessions = {}

_LAST_SESSIONS_FILE = _BASE_DIR / "memoria_chroma/last_sessions.json"

def _read_last_session(key: str) -> str:
    try:
        if _LAST_SESSIONS_FILE.exists():
            data = json.loads(_LAST_SESSIONS_FILE.read_text(encoding="utf-8"))
            return data.get(key, "")
    except Exception:
        pass
    return ""

def _write_last_session(key: str, dt: str):
    try:
        _LAST_SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if _LAST_SESSIONS_FILE.exists():
            data = json.loads(_LAST_SESSIONS_FILE.read_text(encoding="utf-8"))
        data[key] = dt
        _LAST_SESSIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ================================================================
# MODELOS
# ================================================================

class SessionCreate(BaseModel):
    username: str
    db_type: str        # "mysql" o "postgres"
    active_dataset: str = "demo_db"  # "demo_db" | "spider:<db_id>" | "wikisql"

class QueryRequest(BaseModel):
    session_id: str
    message: str

class ReindexRequest(BaseModel):
    db_type: str = "mysql"
    active_dataset: str = "demo_db"
    session_id: str = ""

class DatasetListResponse(BaseModel):
    datasets: list


# ================================================================
# HELPERS
# ================================================================

def _safe(d, key, default=None):
    if d is None:
        return default
    return d.get(key, default) if isinstance(d, dict) else default


def _handle_conversacional(user_input: str, session: dict) -> str:
    # Genera una respuesta de chat ligera para mensajes no-SQL (saludos, preguntas generales).
    # Usa el historial reciente como contexto para mantener coherencia conversacional.
    # No revela informacion del schema para evitar filtracion de metadatos de la BD.
    try:
        from llm_client import get_client, get_model, call_llm
        client = get_client()
        model  = get_model("classifier")
        history = session.get("conversation_history", [])
        hist_text = "\n".join(
            [f"[{h['role']}]: {h['content'][:80]}" for h in history[-10:]]
        ) if history else ""
        result = call_llm(
            client, model,
            messages=[{"role": "user", "content":
                f"Historial:\n{hist_text}\n\nEl usuario dice: \"{user_input}\"\n\nResponde brevemente:"}],
            system=(
                "You are the conversational layer of an NL-to-SQL system. "
                "You may respond to: greetings, small talk, and questions about how this system works. "
                "If the user asks anything outside those categories — recommendations, opinions, "
                "external information, or any topic unrelated to querying a database — politely "
                "explain that you can only answer questions about the data in the active database "
                "and invite them to ask a data question. "
                "Reply in the same language the user wrote. Maximum 3 sentences. "
                "Never reveal real table or column names."
            ),
            temperature=0.3, max_tokens=150,
        )
        return result["content"].strip()
    except Exception:
        return "No estoy seguro de cómo ayudarte con eso. Intenta hacerme una pregunta sobre la base de datos."


# ================================================================
# RUTAS
# ================================================================

@app.get("/")
def root():
    return FileResponse(str(_BASE_DIR / "static/index.html"))


def _apply_dataset(active_dataset: str):
    """Aplica el dataset activo: actualiza env, recarga config y limpia cache APS."""
    import importlib
    os.environ["ACTIVE_DATASET"] = active_dataset
    import config as _cfg
    importlib.reload(_cfg)
    # Limpiar cache APS para que tome el nuevo schema/vector_db
    try:
        import orchestrator.nodes as _nodes
        _nodes._aps_cache.clear()
    except Exception:
        pass
    return _cfg


@app.get("/api/datasets")
def list_datasets():
    """Lista los datasets disponibles agrupados por benchmark de origen."""
    import json as _json
    from pathlib import Path as _Path
    import os as _os

    # Base: la BD demo propia del sistema (escuela)
    datasets = [
        {"id": "demo_db", "label": "Demo DB (escuela)", "engine": "mysql", "group": ""},
    ]

    # Spider 1.0: enumerar db_ids desde el archivo de preguntas de experimentos.
    # Cada db_id es una BD diferente del benchmark (concert_singer, flight_2, etc.)
    spider_q = _BASE_DIR / "agents/MCP/metadata/mysql/spider/experiment_questions.json"
    if spider_q.exists():
        db_ids = sorted(set(q["db_id"] for q in _json.load(open(spider_q, encoding="utf-8"))))
        for db_id in db_ids:
            datasets.append({
                "id":     f"spider:{db_id}",
                "label":  db_id.replace("_", " ").title(),
                "engine": "both",
                "group":  "Spider 1.0",
            })

    # BIRD: enumerar subdirectorios que tengan schema.json (BD completa indexada)
    bird_path = _BASE_DIR / "agents/MCP/metadata/mysql/bird"
    if bird_path.exists():
        bird_dbs = sorted(
            d.name for d in bird_path.iterdir()
            if d.is_dir() and (d / "schema.json").exists()
        )
        for db_id in bird_dbs:
            datasets.append({
                "id":     f"bird:{db_id}",
                "label":  db_id.replace("_", " ").title(),
                "engine": "both",
                "group":  "BIRD",
            })

    return {"datasets": datasets}


@app.post("/api/session")
def create_session(data: SessionCreate):
    # Normalizar el nombre de usuario: mayusculas, max 20 chars
    username = data.username.strip().upper()[:20]
    if not username:
        raise HTTPException(status_code=400, detail="Nombre requerido")

    session_id = str(uuid.uuid4())

    # Aplicar el dataset activo antes de crear el agente APS para que
    # config.py resuelva los paths correctos (schema, chroma, diccionario)
    cfg = _apply_dataset(data.active_dataset)

    # Inicializar LTM: cada usuario+backend+dataset tiene su propio ChromaDB
    ltm = None
    try:
        from config import MEMORY_LONG_TERM_ENABLED
        if MEMORY_LONG_TERM_ENABLED:
            from memory.long_term_memory import LongTermMemory
            ltm = LongTermMemory(username, data.db_type, data.active_dataset)
    except Exception:
        pass

    # session_key identifica univocamente la combinacion usuario+backend+dataset
    # para recuperar el timestamp de la ultima sesion (mostrado en la UI)
    session_key  = f"{username}_{data.db_type}_{data.active_dataset}".lower().replace(":", "_")
    last_session = _read_last_session(session_key)

    # Estado completo de la sesion: todo lo que persiste entre queries del mismo usuario
    sessions[session_id] = {
        "username":               username,
        "db_type":                data.db_type,
        "active_dataset":         data.active_dataset,
        "session_key":            session_key,
        "conversation_history":   [],
        "short_term_cache":       {},
        "last_result":            None,
        "last_status":            None,
        "last_successful_result": None,
        "ltm":                    ltm,
        "session_start":          datetime.now().isoformat(),
        "last_session":           last_session,
        "q_success":              0,
        "q_rejected":             0,
        "q_error":                0,
    }

    return {
        "session_id":     session_id,
        "username":       username,
        "active_dataset": data.active_dataset,
        "database":       cfg.ACTIVE_DATABASE,
        "schema_path":    cfg.SCHEMA_PATH,
    }


@app.post("/api/query")
def process_query(data: QueryRequest):
    if data.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")

    session = sessions[data.session_id]
    user_input = data.message.strip()
    if not user_input:
        return {"type": "empty"}


    cmd = user_input.lower()

    # Comandos de control
    if cmd in ("nuevo tema", "nuevo", "reset", "cambiar tema"):
        session["last_result"] = None
        session["last_successful_result"] = None
        session["last_status"] = None
        session["conversation_history"].clear()
        return {"type": "system", "message": "Contexto reiniciado. Puedes preguntar sobre un tema nuevo."}

    if cmd in ("limpiar", "clear"):
        n = len(session["short_term_cache"])
        session["short_term_cache"].clear()
        return {"type": "system", "message": f"Memoria limpiada ({n} preguntas eliminadas)."}

    if cmd == "memoria":
        n_short = len(session["short_term_cache"])
        n_long = 0
        ltm = session.get("ltm")
        if ltm:
            try:
                stats = ltm.get_stats()
                n_long = stats.get("total", 0)
            except Exception:
                pass
        return {
            "type": "system",
            "message": f"Memoria de sesión: {n_short} pregunta(s).\nMemoria a largo plazo: {n_long} pregunta(s)."
        }

    if cmd in ("borrar memoria", "limpiar memoria", "formatear memoria"):
        n_short = len(session["short_term_cache"])
        session["short_term_cache"].clear()
        n_long = 0
        ltm = session.get("ltm")
        if ltm:
            try:
                stats = ltm.get_stats()
                n_long = stats.get("total", 0)
                ltm.clear()
            except Exception:
                pass
        session["last_result"] = None
        session["last_successful_result"] = None
        session["last_status"] = None
        session["conversation_history"].clear()
        return {
            "type": "system",
            "message": f"Memoria limpiada: {n_short} de sesión + {n_long} de largo plazo eliminadas."
        }

    # Aplicar dataset de la sesion antes de ejecutar
    _apply_dataset(session.get("active_dataset", "demo_db"))

    # Ejecutar grafo (el orquestador detecta context switch internamente)
    try:
        from orchestrator.graph import run_query

        result = run_query(
            user_input,
            memory_short_term=session["short_term_cache"],
            memory_long_term=session["ltm"],
            user_intent="otro",
            last_result_for_as=session["last_successful_result"],
            db_type=session["db_type"],
            conversation_history=session["conversation_history"],
        )

        enriched_query = result.get("user_input", user_input)
        result["user_input"] = user_input

        try:
            from utils.conversation_logger import log_query as _log_query
            _log_query(session["username"], "api", user_input, result, enriched_query=enriched_query)
        except Exception:
            pass

        # Conversacional
        if result.get("user_intent") == "conversacional":
            text = _handle_conversacional(user_input, session)
            session["conversation_history"].append({"role": "user", "content": user_input})
            session["conversation_history"].append({"role": "system", "content": f"Respuesta conversacional: {text[:80]}"})
            return {"type": "conversacional", "message": text}

        # Sustentacion
        if result.get("user_intent") == "sustentacion":
            ae = _safe(result, "ae_result", {})
            answer = _safe(ae, "explanation", {}).get("final_reasoning", "")
            return {"type": "sustentacion", "message": answer or "No hay una consulta anterior para explicar."}

        # Rechazada por orquestador
        if result.get("is_out_of_scope"):
            session["q_rejected"] += 1
            reason = result.get("rejection_reason", "fuera del alcance del sistema")
            return {"type": "rejected", "message": f"No puedo responder esa consulta: {reason}"}

        # Cambio de contexto detectado por el orquestador
        if result.get("context_switched"):
            session["last_result"] = None
            session["last_successful_result"] = None
            session["last_status"] = None
            session["conversation_history"].clear()
            # Si solo fue el reset sin consulta embebida, confirmar y salir
            enriched_after_switch = result.get("user_input", user_input)
            if not result.get("final_sql") and not result.get("ar_result"):
                return {"type": "system", "message": "Contexto reiniciado. Puedes preguntar sobre un tema nuevo."}

        # Extraer datos del resultado
        final_sql = result.get("final_sql", "")
        ar = _safe(result, "ar_result", {})
        ar_valid = ar.get("is_valid_query", True)
        aps = _safe(result, "aps_result", {})
        aps_success = aps.get("success", True)
        aps_below = aps.get("below_threshold", False)
        ae = _safe(result, "ae_result", {})

        # Actualizar estado de sesion
        if not ar_valid:
            session["last_status"] = "oe1_reject"
        elif not aps_success or aps_below:
            session["last_status"] = "oe2_reject"
        elif final_sql:
            session["last_status"] = "success"
            # Solo guardar campos serializables a JSON — el estado completo del grafo
            # incluye objetos Python no serializables (ToolRegistry, LTM, embeddings)
            # que causarían "Object of type ToolRegistry is not JSON serializable"
            # cuando se pasen al siguiente agente como last_result_for_as via HTTP.
            _SERIALIZABLE_KEYS = (
                "final_sql", "ar_result", "aps_result", "ag_result", "av_result",
                "ae_result", "overall_confidence", "user_input", "db_type",
                "iteration_count", "all_reasoning", "agent_times",
            )
            session["last_successful_result"] = {
                k: result.get(k) for k in _SERIALIZABLE_KEYS
            }
        else:
            session["last_status"] = "unknown_fail"
        session["last_result"] = result

        # Actualizar historial
        if final_sql:
            summary = f"Pregunta: \"{enriched_query}\" -> SQL generado: {final_sql[:80]}"
        elif not ar_valid:
            summary = f"Pregunta: \"{user_input}\" -> Rechazada (no es consulta BD)"
        elif not aps_success or aps_below:
            summary = f"Pregunta: \"{user_input}\" -> Rechazada (sin tablas relevantes)"
        else:
            summary = f"Pregunta: \"{user_input}\" -> Sin resultado"
        session["conversation_history"].append({"role": "user", "content": enriched_query})
        session["conversation_history"].append({"role": "system", "content": summary})

        # Sincronizar STM: memory_store_node actualiza el estado del grafo pero no el
        # dict de sesión. Sin este sync, close_session() flushea un dict vacío a LTM.
        new_stm = result.get("memory_short_term")
        if isinstance(new_stm, dict) and new_stm:
            session["short_term_cache"].update(new_stm)

        # Casos de rechazo/error
        if not ar_valid:
            session["q_rejected"] += 1
            if ar.get("confidence_score", 1.0) < 0.10:
                msg = "No puedo ejecutar operaciones de modificación de datos. Solo genero consultas de lectura (SELECT)."
            else:
                msg = "Esta pregunta está fuera del alcance del sistema. Solo puedo responder preguntas sobre los datos de la base de datos activa."
            return {"type": "error", "message": msg}

        if not aps_success or aps_below:
            session["q_rejected"] += 1
            ae_msg = _safe(ae, "explanation", {}).get("final_reasoning", "")
            no_info = aps.get("no_info_reason", "")
            msg = ae_msg or no_info or "La base de datos no tiene información sobre eso. No encontré columnas relacionadas con tu pregunta."
            # Si hay un resultado previo exitoso, ofrecerlo como referencia
            prev = session.get("last_successful_result")
            prev_sql = (prev or {}).get("final_sql", "") if prev else ""
            prev_query = ""
            if prev_sql and session.get("conversation_history"):
                for h in reversed(session["conversation_history"]):
                    if h.get("role") == "system" and "SQL generado:" in h.get("content", ""):
                        prev_query = h["content"].split("Pregunta:")[1].split("->")[0].strip().strip('"') if "Pregunta:" in h["content"] else ""
                        break
            return {
                "type": "no_column",
                "message": msg,
                "prev_sql": prev_sql,
                "prev_query": prev_query,
            }

        if ae.get("reformulation_reason") == "unfixable_schema_error":
            session["q_error"] += 1
            reason = (ae.get("explanation") or {}).get("final_reasoning", "")
            msg = reason if reason else "La base de datos no tiene esa información. No existe columna o tabla para ese filtro."
            return {"type": "error", "message": msg}

        if ae.get("reformulation_reason") == "max_retries_exceeded":
            session["q_error"] += 1
            return {"type": "error", "message": "No logré generar una consulta SQL confiable. Intenta reformular la pregunta de otra manera."}

        if not final_sql:
            session["q_error"] += 1
            return {"type": "error", "message": "No se pudo generar una consulta para esa pregunta."}

        # Formatear el SQL para legibilidad: cada clausula principal en nueva linea.
        # El SQL viene en una sola linea del agente; se inserta \n antes de cada keyword.
        keywords = ["SELECT ", "FROM ", "INNER JOIN ", "LEFT JOIN ", "RIGHT JOIN ",
                    "WHERE ", "GROUP BY ", "ORDER BY ", "HAVING ", "LIMIT ",
                    "AND ", "OR ", "ON ", "WITH ", "UNION "]
        sql_fmt = " ".join(final_sql.split())   # colapsar whitespace interno
        for kw in keywords:
            sql_fmt = sql_fmt.replace(kw, "\n" + kw)
        sql_fmt = sql_fmt.strip()

        explanation = _safe(ae, "explanation", {})
        explanation_text = explanation.get("final_reasoning", "") if explanation else ""
        confidence = result.get("overall_confidence", 0)
        cache_hit = result.get("memory_cache_hit", False)
        cache_source = result.get("memory_source", "")

        session["q_success"] += 1
        return {
            "type": "success",
            "sql": sql_fmt,
            "explanation": explanation_text,
            "confidence": round(confidence * 100, 1),
            "cache_hit": cache_hit,
            "cache_source": cache_source,
            "context_switched": result.get("context_switched", False),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"type": "error", "message": f"Error interno: {str(e)}"}


@app.get("/api/hints/{session_id}")
def get_hints(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    session = sessions[session_id]
    hints = []
    ltm = session.get("ltm")
    if ltm:
        try:
            hints = ltm.get_recent(5)
        except Exception:
            pass
    return {"hints": hints}


@app.post("/api/reset/{session_id}")
def reset_context(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    session = sessions[session_id]
    session["last_result"] = None
    session["last_successful_result"] = None
    session["last_status"] = None
    session["conversation_history"].clear()
    return {"ok": True}


@app.get("/api/stats/{session_id}")
def get_stats(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    session = sessions[session_id]
    n_short = len(session.get("short_term_cache", {}))
    n_long = 0
    ltm = session.get("ltm")
    if ltm:
        try:
            stats = ltm.get_stats()
            n_long = stats.get("total", 0)
        except Exception:
            pass
    q_success  = session.get("q_success", 0)
    q_rejected = session.get("q_rejected", 0)
    q_error    = session.get("q_error", 0)
    return {
        "short_term":    n_short,
        "long_term":     n_long,
        "session_start": session.get("session_start", ""),
        "last_session":  session.get("last_session", ""),
        "q_total":       q_success + q_rejected + q_error,
        "q_success":     q_success,
        "q_rejected":    q_rejected,
        "q_error":       q_error,
    }


@app.post("/api/reindex")
def reindex(data: ReindexRequest):
    try:
        import shutil, gc
        # Resolver dataset desde sesión si se provee session_id
        active_dataset = data.active_dataset
        db_type = data.db_type
        if data.session_id and data.session_id in sessions:
            s = sessions[data.session_id]
            active_dataset = s.get("active_dataset", active_dataset)
            db_type = s.get("db_type", db_type)

        cfg = _apply_dataset(active_dataset)
        # Elegir el schema correcto segun el backend solicitado.
        # config.py expone SCHEMA_PATH (mysql) y PG_SCHEMA_PATH (postgres),
        # cada uno apunta a su carpeta independiente en MCP/metadata/.
        schema_path = cfg.PG_SCHEMA_PATH if db_type == "postgres" else cfg.SCHEMA_PATH

        # Forzar GC para que ChromaDB libere los archivos antes de borrar
        gc.collect()
        if os.path.exists(cfg.VECTOR_DB_PATH):
            try:
                shutil.rmtree(cfg.VECTOR_DB_PATH)
            except OSError:
                pass  # Archivo en uso — reindexar sobre los datos existentes
        from agents.APS import SchemaMatcherAgent
        agent = SchemaMatcherAgent(
            schema_path=schema_path,
            vector_db_path=cfg.VECTOR_DB_PATH,
            db_type=db_type,
        )
        count = agent.tables_col.count() if hasattr(agent, "tables_col") else 0
        return {
            "ok": True,
            "tables_indexed": count,
            "db_type": db_type,
            "active_dataset": active_dataset,
            "schema_path": schema_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/memory/{session_id}")
def clear_memory(session_id: str, scope: str = "all"):
    """Borra LTM. scope='hour' = última hora, scope='all' = todo."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    session = sessions[session_id]
    ltm = session.get("ltm")
    if not ltm:
        return {"ok": True, "deleted": 0}
    try:
        if scope == "hour":
            deleted = ltm.clear_last_hour()
        else:
            deleted = ltm.clear()
        return {"ok": True, "deleted": deleted, "scope": scope}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
def close_session(session_id: str):
    if session_id not in sessions:
        return {"ok": False}
    session = sessions[session_id]
    # Flush memoria corto a largo plazo
    try:
        ltm = session.get("ltm")
        cache = session.get("short_term_cache", {})
        if ltm and ltm.enabled and cache:
            for key, entry in cache.items():
                if entry.get("result") and entry.get("query"):
                    ltm.store(entry["query"], entry["result"],
                              original_query=entry.get("original_query"))
    except Exception:
        pass
    # Guardar timestamp de cierre como "última sesión"
    _write_last_session(session.get("session_key", ""), datetime.now().isoformat())
    del sessions[session_id]
    return {"ok": True}
