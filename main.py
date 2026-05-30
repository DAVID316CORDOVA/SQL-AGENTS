"""
main.py

Punto de entrada del sistema NL->SQL multi-agente.

Flujo:
  1. Cache RAM exacto (0ms)
  2. Clasificador: conversacional vs query (~1-2s)
  3. Enriquecimiento: resolver referencias ("dame su correo") (~1-2s)
  4. Grafo LangGraph: memory_check -> AR -> APS -> AG -> AV -> AE
  5. Si SQL exitoso: guardar resumen en TXT + sustentador bajo demanda
  6. Al cerrar sesion: flush corto plazo a ChromaDB

Salida normal: solo SQL generado (sin confianza ni explicacion)
Modo debug: muestra confianza por agente + diagnostico

Comandos:
  salir          Cerrar sesion y guardar memoria
  debug          Activar/desactivar analisis por agente
  nuevo tema     Reiniciar contexto del sustentador
  memoria        Ver estado de memorias
  limpiar        Limpiar memoria corto plazo
  test           Bateria de pruebas
  extract        Extraer schema de BD
  reindex        Reindexar ChromaDB

Ubicacion: main.py (raiz)
"""

import sys
import os
import re
import time
import hashlib
import shutil
from datetime import datetime

from config import (
    VECTOR_DB_PATH, MAX_RETRIES, CONFIDENCE_WEIGHTS,
    MEMORY_SHORT_TERM_ENABLED, MEMORY_LONG_TERM_ENABLED,
    EMBEDDING_MODEL,
    ORCHESTRATOR_SUMMARY_MAX_TOKENS, SUMMARIES_PATH
)

def format_sql(sql):
    """Formatea SQL para impresion legible sin cortar nada."""
    if not sql:
        return sql
    # Limpiar whitespace excesivo
    sql = " ".join(sql.split())
    # Agregar saltos de linea antes de keywords principales
    keywords = [
        "SELECT ", "FROM ", "INNER JOIN ", "LEFT JOIN ", "RIGHT JOIN ",
        "WHERE ", "GROUP BY ", "ORDER BY ", "HAVING ", "LIMIT ",
        "AND ", "OR ", "ON ", "UNION ", "WITH ", "AS ("
    ]
    result = sql
    for kw in keywords:
        result = result.replace(kw, "\n" + kw)
    # Limpiar primer salto de linea
    result = result.strip()
    return result


# ================================================================
# PREGUNTAS DE PRUEBA
# ================================================================

TEST_QUERIES = [
    {"query": "quien es el mejor jugador del mundo",
     "esperado": "AR rechaza", "escenario": "AR rechaza"},
    {"query": "cual es la capital de Francia",
     "esperado": "AR rechaza", "escenario": "AR rechaza"},
    {"query": "cuantos hospitales hay en la ciudad",
     "esperado": "APS rechaza", "escenario": "APS rechaza"},
    {"query": "cuantos estudiantes hay",
     "esperado": "COUNT sobre students", "escenario": "Simple"},
    {"query": "dame los nombres de los cursos",
     "esperado": "SELECT course_name FROM courses", "escenario": "Simple"},
    {"query": "cuales son los emails de los profesores",
     "esperado": "SELECT email FROM professors", "escenario": "Simple"},
    {"query": "que cursos tiene cada estudiante",
     "esperado": "JOIN students-enrollments-courses", "escenario": "JOIN"},
    {"query": "cuantos estudiantes tiene cada profesor",
     "esperado": "JOIN professors-courses-enrollments", "escenario": "JOIN"},
    {"query": "cual es el promedio de notas por curso",
     "esperado": "AVG(grade) GROUP BY course", "escenario": "Agregacion"},
    {"query": "cuantos estudiantes activos hay por ciudad",
     "esperado": "COUNT WHERE status GROUP BY city", "escenario": "Agregacion"},
    {"query": "dame el ranking de estudiantes por nota en cada curso",
     "esperado": "ROW_NUMBER PARTITION BY course", "escenario": "Window"},
    {"query": "cuales son los 3 mejores estudiantes de cada departamento",
     "esperado": "Top N por grupo con CTE", "escenario": "Window"},
    {"query": "dame todos los datos de todos",
     "esperado": "AR rechaza: ambigua", "escenario": "Trampa-vaga"},
    {"query": "borra todos los estudiantes inactivos",
     "esperado": "AR rechaza: DELETE", "escenario": "Trampa-DML"},
]

SEP = "=" * 150
SEP2 = "-" * 150
SEP_STAR = "*" * 150
SEP_AGENT = "=" * 100


# ================================================================
# NORMALIZACION DE NOMBRE
# ================================================================

def normalize_username(raw):
    cleaned = re.sub(r'[_\-]+', ' ', raw.strip())
    cleaned = re.sub(r'[^\w\s]', '', cleaned).strip()
    if not cleaned:
        return "USUARIO"
    parts = cleaned.split()
    if len(parts) == 1:
        return parts[0].upper()
    return " ".join(w.capitalize() for w in parts[:2])


# ================================================================
# EMBEDDING HELPER
# ================================================================

def get_embedding(text):
    try:
        from utils.embeddings import get_embedding as _get_emb
        return _get_emb(text, EMBEDDING_MODEL)
    except Exception:
        return None


# ================================================================
# HELPERS DE IMPRESION
# ================================================================

def _safe(d, key, default=None):
    if d is None:
        return default
    return d.get(key, default) if isinstance(d, dict) else default


def print_sql(sql, indent=4):
    prefix = " " * indent
    formatted = format_sql(sql)
    for line in formatted.split("\n"):
        print(f"{prefix}{line}")


def print_result(result, show_analysis=False, last_successful_result=None):
    """
    Salida LIMPIA:
    - Rechazos: mensaje amigable
    - SQL exitoso: solo pregunta + SQL (sin confianza, sin explicacion)
    - Debug: agrega confianza por agente
    """
    user_input = result.get("user_input", "")
    final_sql = result.get("final_sql", "")
    ar = _safe(result, "ar_result", {})
    aps = _safe(result, "aps_result", {})
    ae = _safe(result, "ae_result", {})

    ar_valid = ar.get("is_valid_query", True)
    aps_success = aps.get("success", True)
    aps_below = aps.get("below_threshold", False)

    # --- CASO 1: AR rechazo ---
    if not ar_valid:
        print(f"\n  No puedo generar una consulta SQL para esa pregunta.")
        reasoning = ar.get("reasoning", "")
        if "opinion" in reasoning.lower() or "general" in reasoning.lower():
            print(f"  Parece ser una pregunta de conocimiento general o de opinion.")
        elif "delete" in reasoning.lower() or "update" in reasoning.lower():
            print(f"  Solo puedo realizar consultas de lectura (SELECT).")
        else:
            print(f"  La pregunta no parece estar relacionada con la base de datos.")
        print(f"  Intenta con una pregunta que solicite datos de la base de datos.")
        if show_analysis:
            print_agent_analysis(result)
        return

    # --- CASO 2: APS rechazo ---
    if not aps_success or aps_below:
        # Intentar explicar con razonamiento de AR si hay pistas del campo faltante
        ar_refined = ar.get("refined_query", "")
        aps_tables = aps.get("tables", {})
        aps_columns = aps.get("columns", [])

        # Mensaje principal de rechazo
        no_info = aps.get("no_info_reason", "")
        if no_info:
            print(f"\n  {no_info}")
        else:
            print(f"\n  Lo siento, la base de datos no tiene informacion sobre eso.")
            print(f"  No encontre columnas o tablas relacionadas con tu pregunta.")

        # Si hay un resultado exitoso previo, ofrecerlo como alternativa
        if last_successful_result:
            prev_sql = last_successful_result.get("final_sql", "")
            prev_input = last_successful_result.get("user_input", "")
            if prev_sql:
                print(f"\n  Lo mas cercano que puedo ofrecerte es la consulta anterior:")
                if prev_input:
                    print(f"    Pregunta: \"{prev_input}\"")
                print(f"  SQL disponible:")
                print_sql(prev_sql)
                print(f"\n  Puedes seguir haciendome preguntas sobre esos datos.")
        if show_analysis:
            print_agent_analysis(result)
        return

    # --- CASO 3: Error no corregible de schema (columna/tabla inexistente) ---
    requires_reform = ae.get("requires_reformulation", False)
    reform_reason = ae.get("reformulation_reason", "")
    if requires_reform and reform_reason == "unfixable_schema_error":
        explanation = (ae.get("explanation") or {}).get("final_reasoning", "")
        msg = explanation if explanation else "La base de datos no tiene informacion para ese filtro."
        print(f"\n  {msg}")
        return

    # --- CASO 4: Max reintentos ---
    if requires_reform and reform_reason == "max_retries_exceeded":
        print(f"\n  No logre generar una consulta SQL confiable para esa pregunta.")

        # Prioridad 1: errores criticos de AV (son los mas especificos)
        oe5_errors = _safe(result, "av_result", {}).get("errors", [])
        if oe5_errors:
            for err in oe5_errors[:2]:
                print(f"  {err}")

        # Prioridad 2: razonamiento de AG si menciona limitacion del schema
        oe4_reasoning = _safe(result, "ag_result", {}).get("reasoning", "")
        if oe4_reasoning and ("no existe" in oe4_reasoning.lower() or
                              "no es posible" in oe4_reasoning.lower() or
                              "no hay columna" in oe4_reasoning.lower() or
                              "eliminado" in oe4_reasoning.lower()):
            reason_clean = oe4_reasoning.replace("[CORRECCION]", "").strip()
            if len(reason_clean) > 20 and not oe5_errors:
                print(f"  Motivo: {reason_clean[:600]}")

        # Si no hay errores especificos, dar sugerencia generica
        if not oe5_errors and not oe4_reasoning:
            print(f"  Intenta reformularla de otra manera o ser mas especifico.")

        if show_analysis:
            print_agent_analysis(result)
        return

    # --- CASO 4: Sin SQL ---
    if not final_sql:
        print(f"\n  No se pudo generar una consulta para esa pregunta.")
        print(f"  Intenta reformular tu pregunta o ser mas especifico.")
        if show_analysis:
            print_agent_analysis(result)
        return

    # --- CASO 5: SQL exitoso ---
    # Modo normal: SQL + explicacion de AE + disclaimer
    explanation = _safe(ae, "explanation", {})

    print(SEP2)
    print(f"\n  Pregunta: {user_input}")
    # Mostrar pregunta refinada si AR la cambio
    refined = ar.get("refined_query", "")
    if refined and refined.lower().strip() != user_input.lower().strip():
        print(f"  Refinada: {refined}")
    print(f"\n  SQL generado:")
    print_sql(final_sql)

    # Explicacion de AE (siempre en modo normal)
    if explanation.get("final_reasoning"):
        print(f"\n  Explicacion:")
        for line in explanation["final_reasoning"].split("\n"):
            line = line.strip()
            if line:
                print(f"    {line}")

    # Disclaimer (solo en modo normal, no en debug)
    if not show_analysis:
        print(f"\n  * El sistema puede cometer errores. Verifica las respuestas.")

    print(SEP2)

    # Solo en debug se muestra confianza y analisis
    if show_analysis:
        print_agent_analysis(result)


def print_agent_analysis(result):
    """Solo se muestra en modo debug."""
    print(f"\n{SEP2}")
    print(f"  ANALISIS POR AGENTE")
    print(SEP2)

    # Razonamiento del Orquestador
    all_reasoning = result.get("all_reasoning", [])
    orch_lines = [r for r in all_reasoning if "[ORCHESTRATOR]" in r]
    if orch_lines:
        print(f"\n  [ORCHESTRATOR]")
        for line in orch_lines:
            clean = line.replace("[ORCHESTRATOR]", "").strip()
            print(f"    {clean}")
        ctx = "Si" if result.get("context_switched") else "No"
        scope = "RECHAZADA" if result.get("is_out_of_scope") else "Aceptada"
        print(f"    Cambio de contexto: {ctx} | Consulta: {scope}")
        if result.get("rejection_reason"):
            print(f"    Motivo rechazo: {result['rejection_reason']}")
        if result.get("memory_cache_hit"):
            src = result.get("memory_source", "memoria")
            print(f"    Respuesta desde: {src}")

    ar = _safe(result, "ar_result", {})
    aps = _safe(result, "aps_result", {})
    ag = _safe(result, "ag_result", {})
    av = _safe(result, "av_result", {})

    arc = ar.get("confidence_score", 0)
    ar_valid = ar.get("is_valid_query")
    apsc = aps.get("confidence_score", 0)
    aps_success = aps.get("success", False)
    aps_below = aps.get("below_threshold", False)
    aps_bypassed = aps.get("bypassed", False)

    # AR
    if not ar_valid:
        rej = 1.0 - arc
        print(f"\n  [AR - Refiner] Rechazo (confianza del rechazo: {rej:.1%})")
        print(f"    La pregunta no es una consulta de base de datos.")
    else:
        print(f"\n  [AR - Refiner] Confianza: {arc:.1%}")
        print(f"    Valida: Si")
        if ar.get("refined_query"):
            print(f"    Refinada: {ar['refined_query']}")
        ar_skills = ar.get("skills_used")
        if ar_skills:
            print(f"    Skills: {', '.join(ar_skills)}")
    print(f"  {SEP_AGENT}")

    # APS
    if not ar_valid:
        print(f"\n  [APS - Proximidad] No ejecutado")
    elif not aps_success or aps_below:
        rej = 1.0 - apsc
        threshold_used = aps.get("threshold_used", 0.35)
        print(f"\n  [APS - Proximidad] Rechazo (confianza del rechazo: {rej:.1%})")
        print(f"    Similitud coseno: {apsc:.1%} (umbral: {threshold_used:.0%})")
        missing = aps.get("missing_columns", [])
        if missing:
            print(f"    Columnas faltantes: {', '.join(missing)}")
    elif aps_success:
        tag = " [BYPASS]" if aps_bypassed else (" [CONTEXTO-CONT]" if aps.get("continuation_context") else "")
        print(f"\n  [APS - Proximidad] Similitud: {apsc:.1%}{tag}")
        tables = aps.get("tables") or {}
        for tn, ti in tables.items():
            sim = ti.get("similarity", ti.get("similarity_score", 0))
            injected = " [INYECTADA]" if ti.get("_injected") else ""
            cols = ti.get("all_columns", [])
            print(f"    Tabla: {tn} (sim: {sim:.4f}){injected}")
            if cols:
                print(f"      Columnas: {', '.join(cols[:10])}")
        joins = aps.get("joins", [])
        if joins:
            for j in joins:
                print(f"    JOIN: {j.get('join_hint', '')}")
        aps_skills = aps.get("skills_used", ["search_tables", "search_columns", "find_joins"])
        print(f"    Skills: {', '.join(aps_skills)}")
    print(f"  {SEP_AGENT}")

    # AG - mostrar SQL generado Y el que fallo
    agc = ag.get("confidence_score", 0)
    if ag.get("success"):
        print(f"\n  [AG - SQL Generator] Confianza: {agc:.1%}")
        print(f"    Estrategia: {ag.get('strategy', 'N/A')}")
        ag_skills = ag.get("skills_used", ["validate_sql_safety", "fix_reserved_words"])
        print(f"    Skills: {', '.join(ag_skills)}")
        print(f"    SQL:")
        print_sql(ag.get("sql", ""), indent=6)
        if ag.get("reasoning"):
            print(f"    Razonamiento: {ag['reasoning'][:500]}")
    elif ar_valid and aps_success:
        print(f"\n  [AG - SQL Generator] Fallo ({result.get('iteration_count', 3)} intentos)")
        if ag.get("sql"):
            print(f"    Ultimo SQL intentado:")
            print_sql(ag.get("sql", ""), indent=6)
        if ag.get("reasoning"):
            print(f"    Razon: {ag['reasoning'][:600]}")
        if ag.get("error"):
            print(f"    Error: {ag['error'][:300]}")
    else:
        print(f"\n  [AG - SQL Generator] No ejecutado")
    print(f"  {SEP_AGENT}")

    # AV - mostrar errores detallados
    avc = av.get("confidence_score", 0)
    if av.get("success") is not None and ag.get("success"):
        print(f"\n  [AV - SQL Validator] Confianza: {avc:.1%}")
        av_skills = av.get("skills_used", ["check_syntax_rules", "explain_query"])
        print(f"    Skills: {', '.join(av_skills)}")
        print(f"    Valida: {'Si' if av.get('is_valid') else 'No'}")

        explain = av.get("explain_result", {})
        if explain.get("executed"):
            rows = explain.get("estimated_rows", 0)
            full_scan = explain.get("has_full_scan", False)
            analyze = explain.get("analyze_executed", False)
            actual_time = explain.get("actual_time_ms")
            needs_opt = explain.get("needs_optimization", False)
            print(f"    [EXPLAIN] Filas estimadas: {rows} | Full scan: {'Si' if full_scan else 'No'}")
            if analyze and actual_time is not None:
                print(f"    [ANALYZE] Tiempo real: {actual_time:.1f}ms | Optimizar: {'Si' if needs_opt else 'No'}")

        for w in av.get("warnings", []):
            print(f"    [WARN] {w}")
        for e in av.get("errors", []):
            print(f"    [ERROR] {e}")
        if av.get("feedback_for_ag"):
            print(f"    Feedback a AG: {av['feedback_for_ag'][:500]}")
    elif ag.get("success"):
        print(f"\n  [AV - SQL Validator] Fallo")
    else:
        print(f"\n  [AV - SQL Validator] No ejecutado")
    print(f"  {SEP_AGENT}")

    # Tabla de confianza
    final_sql = result.get("final_sql", "")
    is_rejection = not ar_valid or not aps_success or aps_below or not final_sql
    overall = result.get("overall_confidence", 0)

    if is_rejection:
        print(f"\n  [RESULTADO] Consulta rechazada")
        if not ar_valid:
            print(f"    Motivo: AR rechazo la pregunta (no es consulta de BD)")
        elif not aps_success or aps_below:
            print(f"    Motivo: APS no encontro tablas/columnas relevantes")
        elif not final_sql:
            iters = result.get("iteration_count", 0)
            print(f"    Motivo: AG+AV no lograron SQL valido en {iters} intentos")
            print(f"    Calidad del mejor SQL intentado: {overall:.1%} (umbral minimo no alcanzado)")
    else:
        agents = [
            ("AR - Refiner", arc, "AR", True),
            ("APS - Proximidad", apsc, "APS", True),
            ("AG - SQL Generator", agc, "AG", True),
            ("AV - SQL Validator", avc, "AV", True),
        ]
        print(f"\n  [TABLA DE CONFIANZA]")
        print(f"    {'Agente':<25} {'Confianza':>10} {'Peso':>6} {'Ponderado':>10}")
        print(f"    {'-'*55}")
        for name, conf, wk, executed in agents:
            w = CONFIDENCE_WEIGHTS.get(wk, 0)
            print(f"    {name:<25} {conf:>9.1%} {w:>5.0%} {conf*w:>10.3f}")
        print(f"    {'-'*55}")
        print(f"    {'TOTAL':<25} {'':>10} {'':>6} {overall:>9.1%}")

    # Tiempos por agente
    agent_times = result.get("agent_times", {})
    if agent_times:
        print(f"\n  [TIEMPOS POR AGENTE]")
        total_t = 0
        for agent_name in ["orchestrator", "AR", "APS", "AG", "AV", "AE"]:
            t = agent_times.get(agent_name, 0)
            if t > 0:
                print(f"    {agent_name:<20} {t:>6.1f}s")
                total_t += t
        print(f"    {'-'*28}")
        print(f"    {'TOTAL':<20} {total_t:>6.1f}s")


# ================================================================
# RESUMEN DEL ORQUESTADOR (se guarda en TXT, no usa LLM)
# ================================================================

def build_orchestrator_summary(result):
    """
    Genera un resumen de texto concatenando lo que cada agente retorno.
    NO usa LLM — es concatenacion programatica.
    Se guarda en TXT para que el sustentador lo lea bajo demanda.
    Controlado por ORCHESTRATOR_SUMMARY_MAX_TOKENS en config.py.
    """
    lines = []

    ar = _safe(result, "ar_result", {})
    aps = _safe(result, "aps_result", {})
    ag = _safe(result, "ag_result", {})
    av = _safe(result, "av_result", {})

    # AR
    lines.append(f"[AR - Refinador] Confianza: {ar.get('confidence_score', 0):.1%}")
    lines.append(f"  Pregunta refinada: \"{ar.get('refined_query', '')}\"")
    if ar.get("reasoning"):
        lines.append(f"  Razonamiento: {ar['reasoning'][:200]}")

    # APS
    apsc = aps.get("confidence_score", 0)
    tables = aps.get("tables", {})
    lines.append(f"\n[APS - Proximidad] Similitud: {apsc:.1%}")
    if tables:
        lines.append(f"  Tablas: {', '.join(tables.keys())}")
    joins = aps.get("joins", [])
    if joins:
        hints = [j.get("join_hint", "") for j in joins[:3]]
        lines.append(f"  Joins: {'; '.join(hints)}")

    # AG
    lines.append(f"\n[AG - Generador] Confianza: {ag.get('confidence_score', 0):.1%}")
    lines.append(f"  Estrategia: {ag.get('strategy', 'N/A')}")
    lines.append(f"  SQL: {ag.get('sql', '')[:200]}")
    if ag.get("reasoning"):
        lines.append(f"  Razonamiento: {ag['reasoning'][:150]}")

    # AV
    lines.append(f"\n[AV - Validador] Confianza: {av.get('confidence_score', 0):.1%}")
    lines.append(f"  Valida: {'Si' if av.get('is_valid') else 'No'}")
    errors = av.get("errors", [])
    if errors:
        lines.append(f"  Errores: {'; '.join(errors[:2])}")
    iters = result.get("iteration_count", 0)
    if iters > 1:
        lines.append(f"  Iteraciones: {iters}")

    # Global
    lines.append(f"\n[GLOBAL] Confianza: {result.get('overall_confidence', 0):.1%}")

    summary = "\n".join(lines)

    # Truncar por tokens parametrizables
    max_chars = ORCHESTRATOR_SUMMARY_MAX_TOKENS * 4
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n[... truncado]"

    return summary


def save_summary_to_file(username, user_input, final_sql, summary):
    """
    Guarda el resumen del orquestador en un archivo TXT.
    Estructura: summaries/{username}/{timestamp}_{hash}.txt
    """
    try:
        user_dir = os.path.join(SUMMARIES_PATH, username)
        os.makedirs(user_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = hashlib.md5(user_input.encode()).hexdigest()[:8]
        filename = f"{timestamp}_{query_hash}.txt"
        filepath = os.path.join(user_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Usuario: {username}\n")
            f.write(f"Fecha: {datetime.now().isoformat()}\n")
            f.write(f"Pregunta: {user_input}\n")
            f.write(f"SQL: {final_sql}\n")
            f.write(f"\n{'='*50}\n")
            f.write(f"RESUMEN DEL PROCESO:\n")
            f.write(f"{'='*50}\n\n")
            f.write(summary)

        return filepath
    except Exception:
        return None


def get_latest_summary(username):
    """Lee el ultimo resumen guardado para un usuario."""
    try:
        user_dir = os.path.join(SUMMARIES_PATH, username)
        if not os.path.exists(user_dir):
            return None
        files = sorted(os.listdir(user_dir), reverse=True)
        if not files:
            return None
        filepath = os.path.join(user_dir, files[0])
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None




def handle_conversational(user_input, last_status, last_result, history=None):
    """Responde preguntas sobre el sistema."""
    try:
        from llm_client import get_client, get_model, call_llm
        _client = get_client()
        model = get_model("classifier")

        hist_text = ""
        if history:
            hist_lines = [f"[{h['role']}]: {h['content'][:80]}" for h in history[-10:]]
            hist_text = "\n".join(hist_lines)

        _result = call_llm(
            _client, model,
            messages=[{"role": "user", "content": f"""History:
{hist_text if hist_text else "No history."}

The user says: "{user_input}"

Reply briefly:"""}],
            system="""You are an assistant for an NL-to-SQL system.
Respond in English, maximum 3 lines. Be friendly.
If the user wants to change topic, tell them to type "new topic".
NEVER reveal real table or column names from the database.
NEVER invent data or claim whether records exist or not.
If the question is about why a query failed, say the system could not
generate the SQL and suggest rephrasing the question. Do not invent reasons.""",
            temperature=0.3, max_tokens=150,
        )
        answer = _result["content"].strip()
        print(f"\n  {answer}")

        if history is not None:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "system", "content": f"Respuesta conversacional: {answer[:80]}"})
    except Exception:
        print(f"\n  No estoy seguro de como ayudarte con eso.")
        print(f"  Intenta hacerme una pregunta sobre la base de datos.")


def handle_sustentation(user_input, username, last_result, sustainer):
    """Llama al agente sustentador para explicar la ultima query."""
    if not last_result or not last_result.get("final_sql"):
        print(f"\n  No hay una consulta anterior para explicar.")
        print(f"  Primero hazme una pregunta que genere SQL.")
        return

    # Verificar permisos via Tool Registry
    try:
        from orchestrator.graph import get_registry
        reg = get_registry()
        if reg:
            reg.execute("AS", "sustain_query")
    except PermissionError:
        print(f"\n  Sustentador no disponible (permiso denegado).")
        return
    except Exception:
        pass  # Si no hay registry, continuar normal

    if sustainer:
        answer = sustainer.process(last_result, user_input)
        print(f"\n  Sustentacion:")
        for line in answer.split("\n"):
            line = line.strip()
            if line:
                print(f"    {line}")
    else:
        print(f"\n  Agente sustentador no disponible.")





# ================================================================
# EXTRACT / REINDEX
# ================================================================

def do_extract():
    from agents.APS.schema_extractor import SchemaExtractor
    extractor = SchemaExtractor()
    extractor.extract_and_save()
    print("Schema extraido.")

def do_reindex():
    import shutil
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        print(f"  ChromaDB borrada: {VECTOR_DB_PATH}")
    # Limpiar cache de APS para que la siguiente consulta use el indice nuevo
    try:
        from orchestrator import nodes as _nodes
        _nodes._aps_cache.clear()
        print(f"  Cache APS limpiado.")
    except Exception:
        pass
    from agents.APS import SchemaMatcherAgent
    agent = SchemaMatcherAgent()
    count = agent.tables_col.count() if hasattr(agent, 'tables_col') else 0
    print(f"  Reindexado completado. Tablas indexadas: {count}")
    try:
        test = agent._search_tables("cuantos alumnos hay", n=5)
        print(f"  Prueba 'cuantos alumnos hay':")
        for t in test:
            print(f"    {t['table_name']}: sim={t['similarity']:.3f}")
    except Exception as e:
        print(f"  Error en prueba: {e}")


# ================================================================
# TEST
# ================================================================

def do_test(verbose=False):
    from orchestrator.graph import run_query

    print(SEP)
    print(f"  BATERIA DE PRUEBAS ({len(TEST_QUERIES)} preguntas)")
    print(SEP)

    summary = []
    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        escenario = test["escenario"]

        print(f"\n  [{i}/{len(TEST_QUERIES)}] [{escenario}] {query[:50]}")
        start = time.time()
        try:
            result = run_query(query)
            elapsed = time.time() - start
            final_sql = result.get("final_sql", "")
            confidence = result.get("overall_confidence", 0)
            ar_valid = _safe(_safe(result, "ar_result", {}), "is_valid_query", True)

            if not ar_valid:
                status = "RECHAZO"
            elif final_sql:
                status = f"SQL OK ({confidence:.1%})"
            else:
                status = "SIN SQL"

            print(f"    {status} ({elapsed:.1f}s)")
            if verbose and final_sql:
                print_sql(final_sql, indent=6)
            summary.append({"status": status, "confidence": confidence, "time": elapsed})
        except Exception as e:
            print(f"    ERROR: {e}")
            summary.append({"status": "ERROR", "confidence": 0, "time": 0})

    total = len(summary)
    ok = sum(1 for r in summary if "OK" in r["status"])
    t_total = sum(r["time"] for r in summary)
    avg_conf = sum(r["confidence"] for r in summary if "OK" in r["status"]) / max(ok, 1)
    print(f"\n  SQL generado: {ok}/{total} | Confianza: {avg_conf:.1%} | Tiempo: {t_total:.1f}s")
    print(SEP)


# ================================================================
# FLUSH MEMORY
# ================================================================

def flush_short_to_long(short_term_cache, ltm):
    if not ltm or not ltm.enabled or not short_term_cache:
        return 0
    count = 0
    for key, entry in short_term_cache.items():
        result = entry.get("result")
        query = entry.get("query", "")
        if result and query:
            ltm.store(query, result, original_query=entry.get("original_query"))
            count += 1
    return count


# ================================================================
# MAIN
# ================================================================

def main():
    try:
        import readline  # noqa
    except ImportError:
        try:
            import pyreadline3  # noqa
        except ImportError:
            pass

    print(SEP)
    print("  Sistema NL->SQL Multi-Agente")
    print("  Flujo: [ORQUESTADOR] -> [MEMORY] -> AR -> APS -> AG -> AV -> AE")
    print(f"  Pesos: AR={CONFIDENCE_WEIGHTS['AR']:.0%} APS={CONFIDENCE_WEIGHTS['APS']:.0%} "
          f"AG={CONFIDENCE_WEIGHTS['AG']:.0%} AV={CONFIDENCE_WEIGHTS['AV']:.0%}")
    print(SEP)

    # --- Pedir nombre ---
    while True:
        try:
            raw_name = input("\n  Ingresa tu nombre: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            return
        if not raw_name:
            print("  Por favor ingresa un nombre.")
            continue
        if "?" in raw_name or len(raw_name) > 40:
            print("  Por favor ingresa un nombre valido.")
            continue
        words = raw_name.replace("_", " ").replace("-", " ").split()
        if len(words) > 3:
            print("  Ingresa solo tu nombre (maximo 2 palabras).")
            continue
        break

    username = normalize_username(raw_name)
    print(f"\n  Bienvenido {username}!")

    from utils.conversation_logger import log_session_start, log_session_end, log_query, log_event

    # --- Seleccion de motor de base de datos ---
    print(f"\n  Selecciona el motor de base de datos:")
    print(f"    1. MySQL")
    print(f"    2. PostgreSQL")
    db_type = "mysql"
    while True:
        try:
            db_choice = input("\n  Opcion (1/2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            return
        if db_choice == "1":
            db_type = "mysql"
            print(f"  Motor seleccionado: MySQL")
            break
        elif db_choice == "2":
            db_type = "postgres"
            print(f"  Motor seleccionado: PostgreSQL")
            break
        else:
            print(f"  Por favor ingresa 1 (MySQL) o 2 (PostgreSQL).")

    # --- Seleccion de base de datos / dataset ---
    import json as _json
    from pathlib import Path as _Path

    _spider_dbs = []
    _spider_q_path = _Path("agents/MCP/metadata/mysql/spider/experiment_questions.json")
    if _spider_q_path.exists():
        _spider_dbs = sorted(set(
            q["db_id"] for q in _json.load(open(_spider_q_path, encoding="utf-8"))
        ))

    print(f"\n  Selecciona la base de datos:")
    print(f"    1. demo_db  (base de datos propia - escuela)")
    print(f"    2. wikisql  (WikiSQL dataset)")
    for i, db_id in enumerate(_spider_dbs, 3):
        print(f"    {i}. spider:{db_id}")

    _max_opt = 2 + len(_spider_dbs)
    active_dataset = "demo_db"
    while True:
        try:
            _db_choice = input(f"\n  Opcion (1-{_max_opt}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            return
        if _db_choice == "1":
            active_dataset = "demo_db"
            break
        elif _db_choice == "2":
            active_dataset = "wikisql"
            break
        elif _db_choice.isdigit() and 3 <= int(_db_choice) <= _max_opt:
            active_dataset = f"spider:{_spider_dbs[int(_db_choice) - 3]}"
            break
        else:
            print(f"  Opcion invalida. Ingresa un numero entre 1 y {_max_opt}.")

    # Aplicar dataset seleccionado
    import os as _os
    _os.environ["ACTIVE_DATASET"] = active_dataset
    import importlib as _imp
    import config as _cfg
    _imp.reload(_cfg)
    print(f"  Dataset activo: {active_dataset}")
    print(f"  BD: {_cfg.ACTIVE_DATABASE} | Schema: {_cfg.SCHEMA_PATH}")

    # APS y embeddings se inicializan en la primera consulta (lazy load)

    # --- Memoria largo plazo (ChromaDB) ---
    # NO se carga ni consulta al inicio. Solo se instancia.
    # La busqueda ocurre cuando el usuario hace una pregunta.
    ltm = None
    if MEMORY_LONG_TERM_ENABLED:
        from memory.long_term_memory import LongTermMemory
        from cleanup_memory import register_user
        ltm = LongTermMemory(username)
        register_user(username)

    # --- Agente sustentador (vive fuera del grafo) ---
    sustainer = None
    try:
        from agents.AS.sustainer_agent import SustainerAgent
        sustainer = SustainerAgent()
    except ImportError:
        pass

    # --- Orquestador con estado de sesion (STM, historial, sustentador) ---
    from orchestrator.orchestrator_agent import OrchestratorAgent
    orch = OrchestratorAgent(db_type=db_type, long_term_memory=ltm, sustainer=sustainer)

    # --- Estado de display ---
    show_analysis = False
    last_result = None
    last_status = None

    log_session_start(username, "main", db_type)
    print(f"\n  Motor activo: {db_type.upper()}")
    print(f"  Indicanos tu pregunta sobre la base de datos.")
    print(f"  Si quieres cambiar de tema, escribe 'nuevo tema'.")
    print(f"  Comandos: salir, debug, nuevo tema, memoria, limpiar, test, extract, reindex")
    print(SEP)

    while True:
        try:
            user_input = input(f"\n  {username}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  Cerrando sesion...")
            saved = orch.close_session()
            if saved > 0:
                print(f"  [MEMORIA] {saved} preguntas guardadas en largo plazo (ChromaDB)")
            print(f"  Hasta luego {username}!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("salir", "exit", "quit"):
            saved = orch.close_session()
            if saved > 0:
                print(f"\n  [MEMORIA] {saved} preguntas guardadas en largo plazo (ChromaDB)")
            log_session_end(username, "main", saved)
            print(f"  Hasta luego {username}!")
            break

        if cmd == "debug":
            show_analysis = not show_analysis
            from agents.base_skill_agent import SkillAgent
            SkillAgent.VERBOSE = show_analysis
            print(f"  Analisis por agente: {'ACTIVADO' if show_analysis else 'DESACTIVADO'}")
            continue
        if cmd == "extract":
            do_extract()
            continue
        if cmd == "reindex":
            do_reindex()
            continue
        if cmd == "test":
            do_test(verbose=show_analysis)
            continue
        if cmd == "memoria":
            print(f"\n  Memoria corto plazo (sesion): {len(orch.stm)} preguntas")
            if orch.stm:
                for k in orch.stm:
                    print(f"    - \"{k}\"")
            if ltm:
                stats = ltm.get_stats()
                print(f"  Memoria largo plazo (ChromaDB): {stats['total']} preguntas ({username})")
            if orch.sustainer:
                print(f"  Contexto sustentador: {orch.sustainer.get_info()['context_entries']} interacciones")
            continue
        if cmd in ("limpiar", "clear"):
            n = len(orch.stm)
            orch.stm.clear()
            print(f"  Memoria corto plazo limpiada ({n} preguntas).")
            continue
        if cmd == "registry":
            try:
                from orchestrator.graph import get_registry
                reg = get_registry()
                if reg:
                    info = reg.get_info()
                    print(f"\n  Tool Registry:")
                    print(f"    Tools registradas: {info['total_tools']}")
                    print(f"    Ejecuciones totales: {info['total_executions']}")
                    print(f"    Ejecuciones denegadas: {info['denied_executions']}")
                    print(f"\n  Permisos por agente:")
                    for agent, tools in info["agents"].items():
                        print(f"    {agent}: {', '.join(tools)}")
                    denied = reg.get_denied_executions()
                    if denied:
                        print(f"\n  Ejecuciones DENEGADAS:")
                        for d in denied:
                            print(f"    [{d['agent']}] intento usar '{d['tool']}' → {d['reason']}")
                else:
                    print(f"  Registry no disponible")
            except Exception as e:
                print(f"  Error: {e}")
            continue
        if cmd in ("nuevo tema", "nuevo", "cambiar tema", "reset"):
            orch.reset_context()
            last_result = None
            last_status = None
            print(f"  Contexto reiniciado. Puedes preguntar sobre un tema nuevo.")
            continue

        print("\n  Procesando...")
        start = time.time()

        try:
            result = orch.process(user_input)
            enriched_query = result.get("enriched_query") or result.get("user_input", user_input)
            if show_analysis and enriched_query.lower() != user_input.lower():
                print(f"  [DEBUG] Enriquecida: \"{enriched_query}\"")
            result["user_input"] = user_input
            elapsed = time.time() - start

            if result.get("user_intent") == "conversacional":
                handle_conversational(user_input, last_status, last_result, orch.conversation_history)
                print(f"\n  {SEP_STAR}")
                continue

            if result.get("user_intent") == "clarification_needed":
                print(f"\n  {enriched_query}")
                print(f"\n  {SEP_STAR}")
                continue

            if result.get("user_intent") == "sustentacion":
                ae = result.get("ae_result") or {}
                answer = (ae.get("explanation") or {}).get("final_reasoning", "")
                if answer:
                    print(f"\n  Sustentacion:")
                    for line in answer.split("\n"):
                        line = line.strip()
                        if line:
                            print(f"    {line}")
                else:
                    print(f"\n  No hay una consulta anterior para explicar.")
                print(f"\n  {SEP_STAR}")
                continue

            if result.get("is_out_of_scope"):
                reason = result.get("rejection_reason", "fuera del alcance del sistema")
                print(f"\n  No puedo responder esa consulta: {reason}")
                print(f"\n  {SEP_STAR}")
                continue

            if show_analysis and result.get("context_switched"):
                print("  [ORCHESTRATOR] Cambio de tema detectado - historial reiniciado")

            if result.get("memory_cache_hit"):
                sim = result.get("memory_similarity", 0)
                orig = result.get("memory_original_query", "")
                source = result.get("memory_source", "?")
                if show_analysis:
                    print(f"\n  [MEMORIA] Respuesta de {source} (sim: {sim:.1%})")
                    if orig and orig.lower() != user_input.lower():
                        print(f"  Pregunta similar: \"{orig}\"")

                try:
                    from orchestrator.graph import get_registry
                    reg = get_registry()
                    if reg and not reg.can_use("AE", "explain_query"):
                        ae = result.get("ae_result", {})
                        if ae:
                            exp = ae.get("explanation", {})
                            if exp:
                                exp["final_reasoning"] = "Explicacion no disponible."
                except Exception:
                    pass

                if result.get("final_sql"):
                    orch_summary = build_orchestrator_summary(result)
                    save_summary_to_file(username, user_input, result.get("final_sql", ""), orch_summary)

                print_result(result, show_analysis=show_analysis,
                             last_successful_result=orch._last_successful_result)
                if show_analysis:
                    print(f"  Tiempo: {elapsed:.1f}s (cache)")

                last_result = result
                last_status = "success" if result.get("final_sql") else "unknown_fail"
                print(f"\n  {SEP_STAR}")
                continue

            # Resultado del pipeline completo
            final_sql = result.get("final_sql", "")
            ar_valid = (result.get("ar_result") or {}).get("is_valid_query", False)
            aps_success = (result.get("aps_result") or {}).get("success", True)
            aps_below = (result.get("aps_result") or {}).get("below_threshold", False)

            if final_sql and ar_valid:
                orch_summary = build_orchestrator_summary(result)
                save_summary_to_file(username, user_input, final_sql, orch_summary)

            log_query(username, "main", user_input, result, enriched_query=enriched_query)
            print_result(result, show_analysis=show_analysis,
                         last_successful_result=orch._last_successful_result)
            print(f"\n  Tiempo: {elapsed:.1f}s")

            if show_analysis:
                aps_s = (result.get("aps_result") or {}).get("success")
                aps_b = (result.get("aps_result") or {}).get("below_threshold")
                iters = result.get("iteration_count", 0)
                print(f"\n  [DIAGNOSTICO]")
                print(f"    AR valida: {ar_valid} | APS success: {aps_s} | APS below: {aps_b}")
                print(f"    Iteraciones: {iters} | SQL: {'Si' if final_sql else 'No'} | "
                      f"Tiempo total: {elapsed:.1f}s")

            reform_reason = (result.get("ae_result") or {}).get("reformulation_reason", "")
            if not ar_valid:
                last_status = "oe1_reject"
            elif not aps_success or aps_below:
                last_status = "oe2_reject"
            elif reform_reason in ("max_retries_exceeded", "unfixable_schema_error"):
                last_status = "max_retries"
            elif final_sql:
                last_status = "success"
            else:
                last_status = "unknown_fail"

            last_result = result

            if show_analysis and ar_valid and last_status == "success":
                print(f"  [MEMORIA] Guardada en corto plazo ({len(orch.stm)} total)")

            print(f"\n  {SEP_STAR}")

        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  Error: {e} ({elapsed:.1f}s)")
            import traceback
            traceback.print_exc()
            print(f"\n  {SEP_STAR}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--extract" in args:
        do_extract()
    elif "--reindex" in args:
        do_reindex()
    elif "--test" in args:
        do_test(verbose="--verbose" in args)
    elif "--eval" in args:
        from evaluator import run_evaluation
        run_evaluation()
    else:
        main()