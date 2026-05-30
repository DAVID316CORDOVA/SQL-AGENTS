# -*- coding: utf-8 -*-
"""
agents/AG/mysql/phase_1/main.py

Evaluacion del AG (Generador de SQL) — Fase 1: busqueda de hiperparametros
con Optuna.

Usa el dataset propio de phase_1 (copia del original de phase_2): 11 preguntas
Spider concert_singer con `input_to_ag` (mock del APS) y `expected_sql` (gold).

Metricas (mismo esquema que AR Fase 1, adaptado a SQL):
  - Faithfulness  : GEval (LLM-juez gpt-4o) sobre (generated_sql, expected_sql)
  - Groundedness  : GEval sobre (input, generated_sql) — no inventa entidades
  - ROUGE-L (SQL) : rouge_l_sql() con normalizacion de SQL (sin LLM-juez)
  - Brier         : sklearn sobre (confidence, outcome=R>=0.80)
  - ECE           : NumPy idem

Funcion objetivo (Optuna maximiza):
    combined_score = F + G + R_sql - Brier - ECE   (max 3.0)
"""

import os
import sys
import json
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

from deepeval.test_case.llm_test_case import LLMTestCase

from metricas_lib import (
    faithfulness_geval,
    groundedness_geval,
    rouge_l_sql,
    brier_score,
    ece,
)
from llm_client import get_client_for_model

# ─────────────────────────────────────────────────────────────────────
# Configuracion
# ─────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")

# Umbral de outcome para Brier/ECE: outcome=1 si rouge_l_sql >= 0.80
ROUGE_THRESHOLD_FOR_OUTCOME = 0.70


# ─────────────────────────────────────────────────────────────────────
# Carga del dataset
# ─────────────────────────────────────────────────────────────────────

def load_questions(dataset_path: str = DATASET_PATH) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# ─────────────────────────────────────────────────────────────────────
# Evaluacion de UNA pregunta
# ─────────────────────────────────────────────────────────────────────

def _build_aps_result(question: dict) -> dict:
    """
    Construye el dict que el AG espera (formato aps_result) desde el
    formato simplificado del dataset (refined_query, selected_tables,
    selected_columns).

    Las "selected_columns" del dataset estan en formato APS nativo
    (lista de {table_name, column_name}). Las agrupamos por tabla para
    armar tables_dict, donde cada tabla solo expone como `all_columns`
    las columnas seleccionadas para esa tabla (no todas las del schema).
    """
    tables_dict = {}
    for col in question.get("selected_columns", []):
        tname = col["table_name"]
        cname = col["column_name"]
        tables_dict.setdefault(tname, {"all_columns": []})
        if cname not in tables_dict[tname]["all_columns"]:
            tables_dict[tname]["all_columns"].append(cname)

    # Aseguramos que cada tabla seleccionada tenga entrada (aunque no
    # haya columnas — caso defensivo, no deberia pasar con dataset bien armado)
    for t in question.get("selected_tables", []):
        tables_dict.setdefault(t["table_name"], {"all_columns": []})

    return {
        "success":     True,
        "input_query": question["refined_query"],
        "tables":      tables_dict,
        "joins":       [],          # Fase 1 no le pasa joins
        "original_intent": {},      # Fase 1 no le pasa intent
    }


def _evaluate_one(agent, question: dict,
                  f_metric, g_metric,
                  temperature: float) -> dict:
    """
    Construye el aps_result desde el dataset simplificado, llama
    AG.process(aps_result) y compara generated_sql vs expected_sql.
    Registra las skills invocadas por el LLM durante el loop.
    """
    aps_input    = _build_aps_result(question)
    expected_sql = question["expected_sql"]
    user_query   = question["refined_query"]

    # ── Tracking de skills invocadas ──────────────────────────────────
    skills_called: list[dict] = []
    _original_execute = agent._execute_skill

    def _tracking_execute(skill_name: str, args: dict) -> str:
        result = _original_execute(skill_name, args)
        try:
            result_parsed = json.loads(result)
        except Exception:
            result_parsed = result
        skills_called.append({
            "skill":    skill_name,
            "args":     args,
            "response": result_parsed,
        })
        return result

    agent._execute_skill = _tracking_execute
    # ─────────────────────────────────────────────────────────────────

    t0 = time.time()
    try:
        r = agent.process(aps_result=aps_input, temperature=temperature)
        elapsed = time.time() - t0
        generated_sql = r.get("sql", "") or ""
        confidence    = float(r.get("confidence_score", 0.5))
        success       = bool(r.get("success", False))
    except Exception as e:
        agent._execute_skill = _original_execute  # restaurar siempre
        return {
            "input_query":   user_query,
            "expected_sql":  expected_sql,
            "generated_sql": "",
            "confidence":    0.5,
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome": 0,
            "time_s": 0.0,
            "skills_called": skills_called,
            "error": str(e),
        }
    finally:
        agent._execute_skill = _original_execute  # restaurar siempre

    # Si AG fallo (validacion safety, etc), score 0 sin llamar al juez
    if not success or not generated_sql:
        return {
            "input_query":   user_query,
            "expected_sql":  expected_sql,
            "generated_sql": generated_sql,
            "confidence":    confidence,
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome": 0,
            "time_s": round(elapsed, 2),
            "error": r.get("error", "AG returned success=False"),
        }

    # 1) Faithfulness: ¿el SQL generado es fiel al SQL esperado?
    tc_f = LLMTestCase(
        input=user_query,
        actual_output=generated_sql,
        expected_output=expected_sql,
    )
    f_metric.measure(tc_f)
    F = float(f_metric.score or 0.0)

    # 2) Groundedness: ¿el SQL generado se apoya en la pregunta original?
    g_metric.measure(tc_f)
    G = float(g_metric.score or 0.0)

    # 3) ROUGE-L sobre SQL normalizado
    R = rouge_l_sql(generated_sql, expected_sql)
    outcome = 1 if R >= ROUGE_THRESHOLD_FOR_OUTCOME else 0

    # Resumen legible de skills llamadas (sin args para no inflar el JSON)
    skills_summary = [s["skill"] for s in skills_called]

    return {
        "input_query":   user_query,
        "expected_sql":  expected_sql,
        "generated_sql": generated_sql,
        "confidence":    confidence,
        "F": round(F, 3),
        "G": round(G, 3),
        "R": round(R, 3),
        "outcome": outcome,
        "time_s": round(elapsed, 2),
        "skills_called": skills_summary,
        "skills_detail": skills_called,
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────
# Evaluacion completa (1 trial = 1 combinacion model+temperature)
# ─────────────────────────────────────────────────────────────────────

def evaluate_ag(model: str = "gpt-4o-mini",
                temperature: float = 0.1,
                questions: list = None,
                verbose: bool = False) -> dict:
    """
    Evalua el AG con (model, temperature) sobre las 11 preguntas.

    Devuelve dict con metricas agregadas + per_question.

    Las metricas LLM-juez (F, G) se construyen UNA vez antes del loop;
    Brier y ECE se calculan UNA vez al final sobre las listas paralelas.
    """
    from agents.AG.mysql.sql_generator_agent import SQLGeneratorAgent

    if questions is None:
        questions = load_questions()

    agent = SQLGeneratorAgent()
    agent.model  = model
    # Refresca el cliente segun el modelo (gpt-* OpenAI, claude-* Anthropic, etc)
    agent.client = get_client_for_model(model)

    f_metric = faithfulness_geval()
    g_metric = groundedness_geval()

    F_list, G_list, R_list = [], [], []
    confidences, outcomes  = [], []
    per_question = []

    for q in questions:
        result = _evaluate_one(agent, q, f_metric, g_metric, temperature)
        per_question.append(result)

        F_list.append(result["F"])
        G_list.append(result["G"])
        R_list.append(result["R"])
        confidences.append(result["confidence"])
        outcomes.append(result["outcome"])

        if verbose:
            err_tag = f"  ERROR={result['error'][:30]}" if result.get("error") else ""
            idx = len(per_question) - 1
            skills_str = ", ".join(result.get("skills_called", [])) or "—"
            print(f"  [{idx:>2}] F={result['F']:.3f}  "
                  f"G={result['G']:.3f}  R={result['R']:.3f}  "
                  f"conf={result['confidence']:.2f}  "
                  f"outcome={result['outcome']}  "
                  f"t={result['time_s']:.2f}s  "
                  f"skills=[{skills_str}]{err_tag}")

    n = len(per_question)
    avg = lambda xs: round(sum(xs) / n, 3) if n else 0.0

    F_avg = avg(F_list)
    G_avg = avg(G_list)
    R_avg = avg(R_list)

    B = round(brier_score(confidences, outcomes), 4)
    E = round(ece(confidences, outcomes, n_bins=5), 3)

    combined = round(F_avg + G_avg + R_avg - B - E, 3)

    avg_time = round(
        sum(r["time_s"] for r in per_question) / n, 2
    ) if n else 0.0

    return {
        "model":          model,
        "temperature":    temperature,
        "n_questions":    n,
        "Faithfulness":   F_avg,
        "Groundedness":   G_avg,
        "ROUGE-L_SQL":    R_avg,
        "Brier":          B,
        "ECE":            E,
        "combined_score": combined,
        "avg_time_s":     avg_time,
        "per_question":   per_question,
    }


def save_results(metrics: dict, tag: str = "single") -> str:
    """Guarda metrics en results/<tag>_<timestamp>.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{tag}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
