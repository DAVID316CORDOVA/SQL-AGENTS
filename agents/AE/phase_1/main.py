# -*- coding: utf-8 -*-
"""
agents/AE/phase_1/main.py

Evaluacion del AE (Agente Explicador) — Fase 1: busqueda de hiperparametros
con Optuna.

AE recibe un estado completo del pipeline (refined_query + tables/columns +
generated_sql) y produce una explicacion en lenguaje natural sobre lo que
hace la SQL.

Metricas (mismas que AR — output NL):
  - Faithfulness  : GEval (juez gpt-4o) (explanation, expected_explanation)
  - Groundedness  : GEval (refined_query + sql, explanation)
  - ROUGE-L       : Scorer DeepEval determinista
  - Brier         : sklearn sobre (confidence_score, outcome=F>=0.80)
  - ECE           : NumPy idem

confidence_score se deriva del confidence_level que reporta el AE
(alta=0.90, media=0.60, baja=0.30) — el AE no emite un numero directo.

Funcion objetivo (asesor 2026-05-13):
    combined_score = (2*F + ROUGE-L + 2*G - ECE - Brier) / 5   (max 1.0)
    F y G pesan doble por ser metricas semanticas LLM-as-judge.

Outcome para calibracion:
    outcome = 1 si F >= 0.80   (semantico, igual que AR)
"""

import os
import sys
import json
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
    rouge_l_score,
    brier_score,
    ece,
)
from llm_client import get_client_for_model

# ─────────────────────────────────────────────────────────────────────
# Configuracion
# ─────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")

# Outcome para Brier/ECE: G (Groundedness) >= 0.80 — mas estable en paraphrases
GROUNDEDNESS_THRESHOLD_FOR_OUTCOME = 0.80

# Mapeo del confidence_level categorico del AE a float numerico
_CONFIDENCE_LEVEL_MAP = {
    "alta":   0.90,
    "high":   0.90,
    "media":  0.60,
    "medium": 0.60,
    "baja":   0.30,
    "low":    0.30,
}


# ─────────────────────────────────────────────────────────────────────
# Carga del dataset
# ─────────────────────────────────────────────────────────────────────

def load_samples(dataset_path: str = DATASET_PATH) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]


def _detect_scenario(sample: dict) -> str:
    """
    Infiere el escenario del pipeline a partir de los campos presentes
    en la muestra, sin depender de un campo 'scenario' explicito.

      ar_rejection_reason                          → AR_FAILURE
      aps_rejection_reason + last_useful_sql        → CHAIN_REJECTION
      aps_rejection_reason                          → APS_FAILURE
      (av_iterations | last_av_error) + sin SQL     → AG_AV_EXHAUSTED
      (av_iterations | av_first_error) + con SQL    → SUCCESS con correccion
      en caso contrario                             → SUCCESS limpio
    """
    if sample.get("ar_rejection_reason"):
        return "AR_FAILURE"
    if sample.get("aps_rejection_reason") and sample.get("last_useful_sql"):
        return "CHAIN_REJECTION"
    if sample.get("aps_rejection_reason"):
        return "APS_FAILURE"
    # AG_AV_EXHAUSTED: hay iteraciones/error AV pero NO hay SQL final
    if (sample.get("av_iterations") or sample.get("last_av_error")) and not sample.get("generated_sql"):
        return "AG_AV_EXHAUSTED"
    # SUCCESS (limpio o con correccion automatica — ambos tienen generated_sql)
    return "SUCCESS"


def _build_full_state(sample: dict) -> dict:
    """
    Construye el dict que AE.process(full_state) espera, desde el formato
    simplificado del dataset. Soporta los 5 escenarios del AE:
      SUCCESS | APS_FAILURE | AR_FAILURE | AG_AV_EXHAUSTED | CHAIN_REJECTION
    """
    scenario = _detect_scenario(sample)

    # Construir tables_dict desde selected_columns/tables (puede estar vacio)
    tables_dict = {}
    for col in sample.get("selected_columns", []):
        tname = col["table_name"]
        cname = col["column_name"]
        tables_dict.setdefault(tname, {"all_columns": []})
        if cname not in tables_dict[tname]["all_columns"]:
            tables_dict[tname]["all_columns"].append(cname)
    for t in sample.get("selected_tables", []):
        tables_dict.setdefault(t["table_name"], {"all_columns": []})

    # user_input: puede venir del campo "user_input" (fallos AR) o "refined_query"
    user_input = sample.get("user_input") or sample.get("refined_query", "")

    # ---- AR ----
    # AR real devuelve: success(siempre True), is_valid_query, refined_query, reasoning, confidence_score
    if scenario == "AR_FAILURE":
        ar_result = {
            "success":        True,          # AR siempre retorna success=True
            "is_valid_query": False,         # pero is_valid_query=False cuando rechaza
            "refined_query":  "",
            "confidence_score": 0.1,
            "reasoning": sample.get("ar_rejection_reason", "Input cannot be interpreted as a database query."),
        }
    else:
        ar_result = {
            "success":        True,
            "is_valid_query": True,
            "refined_query":  sample.get("ar_refined_query") or sample.get("refined_query", user_input),
            "confidence_score": 0.95,
            "reasoning":      "Valid database query identified.",
        }

    # ---- APS ----
    # APS real devuelve: success, below_threshold, no_info_reason, tables, confidence_score
    if scenario in ("APS_FAILURE", "CHAIN_REJECTION"):
        aps_result = {
            "success":              False,
            "below_threshold":      True,
            "below_threshold_reason": "tables",
            "tables":               {},
            "confidence_score":     0.0,
            "no_info_reason":       sample.get("aps_rejection_reason", "No related tables found in schema."),
        }
    else:
        aps_result = {
            "success":        True,
            "tables":         tables_dict,
            "confidence_score": 0.85,
        }

    # ---- AG ----
    final_sql = sample.get("generated_sql", "")
    ag_result = {
        "success":          bool(final_sql),
        "sql":              final_sql,
        "confidence_score": 1.0 if final_sql else 0.0,
        "strategy":         sample.get("ag_strategy", ""),   # e.g. simple|join|aggregation|subquery
    }

    # ---- AV ----
    # av_reasoning: lo que el AV reporto como justificacion (exito o fallo).
    # Se pasa al AE para que lo use como fundamento de la explicacion.
    av_reasoning = sample.get("av_reasoning", "")

    if scenario == "AG_AV_EXHAUSTED":
        n_iter   = sample.get("av_iterations", 3)
        last_err = sample.get("last_av_error", "Query generation failed after maximum retries.")
        av_result = {
            "success":          False,
            "is_valid":         False,
            "unfixable":        True,
            "needs_correction": False,
            "errors":           [last_err],
            "reasoning":        av_reasoning or last_err,
        }
        iteration_history  = [{"iteration": i + 1, "error": last_err} for i in range(n_iter)]
        overall_confidence = 0.2
    elif final_sql:
        # SUCCESS: limpio (0 iteraciones previas) o con correccion automatica (>=1)
        # av_first_error: error del primer intento que el AV rechazo antes de corregir
        n_corrections = sample.get("av_iterations", 0)
        first_err     = sample.get("av_first_error", "")
        av_result = {
            "success":          True,
            "is_valid":         True,
            "unfixable":        False,
            "needs_correction": False,
            "errors":           [],
            "reasoning":        av_reasoning or "The SQL passed all validation checks.",
        }
        # Construir historial de iteraciones fallidas previas al exito
        if n_corrections > 0 and first_err:
            iteration_history = [{"iteration": i + 1, "error": first_err}
                                  for i in range(n_corrections)]
        else:
            iteration_history = []
        overall_confidence = 1.0
    else:
        av_result = {
            "success":          False,
            "is_valid":         False,
            "unfixable":        False,
            "needs_correction": False,
            "errors":           [],
            "reasoning":        "",
        }
        iteration_history  = []
        overall_confidence = 0.0

    # ---- CHAIN_REJECTION ----
    chain_rejection = (scenario == "CHAIN_REJECTION")
    last_useful_sql = sample.get("last_useful_sql", "") if chain_rejection else ""

    return {
        "user_input": user_input,
        "ar_result": ar_result,
        "aps_result": aps_result,
        "ag_result": ag_result,
        "av_result": av_result,
        "final_sql": final_sql,
        "overall_confidence": overall_confidence,
        "iteration_history": iteration_history,
        "chain_rejection": chain_rejection,
        "last_useful_sql": last_useful_sql,
    }


def _level_to_float(level: str) -> float:
    """Mapea confidence_level (alta/media/baja) a float."""
    if not level:
        return 0.5
    return _CONFIDENCE_LEVEL_MAP.get(level.lower().strip(), 0.5)


# ─────────────────────────────────────────────────────────────────────
# Evaluacion de UNA muestra
# ─────────────────────────────────────────────────────────────────────

def _evaluate_one(agent, sample: dict,
                  f_metric, g_metric,
                  temperature: float) -> dict:
    """Llama AE.process() y compara explanation con expected_explanation.

    Soporta los 5 escenarios: SUCCESS | APS_FAILURE | AR_FAILURE |
    AG_AV_EXHAUSTED | CHAIN_REJECTION.
    """
    full_state = _build_full_state(sample)
    scenario   = _detect_scenario(sample)
    expected   = sample["expected_explanation"]

    # user_input y sql segun escenario
    user_input = sample.get("user_input") or sample.get("refined_query", "")
    refined    = sample.get("refined_query") or user_input
    sql        = sample.get("generated_sql", "")

    # Contexto de grounding segun escenario.
    # Debe reflejar exactamente lo que el AE recibe en produccion para poder
    # evaluar si su explicacion esta fundamentada en esa informacion.
    if scenario == "AR_FAILURE":
        grounding_input = (
            f"User question: {user_input}\n"
            f"AR rejection reason: {sample.get('ar_rejection_reason', '')}"
        )
    elif scenario == "APS_FAILURE":
        grounding_input = (
            f"User question: {refined}\n"
            f"APS rejection reason: {sample.get('aps_rejection_reason', '')}"
        )
    elif scenario == "AG_AV_EXHAUSTED":
        grounding_input = (
            f"User question: {refined}\n"
            f"Last AV error after {sample.get('av_iterations', 3)} retries: "
            f"{sample.get('last_av_error', '')}\n"
            f"AV reasoning: {sample.get('av_reasoning', '')}"
        )
    elif scenario == "CHAIN_REJECTION":
        grounding_input = (
            f"Follow-up question: {user_input}\n"
            f"APS rejection reason: {sample.get('aps_rejection_reason', '')}\n"
            f"Prior successful SQL: {sample.get('last_useful_sql', '')}"
        )
    else:  # SUCCESS
        grounding_input = (
            f"User question: {refined}\n"
            f"SQL: {sql}\n"
            f"AV validation: {sample.get('av_reasoning', '')}"
        )

    t0 = time.time()
    last_err = None
    r = None
    for attempt in range(2):
        try:
            r = agent.process(full_state=full_state, temperature=temperature)
            last_err = None
            break
        except Exception as e:
            last_err = e
            import time as _t; _t.sleep(3)
    elapsed = time.time() - t0

    _base = {
        "sample_idx":        sample.get("_idx", 0),
        "scenario":          scenario,
        "refined_query":     refined,
        "sql":               sql,
        "expected_explanation":  expected,
    }

    if last_err is not None:
        print(f"    [AE ERROR sample={sample.get('_idx',0)} T={temperature}] {last_err}")
        return {**_base,
            "actual_explanation": "",
            "confidence_level":   "",
            "confidence":         0.5,
            "skills_used":        [],
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome": 0,
            "time_s":  round(elapsed, 2),
            "error":   str(last_err),
        }

    try:
        explanation_obj = r.get("explanation", {}) or {}
        actual_explanation = explanation_obj.get("final_reasoning", "") or ""
        confidence_level   = explanation_obj.get("confidence_level", "")
        confidence         = _level_to_float(confidence_level)
        skills_used        = r.get("skills_used", [])
    except Exception as e:
        return {**_base,
            "actual_explanation": "",
            "confidence_level":   "",
            "confidence":         0.5,
            "skills_used":        [],
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome": 0,
            "time_s":  round(elapsed, 2),
            "error":   str(e),
        }

    if not actual_explanation:
        return {**_base,
            "actual_explanation": actual_explanation,
            "confidence_level":   confidence_level,
            "confidence":         confidence,
            "skills_used":        skills_used,
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome": 0,
            "time_s":  round(elapsed, 2),
            "error":   "AE no devolvio explicacion",
        }

    # 1. Faithfulness: actual vs expected (semantica)
    tc_f = LLMTestCase(
        input=user_input,
        actual_output=actual_explanation,
        expected_output=expected,
    )
    f_metric.measure(tc_f)
    F = float(f_metric.score or 0.0)

    # 2. Groundedness: explicacion debe apoyarse en el contexto real del escenario
    #    Para fallos: el contexto es el motivo del rechazo, no el SQL (que no existe)
    tc_g = LLMTestCase(
        input=grounding_input,
        actual_output=actual_explanation,
        expected_output=expected,
    )
    g_metric.measure(tc_g)
    G = float(g_metric.score or 0.0)

    # 3. ROUGE-L sobre texto NL (lexical, auxiliar)
    R = rouge_l_score(actual_explanation, expected)

    # 4. Outcome basado en Groundedness >= umbral
    outcome = 1 if G >= GROUNDEDNESS_THRESHOLD_FOR_OUTCOME else 0

    return {**_base,
        "actual_explanation":    actual_explanation,
        "confidence_level":      confidence_level,
        "confidence":            confidence,
        "skills_used":           skills_used,
        "F": round(F, 3),
        "G": round(G, 3),
        "R": round(R, 3),
        "outcome": outcome,
        "time_s":  round(elapsed, 2),
        "error":   None,
    }


# ─────────────────────────────────────────────────────────────────────
# Evaluacion completa (1 trial = 1 combinacion model + temperature)
# ─────────────────────────────────────────────────────────────────────

def evaluate_ae(model: str = "gpt-4o-mini",
                temperature: float = 0.3,
                samples: list = None,
                verbose: bool = False) -> dict:
    """
    Evalua AE con (model, temperature). Devuelve dict con metricas agregadas
    + per_sample.
    """
    from agents.AE.explainer_agent import ExplainerAgent

    if samples is None:
        samples = load_samples()

    agent = ExplainerAgent()
    agent.model  = model
    agent.client = get_client_for_model(model)

    # Metricas LLM-juez una sola vez (reuso entre samples)
    f_metric = faithfulness_geval()
    g_metric = groundedness_geval()

    F_list, G_list, R_list = [], [], []
    confidences, outcomes  = [], []
    per_sample = []

    for i, sample in enumerate(samples):
        sample["_idx"] = i + 1
        result = _evaluate_one(agent, sample, f_metric, g_metric, temperature)
        per_sample.append(result)

        F_list.append(result["F"])
        G_list.append(result["G"])
        R_list.append(result["R"])
        confidences.append(result["confidence"])
        outcomes.append(result["outcome"])

        if verbose:
            sc = result.get("scenario", "SUCCESS")[:3]
            print(f"  [{result['sample_idx']:>2}][{sc}] F={result['F']:.3f}  "
                  f"G={result['G']:.3f}  R={result['R']:.3f}  "
                  f"conf={result['confidence']:.2f}  "
                  f"outcome={result['outcome']}  "
                  f"t={result['time_s']:.2f}s")

    n = len(per_sample)
    avg = lambda xs: round(sum(xs) / n, 3) if n else 0.0

    F_avg = avg(F_list)
    G_avg = avg(G_list)
    R_avg = avg(R_list)

    B = round(brier_score(confidences, outcomes), 4)
    E = round(ece(confidences, outcomes, n_bins=5), 3)

    combined = round((2 * F_avg + R_avg + 2 * G_avg - E - B) / 5, 3)

    avg_time = round(
        sum(r["time_s"] for r in per_sample) / n, 2
    ) if n else 0.0

    return {
        "model":          model,
        "temperature":    temperature,
        "n_samples":      n,
        "Faithfulness":   F_avg,
        "Groundedness":   G_avg,
        "ROUGE-L":        R_avg,
        "Brier":          B,
        "ECE":            E,
        "combined_score": combined,
        "avg_time_s":     avg_time,
        "per_sample":     per_sample,
    }


def save_results(metrics: dict, tag: str = "single") -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{tag}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
