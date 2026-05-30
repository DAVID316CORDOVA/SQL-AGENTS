# -*- coding: utf-8 -*-
"""
agents/AE/phase_2/main.py - Evaluacion del AE (Fase 2, paraphrase invariance)

Mismas 5 metricas que Fase 1 (asesor 2026-05-13):

    combined_score = (2*Faithfulness + ROUGE-L + 2*Groundedness - ECE - Brier) / 5
    (max teorico = 1.0)

Diferencia con Fase 1:
  - Fase 1: 11 preguntas distintas, cada una con su expected_explanation.
  - Fase 2: 11 intents x 3 paraphrases = 33 inputs. Las 3 paraphrases de
            cada intent comparten:
              - el MISMO mock pipeline state (tablas, columnas, SQL)
              - el MISMO expected_explanation gold
            Solo cambia la formulacion del input (user_input + refined_query).
            Asi medimos: el AE produce explicaciones equivalentes al gold
            sin importar como se formulo la pregunta?

Hiperparametros: FIJOS (ganador de Fase 1: gpt-4o T=0.3). NO Optuna.
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


PARAPHRASES_PATH = os.path.join(os.path.dirname(__file__), "paraphrases.json")
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "results")

# Outcome para Brier/ECE: G (Groundedness) >= 0.80 — consistente con Fase 1
GROUNDEDNESS_THRESHOLD_FOR_OUTCOME = 0.80

_CONFIDENCE_LEVEL_MAP = {
    "alta":   0.90,
    "high":   0.90,
    "media":  0.60,
    "medium": 0.60,
    "baja":   0.30,
    "low":    0.30,
}


# ---------------------------------------------------------------------
# Carga de dataset
# ---------------------------------------------------------------------

def load_intents(path: str = PARAPHRASES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


def _build_full_state(intent: dict, paraphrase_text: str) -> dict:
    """
    Construye el full_state del pipeline para una paraphrase concreta.

    user_input y refined_query toman el texto de la paraphrase (simulando
    que AR refino la entrada del usuario). El resto (tables/columns/sql)
    es identico al gold del intent: medimos solo el efecto de la
    formulacion sobre la explicacion del AE.
    """
    tables_dict = {}
    for col in intent.get("selected_columns", []):
        tname = col["table_name"]
        cname = col["column_name"]
        tables_dict.setdefault(tname, {"all_columns": []})
        if cname not in tables_dict[tname]["all_columns"]:
            tables_dict[tname]["all_columns"].append(cname)
    for t in intent.get("selected_tables", []):
        tables_dict.setdefault(t["table_name"], {"all_columns": []})

    return {
        "user_input": paraphrase_text,
        "ar_result": {
            "success":        True,
            "is_valid_query": True,
            "refined_query":  paraphrase_text,
            "confidence_score": 0.95,
            "reasoning":      "Valid database query identified.",
        },
        "aps_result": {
            "success": True,
            "tables":  tables_dict,
            "confidence_score": 0.85,
        },
        "ag_result": {
            "success":          True,
            "sql":              intent["generated_sql"],
            "confidence_score": 1.0,
            "strategy":         intent.get("ag_strategy", ""),
        },
        "av_result": {
            "success":          True,
            "is_valid":         True,
            "unfixable":        False,
            "needs_correction": False,
            "errors":           [],
            "reasoning":        intent.get("av_reasoning", "The SQL passed all validation checks."),
        },
        "final_sql":          intent["generated_sql"],
        "overall_confidence": 1.0,
        "iteration_history":  [],
    }


def _level_to_float(level: str) -> float:
    if not level:
        return 0.5
    return _CONFIDENCE_LEVEL_MAP.get(level.lower().strip(), 0.5)


# ---------------------------------------------------------------------
# Evaluacion de UNA paraphrase
# ---------------------------------------------------------------------

def _evaluate_paraphrase(agent, intent_idx: int, paraphrase_idx: int,
                         intent: dict, paraphrase: str,
                         f_metric, g_metric,
                         temperature: float) -> dict:
    expected = intent["expected_explanation"]
    refined  = paraphrase
    sql      = intent["generated_sql"]

    full_state = _build_full_state(intent, refined)

    t0 = time.time()
    try:
        r = agent.process(full_state=full_state, temperature=temperature)
        elapsed = time.time() - t0
        explanation_obj    = r.get("explanation", {}) or {}
        actual_explanation = explanation_obj.get("final_reasoning", "") or ""
        confidence_level   = explanation_obj.get("confidence_level", "")
        confidence         = _level_to_float(confidence_level)
        success            = bool(r.get("success", False))
    except Exception as e:
        return {
            "intent_idx":          intent_idx,
            "paraphrase_idx":      paraphrase_idx,
            "input":               refined,
            "expected_explanation": expected,
            "actual_explanation":  "",
            "confidence_level":    "",
            "confidence":          0.5,
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome": 0,
            "time_s": 0.0,
            "error": str(e),
        }

    if not success or not actual_explanation:
        return {
            "intent_idx":          intent_idx,
            "paraphrase_idx":      paraphrase_idx,
            "input":               refined,
            "expected_explanation": expected,
            "actual_explanation":  actual_explanation,
            "confidence_level":    confidence_level,
            "confidence":          confidence,
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome": 0,
            "time_s": round(elapsed, 2),
            "error": "AE no devolvio explicacion",
        }

    # Faithfulness: actual vs expected (semantica)
    tc_f = LLMTestCase(
        input=refined,
        actual_output=actual_explanation,
        expected_output=expected,
    )
    f_metric.measure(tc_f)
    F = float(f_metric.score or 0.0)

    # Groundedness: actual debe apoyarse en (refined + sql)
    grounding_input = f"User question: {refined}\nSQL: {sql}"
    tc_g = LLMTestCase(
        input=grounding_input,
        actual_output=actual_explanation,
        expected_output=expected,
    )
    g_metric.measure(tc_g)
    G = float(g_metric.score or 0.0)

    # ROUGE-L lexical
    R = rouge_l_score(actual_explanation, expected)

    # Outcome basado en F (semantico, consistente con AR/AE Fase 1)
    outcome = 1 if G >= GROUNDEDNESS_THRESHOLD_FOR_OUTCOME else 0

    return {
        "intent_idx":          intent_idx,
        "paraphrase_idx":      paraphrase_idx,
        "input":               refined,
        "expected_explanation": expected,
        "actual_explanation":  actual_explanation,
        "confidence_level":    confidence_level,
        "confidence":          confidence,
        "F": round(F, 3),
        "G": round(G, 3),
        "R": round(R, 3),
        "outcome": outcome,
        "time_s": round(elapsed, 2),
        "error": None,
    }


# ---------------------------------------------------------------------
# Evaluacion completa
# ---------------------------------------------------------------------

def evaluate_ae_paraphrases(model: str = "gpt-4o",
                            temperature: float = 0.3,
                            intents: list = None,
                            verbose: bool = False) -> dict:
    """
    Evalua AE con (model, temperature) fijos sobre las paraphrases.

    Mismas 5 metricas y misma combined_score formula que Fase 1:
        combined = (2*F + R + 2*G - E - B) / 5  (max 1.0)
    """
    from agents.AE.explainer_agent import ExplainerAgent

    if intents is None:
        intents = load_intents()

    agent = ExplainerAgent()
    agent.model  = model
    agent.client = get_client_for_model(model)

    f_metric = faithfulness_geval()
    g_metric = groundedness_geval()

    F_list, G_list, R_list = [], [], []
    confidences, outcomes  = [], []
    per_paraphrase = []
    per_intent_stats = []

    for i, intent in enumerate(intents):
        if verbose:
            print(f"\n  Intent #{i+1} - {intent['refined_query'][:60]}")

        intent_F, intent_G, intent_R = [], [], []
        for j, paraphrase in enumerate(intent["paraphrases"]):
            result = _evaluate_paraphrase(agent, i + 1, j + 1,
                                          intent, paraphrase,
                                          f_metric, g_metric, temperature)
            per_paraphrase.append(result)

            F_list.append(result["F"])
            G_list.append(result["G"])
            R_list.append(result["R"])
            confidences.append(result["confidence"])
            outcomes.append(result["outcome"])

            intent_F.append(result["F"])
            intent_G.append(result["G"])
            intent_R.append(result["R"])

            if verbose:
                tag = f"{result['intent_idx']}.P{result['paraphrase_idx']}"
                print(f"    [{tag:>5}] "
                      f"F={result['F']:.3f}  "
                      f"G={result['G']:.3f}  "
                      f"R={result['R']:.3f}  "
                      f"conf={result['confidence']:.2f}  "
                      f"outcome={result['outcome']}  "
                      f"t={result['time_s']:.2f}s")

        # Estadisticas por intent (media solamente; reporte informativo)
        m = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0
        per_intent_stats.append({
            "intent_idx": i + 1,
            "F_mean": m(intent_F),
            "G_mean": m(intent_G),
            "R_mean": m(intent_R),
        })

    n = len(per_paraphrase)
    avg = lambda xs: round(sum(xs) / n, 3) if n else 0.0

    F_avg = avg(F_list)
    G_avg = avg(G_list)
    R_avg = avg(R_list)

    B = round(brier_score(confidences, outcomes), 4)
    E = round(ece(confidences, outcomes, n_bins=5), 3)

    combined = round((2 * F_avg + R_avg + 2 * G_avg - E - B) / 5, 3)

    avg_time = round(
        sum(r["time_s"] for r in per_paraphrase) / n, 2
    ) if n else 0.0

    return {
        "model":          model,
        "temperature":    temperature,
        "n_paraphrases":  n,
        "n_intents":      len(intents),
        "Faithfulness":   F_avg,
        "Groundedness":   G_avg,
        "ROUGE-L":        R_avg,
        "Brier":          B,
        "ECE":            E,
        "combined_score": combined,
        "avg_time_s":     avg_time,
        "per_intent":     per_intent_stats,
        "per_paraphrase": per_paraphrase,
    }


def save_results(metrics: dict, tag: str = "phase2") -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{tag}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
