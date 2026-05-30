# -*- coding: utf-8 -*-
"""
agents/AG/mysql/phase_2/main.py

Evaluacion del AG (Generador SQL MySQL) â€” Fase 3: paraphrase invariance.

Para cada intent, las tablas y columnas son CONSTANTES (las que el APS
extraeria), pero el refined_query VARIA â€” son 3 paraphrases del mismo
intent. Mide si el AG genera el mismo SQL pese a la variacion linguistica.

Hiperparametros: FIJOS (los ganadores de Fase 1: gpt-4o, T=0.3).
NO se usa Optuna porque no hay busqueda â€” solo medicion de robustez.

Metricas (mismas que AG Fase 1):
  - Faithfulness  : GEval (juez gpt-4o) (generated_sql, expected_sql)
  - Groundedness  : GEval (paraphrase_text, generated_sql) â€” no inventa
  - ROUGE-L_SQL   : rouge_l_sql() determinista
  - Brier         : sklearn sobre (confidence, outcome=R>=0.80)
  - ECE           : NumPy idem

Funcion objetivo de reporte:
    combined = F + G + ROUGE-L_SQL - Brier - ECE   (max 3.0)
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

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Configuracion
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

PARAPHRASES_PATH = os.path.join(os.path.dirname(__file__), "paraphrases.json")
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "results")

ROUGE_THRESHOLD_FOR_OUTCOME = 0.70


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Carga del dataset
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_intents(path: str = PARAPHRASES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


def _build_aps_result(intent: dict, paraphrase_text: str) -> dict:
    """
    Construye el aps_result que el AG espera.

    Las tablas/columnas vienen del intent (FIJAS), el refined_query es la
    paraphrase actual (VARIABLE).
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
        "success":     True,
        "input_query": paraphrase_text,
        "tables":      tables_dict,
        "joins":       [],
        "original_intent": {},
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Evaluacion de UNA paraphrase
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _evaluate_paraphrase(agent, intent: dict, paraphrase: dict,
                         f_metric, g_metric,
                         temperature: float) -> dict:
    """Evalua el AG sobre una paraphrase del intent."""
    aps_input    = _build_aps_result(intent, paraphrase["text"])
    expected_sql = intent["expected_sql"]

    t0 = time.time()
    try:
        r = agent.process(aps_result=aps_input, temperature=temperature)
        elapsed = time.time() - t0
        generated_sql = r.get("sql", "") or ""
        confidence    = float(r.get("confidence_score", 0.5))
        success       = bool(r.get("success", False))
    except Exception as e:
        return {
            "input_text":    paraphrase["text"],
            "expected_sql":  expected_sql,
            "generated_sql": "",
            "confidence":    0.5,
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome": 0,
            "time_s": 0.0,
            "error": str(e),
        }

    if not success or not generated_sql:
        return {
            "input_text":    paraphrase["text"],
            "expected_sql":  expected_sql,
            "generated_sql": generated_sql,
            "confidence":    confidence,
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome": 0,
            "time_s": round(elapsed, 2),
            "error": r.get("error", "AG returned success=False"),
        }

    # Faithfulness: Â¿el SQL generado es fiel al expected_sql?
    tc_f = LLMTestCase(
        input=paraphrase["text"],
        actual_output=generated_sql,
        expected_output=expected_sql,
    )
    f_metric.measure(tc_f)
    F = float(f_metric.score or 0.0)

    # Groundedness: Â¿el SQL se apoya en la paraphrase original?
    g_metric.measure(tc_f)
    G = float(g_metric.score or 0.0)

    # ROUGE-L sobre SQL normalizado
    R = rouge_l_sql(generated_sql, expected_sql)
    outcome = 1 if R >= ROUGE_THRESHOLD_FOR_OUTCOME else 0

    return {
        "input_text":    paraphrase["text"],
        "expected_sql":  expected_sql,
        "generated_sql": generated_sql,
        "confidence":    confidence,
        "F": round(F, 3),
        "G": round(G, 3),
        "R": round(R, 3),
        "outcome": outcome,
        "time_s": round(elapsed, 2),
        "error": None,
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Evaluacion completa (config fija, sin Optuna)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def evaluate_ag_paraphrases(model: str = "gpt-4o",
                            temperature: float = 0.3,
                            intents: list = None,
                            verbose: bool = False) -> dict:
    """
    Evalua el AG con (model, temperature) fijos sobre 11 intents x 3
    paraphrases = 33 inputs.

    Returns dict con metricas agregadas + per_paraphrase.
    """
    from agents.AG.mysql.sql_generator_agent import SQLGeneratorAgent

    if intents is None:
        intents = load_intents()

    agent = SQLGeneratorAgent()
    agent.model  = model
    agent.client = get_client_for_model(model)

    f_metric = faithfulness_geval()
    g_metric = groundedness_geval()

    F_list, G_list, R_list = [], [], []
    confidences, outcomes  = [], []
    per_paraphrase = []

    for intent_idx, intent in enumerate(intents):
        if verbose:
            print(f"\n  Intent #{intent_idx + 1}")
        for p_idx, paraphrase in enumerate(intent["paraphrases"]):
            result = _evaluate_paraphrase(agent, intent, paraphrase,
                                          f_metric, g_metric, temperature)
            per_paraphrase.append(result)

            F_list.append(result["F"])
            G_list.append(result["G"])
            R_list.append(result["R"])
            confidences.append(result["confidence"])
            outcomes.append(result["outcome"])

            if verbose:
                print(f"    [P{p_idx+1}] "
                      f"F={result['F']:.3f}  "
                      f"G={result['G']:.3f}  "
                      f"R={result['R']:.3f}  "
                      f"conf={result['confidence']:.2f}  "
                      f"outcome={result['outcome']}")

    n = len(per_paraphrase)
    avg = lambda xs: round(sum(xs) / n, 3) if n else 0.0

    F_avg = avg(F_list)
    G_avg = avg(G_list)
    R_avg = avg(R_list)

    B = round(brier_score(confidences, outcomes), 4)
    E = round(ece(confidences, outcomes, n_bins=5), 3)

    combined = round(F_avg + G_avg + R_avg - B - E, 3)

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
        "ROUGE-L_SQL":    R_avg,
        "Brier":          B,
        "ECE":            E,
        "combined_score": combined,
        "avg_time_s":     avg_time,
        "per_paraphrase": per_paraphrase,
    }


def save_results(metrics: dict, tag: str = "phase3") -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{tag}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
