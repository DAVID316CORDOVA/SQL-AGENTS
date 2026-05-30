# -*- coding: utf-8 -*-
"""
agents/AV/mysql/phase_1/main.py

Evaluacion del AV (Agente Validador SQL MySQL) — Fase 1: busqueda de
hiperparametros con Optuna.

Nueva metodologia (asesor 2026-05-13). En lugar de tratar al AV como un
clasificador binario, se mide la calidad de su EXPLICACION (campo
`reasoning`) y la coincidencia textual de su DECISION (`decision_word`,
"correct"/"incorrect"):

  - Faithfulness : GEval (juez gpt-4o) sobre (reasoning, expected_reasoning)
  - Groundedness : GEval (juez gpt-4o) sobre reasoning con contexto
                   (refined_query + sql + schema)
  - ROUGE-L      : rouge_l_score(decision_word, expected_decision_word)
  - Brier        : sobre (confidence, outcome=decision acerto)
  - ECE          : idem

Funcion objetivo:
    combined = (2*F + R + 2*G - E - B) / 5     (max teorico = 1.0)

NOTA: el AV evalua SQL de forma estatica (sin ejecutar contra MySQL).
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
    f1, accuracy, precision, recall, confusion,
    av_faithfulness_geval, av_groundedness_geval, rouge_l_score,
    brier_score, ece,
)
from llm_client import get_client_for_model

# ─────────────────────────────────────────────────────────────────────
# Configuracion
# ─────────────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")

# El outcome de calibracion es "el AV acerto la clase binaria". Es decir,
# si el decision_word predicho coincide con el esperado.


# ─────────────────────────────────────────────────────────────────────
# Carga del dataset
# ─────────────────────────────────────────────────────────────────────

def load_samples(dataset_path: str = DATASET_PATH) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]


def _build_inputs_for_av(sample: dict) -> tuple[dict, dict, dict]:
    """
    Construye los 3 dicts que AV.process(ag_result, ar_result, aps_result)
    espera, desde el formato simplificado del dataset.
    """
    ag_result = {
        "success": True,
        "sql": sample["sql"],
        "confidence_score": 1.0,
    }
    ar_result = {
        "success": True,
        "refined_query": sample["refined_query"],
    }

    tables_dict = {}
    for col in sample.get("selected_columns", []):
        tname = col["table_name"]
        cname = col["column_name"]
        tables_dict.setdefault(tname, {"all_columns": []})
        if cname not in tables_dict[tname]["all_columns"]:
            tables_dict[tname]["all_columns"].append(cname)
    for t in sample.get("selected_tables", []):
        tables_dict.setdefault(t["table_name"], {"all_columns": []})

    aps_result = {
        "success": True,
        "input_query": sample["refined_query"],
        "tables": tables_dict,
        "joins": [],
        "original_intent": {},
    }

    return ag_result, ar_result, aps_result


def _schema_to_text(aps_result: dict) -> str:
    """Serializa el schema (tablas + columnas) como texto para Groundedness."""
    lines = []
    for tname, tinfo in aps_result.get("tables", {}).items():
        cols = ", ".join(tinfo.get("all_columns", []))
        lines.append(f"  {tname}({cols})")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Evaluacion de UNA muestra
# ─────────────────────────────────────────────────────────────────────

def _evaluate_one(agent, sample_idx: int, sample: dict,
                  f_metric, g_metric,
                  temperature: float) -> dict:
    """Llama AV.process() y evalua reasoning con F/G y decision_word con ROUGE-L."""
    ag_result, ar_result, aps_result = _build_inputs_for_av(sample)
    expected_decision     = sample["expected_decision_word"]
    expected_is_valid     = (expected_decision.strip().lower() == "correct")
    expected_reasoning    = sample["expected_reasoning"]

    t0 = time.time()
    try:
        r = agent.process(
            ag_result=ag_result,
            ar_result=ar_result,
            aps_result=aps_result,
            temperature=temperature,
        )
        elapsed = time.time() - t0
        predicted_is_valid = bool(r.get("is_valid", False))
        actual_reasoning   = r.get("reasoning", "") or "(no reasoning)"
        actual_decision    = r.get("decision_word", "incorrect")
        confidence         = float(r.get("confidence_score", 0.5))
    except Exception as e:
        return {
            "sql":               sample["sql"],
            "expected_is_valid": expected_is_valid,
            "predicted_is_valid": False,
            "expected_decision": expected_decision,
            "actual_decision":   "",
            "actual_reasoning":  "",
            "confidence":        0.5,
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome":           0,
            "time_s":            0.0,
            "error":             str(e),
        }

    # Faithfulness: reasoning del AV vs expected_reasoning gold
    tc_f = LLMTestCase(
        input=sample["refined_query"],
        actual_output=actual_reasoning,
        expected_output=expected_reasoning,
    )
    f_metric.measure(tc_f)
    F = float(f_metric.score or 0.0)

    # Groundedness: reasoning debe apoyarse en (refined_query + sql + schema)
    schema_text = _schema_to_text(aps_result)
    grounding_input = (
        f"User question: {sample['refined_query']}\n"
        f"SQL: {sample['sql']}\n"
        f"Schema:\n{schema_text}"
    )
    tc_g = LLMTestCase(
        input=grounding_input,
        actual_output=actual_reasoning,
        expected_output=expected_reasoning,
    )
    g_metric.measure(tc_g)
    G = float(g_metric.score or 0.0)

    # ROUGE-L lexical sobre decision_word: 1.0 si coincide, 0.0 si no.
    R = rouge_l_score(actual_decision, expected_decision)

    # Outcome para Brier/ECE: el AV acerto la clase binaria.
    correct = (actual_decision.strip().lower() == expected_decision.strip().lower())
    outcome = 1 if correct else 0

    return {
        "sql":               sample["sql"],
        "expected_is_valid": expected_is_valid,
        "predicted_is_valid": predicted_is_valid,
        "expected_decision": expected_decision,
        "actual_decision":   actual_decision,
        "actual_reasoning":  actual_reasoning,
        "confidence":        confidence,
        "F": round(F, 3),
        "G": round(G, 3),
        "R": round(R, 3),
        "outcome":           outcome,
        "time_s":            round(elapsed, 2),
        "error":             None,
    }


# ─────────────────────────────────────────────────────────────────────
# Evaluacion completa (1 trial = 1 combinacion model+temperature)
# ─────────────────────────────────────────────────────────────────────

def evaluate_av(model: str = "gpt-4o-mini",
                temperature: float = 0.1,
                samples: list = None,
                verbose: bool = False) -> dict:
    """
    Evalua AV con (model, temperature). Devuelve dict con metricas
    semanticas (F/G/R) + clasificacion (F1/acc) + calibracion (Brier/ECE).

    Evalua AV sin ejecutar contra MySQL (analisis estatico).
    """
    from agents.AV.mysql.sql_validator_agent import SQLValidatorAgent

    if samples is None:
        samples = load_samples()

    agent = SQLValidatorAgent()
    agent.model  = model
    agent.client = get_client_for_model(model)

    f_metric = av_faithfulness_geval()
    g_metric = av_groundedness_geval()

    F_list, G_list, R_list = [], [], []
    true_labels, predicted_labels = [], []
    confidences, outcomes = [], []
    per_sample = []

    for i, sample in enumerate(samples):
        result = _evaluate_one(agent, i + 1, sample,
                               f_metric, g_metric, temperature)
        per_sample.append(result)

        F_list.append(result["F"])
        G_list.append(result["G"])
        R_list.append(result["R"])
        true_labels.append(result["expected_is_valid"])
        predicted_labels.append(result["predicted_is_valid"])
        confidences.append(result["confidence"])
        outcomes.append(result["outcome"])

        if verbose:
            tag = "OK" if result["outcome"] == 1 else "X "
            print(f"  [{i+1:>2}] {tag} "
                  f"F={result['F']:.3f}  G={result['G']:.3f}  "
                  f"R={result['R']:.3f}  conf={result['confidence']:.2f}")

    n = len(per_sample)
    avg = lambda xs: round(sum(xs) / n, 3) if n else 0.0

    F_avg = avg(F_list)
    G_avg = avg(G_list)
    R_avg = avg(R_list)

    # Metricas auxiliares de clasificacion (clase positiva = invalido)
    acc      = round(accuracy(true_labels, predicted_labels), 3)
    prec     = round(precision(true_labels, predicted_labels, positive_class=False), 3)
    rec      = round(recall(true_labels, predicted_labels, positive_class=False), 3)
    f1_score = round(f1(true_labels, predicted_labels, positive_class=False), 3)
    cm       = confusion(true_labels, predicted_labels)

    # Calibracion
    B = round(brier_score(confidences, outcomes), 4)
    E = round(ece(confidences, outcomes, n_bins=5), 3)

    # Funcion objetivo: nueva formula del asesor
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
        "F1":             f1_score,
        "accuracy":       acc,
        "precision":      prec,
        "recall":         rec,
        "Brier":          B,
        "ECE":            E,
        "combined_score": combined,
        "confusion":      cm,
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
