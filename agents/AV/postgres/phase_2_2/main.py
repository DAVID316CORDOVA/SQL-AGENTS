# -*- coding: utf-8 -*-
"""
agents/AV/mysql/phase_2/main.py

Fase 2 del AV MySQL: robustez de la explicacion ante 3 formulaciones
distintas del expected_reasoning (el SQL, query y schema son FIJOS).

Para cada muestra:
  1. El AV genera UNA sola respuesta (reasoning + decision_word)
  2. Se evalua esa respuesta contra 3 expected_reasoning distintos
  3. Se promedian las metricas F de las 3 comparaciones

Formula: combined = (2F + R + 2G - B - E) / 5   (max 1.0)
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
    av_faithfulness_geval, av_groundedness_geval, rouge_l_score,
    brier_score, ece, accuracy, f1, precision, recall, confusion,
)
from llm_client import get_client_for_model

PARAPHRASES_PATH = os.path.join(os.path.dirname(__file__), "paraphrases.json")
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "results")


def load_samples(path: str = PARAPHRASES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["samples"]


def _build_inputs_for_av(sample: dict) -> tuple[dict, dict, dict]:
    tables_dict = {}
    for col in sample.get("selected_columns", []):
        tname = col["table_name"]
        cname = col["column_name"]
        tables_dict.setdefault(tname, {"all_columns": []})
        if cname not in tables_dict[tname]["all_columns"]:
            tables_dict[tname]["all_columns"].append(cname)
    for t in sample.get("selected_tables", []):
        tables_dict.setdefault(t["table_name"], {"all_columns": []})

    return (
        {"success": True, "sql": sample["sql"], "confidence_score": 1.0},
        {"success": True, "refined_query": sample["refined_query"]},
        {"success": True, "input_query": sample["refined_query"],
         "tables": tables_dict, "joins": [], "original_intent": {}},
    )


def _schema_to_text(aps_result: dict) -> str:
    lines = []
    for tname, tinfo in aps_result.get("tables", {}).items():
        cols = ", ".join(tinfo.get("all_columns", []))
        lines.append(f"  {tname}({cols})")
    return "\n".join(lines)


def evaluate_av_paraphrases(model: str = "gpt-4o",
                             temperature: float = 0.1,
                             samples: list = None,
                             verbose: bool = False) -> dict:
    from agents.AV.postgres.sql_validator_agent import SQLValidatorAgent

    if samples is None:
        samples = load_samples()

    agent = SQLValidatorAgent()
    agent.model  = model
    agent.client = get_client_for_model(model)

    f_metric = av_faithfulness_geval()
    g_metric = av_groundedness_geval()

    F_list, G_list, R_list = [], [], []
    true_labels, pred_labels = [], []
    confidences, outcomes = [], []
    per_sample = []

    for s_idx, sample in enumerate(samples):
        ag_result, ar_result, aps_result = _build_inputs_for_av(sample)
        expected_decision   = sample["expected_decision_word"]
        expected_is_valid   = (expected_decision.strip().lower() == "correct")
        expected_reasonings = sample["expected_reasonings"]

        t0 = time.time()
        try:
            r = agent.process(ag_result=ag_result, ar_result=ar_result,
                              aps_result=aps_result, temperature=temperature)
            elapsed = time.time() - t0
            actual_reasoning   = r.get("reasoning", "") or ""
            actual_decision    = r.get("decision_word", "incorrect")
            predicted_is_valid = bool(r.get("is_valid", False))
            confidence         = float(r.get("confidence_score", 0.5))
        except Exception as e:
            per_sample.append({"sql": sample["sql"], "error": str(e),
                               "F": 0.0, "G": 0.0, "R": 0.0,
                               "confidence": 0.5, "outcome": 0, "time_s": 0.0})
            F_list.append(0.0); G_list.append(0.0); R_list.append(0.0)
            confidences.append(0.5); outcomes.append(0)
            true_labels.append(expected_is_valid); pred_labels.append(False)
            continue

        # ROUGE-L decision_word
        R = rouge_l_score(actual_decision, expected_decision)
        outcome = 1 if actual_decision.strip().lower() == expected_decision.strip().lower() else 0

        # Groundedness (fija — el input no cambia entre las 3 paráfrasis)
        schema_text = _schema_to_text(aps_result)
        grounding_input = (
            f"User question: {sample['refined_query']}\n"
            f"SQL: {sample['sql']}\n"
            f"Schema:\n{schema_text}"
        )
        tc_g = LLMTestCase(input=grounding_input, actual_output=actual_reasoning,
                           expected_output=expected_reasonings[0])
        g_metric.measure(tc_g)
        G = float(g_metric.score or 0.0)

        # Faithfulness: promedio contra las 3 formulaciones del expected_reasoning
        F_vals = []
        for exp_r in expected_reasonings:
            tc_f = LLMTestCase(input=sample["refined_query"],
                               actual_output=actual_reasoning, expected_output=exp_r)
            f_metric.measure(tc_f)
            F_vals.append(float(f_metric.score or 0.0))
        F = round(sum(F_vals) / len(F_vals), 3)

        F_list.append(F); G_list.append(G); R_list.append(R)
        confidences.append(confidence); outcomes.append(outcome)
        true_labels.append(expected_is_valid); pred_labels.append(predicted_is_valid)

        entry = {
            "sql": sample["sql"],
            "actual_decision": actual_decision,
            "expected_decision": expected_decision,
            "actual_reasoning": actual_reasoning,
            "F_per_paraphrase": F_vals,
            "F": F, "G": round(G, 3), "R": round(R, 3),
            "confidence": confidence, "outcome": outcome,
            "time_s": round(time.time() - t0, 2),
        }
        per_sample.append(entry)

        if verbose:
            tag = "OK" if outcome == 1 else "X "
            print(f"  [{s_idx+1:>2}] {tag} F={F:.3f} G={G:.3f} R={R:.3f} "
                  f"conf={confidence:.2f} | {actual_decision}")

    n = len(per_sample)
    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0

    F_avg = avg(F_list); G_avg = avg(G_list); R_avg = avg(R_list)

    acc      = round(accuracy(true_labels, pred_labels), 3)
    f1_score = round(f1(true_labels, pred_labels, positive_class=False), 3)
    prec     = round(precision(true_labels, pred_labels, positive_class=False), 3)
    rec      = round(recall(true_labels, pred_labels, positive_class=False), 3)
    cm       = confusion(true_labels, pred_labels)

    B = round(brier_score(confidences, outcomes), 4)
    E = round(ece(confidences, outcomes, n_bins=5), 3)
    combined = round((2 * F_avg + R_avg + 2 * G_avg - B - E) / 5, 3)

    return {
        "model": model, "temperature": temperature,
        "n_samples": n, "n_paraphrases_per_sample": 3,
        "Faithfulness": F_avg, "Groundedness": G_avg, "ROUGE-L": R_avg,
        "F1": f1_score, "accuracy": acc, "precision": prec, "recall": rec,
        "Brier": B, "ECE": E, "combined_score": combined,
        "confusion": cm,
        "avg_time_s": round(sum(r.get("time_s", 0) for r in per_sample) / n, 2) if n else 0,
        "per_sample": per_sample,
    }


def save_results(metrics: dict, tag: str = "phase2") -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{tag}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
