# -*- coding: utf-8 -*-
"""
agents/AS/phase_1/main.py

Evaluation of the AS (Sustainer Agent) - Phase 1.

AS receives (last_result, user_question) and produces a natural-language
justification explaining WHY the pipeline made its decisions.

The evaluation uses the real agentic loop (agent.process()), so the
get_agent_reasoning skill IS invoked during evaluation — consistent with
what happens in production.

Metrics:
  F = Faithfulness  GEval (justification vs expected_justification)
  G = Groundedness  GEval (justification vs pipeline context)
  R = ROUGE-L       deterministic
  B = Brier Score   sklearn (confidence vs outcome = G >= 0.80)
  E = ECE           NumPy

Formula: (2F + R + 2G - B - E) / 5   (max 1.0)
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
    as_faithfulness_geval,
    as_groundedness_geval,
    rouge_l_score,
    brier_score,
    ece,
)
from llm_client import get_client_for_model

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")

# Outcome for Brier/ECE: G >= 0.80 — consistent with AE Phase 1
GROUNDEDNESS_THRESHOLD_FOR_OUTCOME = 0.80


def load_samples(dataset_path: str = DATASET_PATH) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)["samples"]


def _build_last_result(sample: dict) -> dict:
    """
    Builds the pipeline last_result mock from a dataset sample.

    Handles:
      - ag_strategy  : passed to ag_result["strategy"]
      - av_reasoning : passed to av_result["reasoning"]
      - av_iterations: number of AV correction cycles (0 = clean success)
      - av_first_error: error message from the first rejected attempt
      - iteration_count = av_iterations + 1  (at least 1 attempt always)
    """
    refined      = sample["refined_query"]
    sql          = sample.get("generated_sql", "")
    ae_text      = sample.get("ae_explanation", "")
    tables       = sample.get("selected_tables", [])
    columns      = sample.get("selected_columns", [])
    ag_strategy  = sample.get("ag_strategy", "simple")
    av_reasoning = sample.get("av_reasoning",
                               "Query validated on first attempt. No errors detected.")
    av_iterations  = sample.get("av_iterations", 0)
    av_first_error = sample.get("av_first_error", "")

    tables_dict = {}
    for col in columns:
        tname = col["table_name"]
        cname = col["column_name"]
        tables_dict.setdefault(tname, {"all_columns": [], "matched_columns": []})
        if cname not in tables_dict[tname]["matched_columns"]:
            tables_dict[tname]["matched_columns"].append(cname)
        if cname not in tables_dict[tname]["all_columns"]:
            tables_dict[tname]["all_columns"].append(cname)
    for t in tables:
        tables_dict.setdefault(t["table_name"], {"all_columns": [], "matched_columns": []})

    table_names     = list(tables_dict.keys())
    iteration_count = av_iterations + 1  # minimum 1 attempt

    # Build iteration history for AV correction scenarios
    iteration_history = []
    if av_iterations > 0 and av_first_error:
        for k in range(av_iterations):
            iteration_history.append({
                "iteration": k + 1,
                "av_errors": [av_first_error],
                "av_valid":  False,
            })

    return {
        "user_input": sample.get("user_input", refined),
        "ar_result": {
            "success":          True,
            "agent":            "AR-Refiner",
            "is_valid_query":   True,
            "refined_query":    refined,
            "intent":           {},
            "reasoning":        (
                f"Refined the user query into '{refined}'. "
                f"Identified entities, filters and aggregations."
            ),
            "confidence_score": 0.9,
        },
        "aps_result": {
            "success":          True,
            "agent":            "APS-Matcher",
            "tables":           tables_dict,
            "reasoning":        (
                f"Selected tables {table_names} based on semantic similarity. "
                f"Columns matched per table."
            ),
            "confidence_score": 0.87,
        },
        "ag_result": {
            "success":          True,
            "agent":            "AG-Generator",
            "sql":              sql,
            "strategy":         ag_strategy,
            "reasoning":        (
                f"Generated SQL using '{ag_strategy}' strategy with matched tables "
                f"and columns. Final query: {sql[:160]}"
            ),
            "confidence_score": 0.95,
        },
        "av_result": {
            "success":          True,
            "agent":            "AV-Validator",
            "is_valid":         True,
            "errors":           [],
            "reasoning":        av_reasoning,
            "confidence_score": 1.0,
        },
        "ae_result": {
            "explanation": {"final_reasoning": ae_text, "final_sql": sql}
        },
        "final_sql":          sql,
        "iteration_count":    iteration_count,
        "iteration_history":  iteration_history,
    }


def _evaluate_one(agent, model: str, sample: dict,
                  f_metric, g_metric, temperature: float) -> dict:
    """
    Evaluates AS on one sample using the real agentic loop (agent.process()).

    The get_agent_reasoning skill IS invoked when the LLM decides to use it,
    consistent with production behavior.
    """
    refined        = sample["refined_query"]
    sql            = sample.get("generated_sql", "")
    user_question  = sample["user_question"]
    expected       = sample["expected_justification"]
    table_names    = [t["table_name"] for t in sample.get("selected_tables", [])]
    ag_strategy    = sample.get("ag_strategy", "simple")
    av_reasoning   = sample.get("av_reasoning", "")
    av_iterations  = sample.get("av_iterations", 0)
    ae_explanation = sample.get("ae_explanation", "")

    last_result = _build_last_result(sample)

    t0 = time.time()
    try:
        result        = agent.process(last_result=last_result,
                                      user_question=user_question,
                                      temperature=temperature)
        elapsed       = time.time() - t0
        justification = result.get("justification", "")
        confidence    = float(result.get("confidence_score", 0.75))
    except Exception as exc:
        return {
            "user_question": user_question, "expected": expected,
            "actual": "", "confidence": 0.5,
            "F": 0.0, "G": 0.0, "R": 0.0, "outcome": 0,
            "time_s": round(time.time() - t0, 2), "error": str(exc),
        }

    if not justification:
        return {
            "user_question": user_question, "expected": expected,
            "actual": "", "confidence": confidence,
            "F": 0.0, "G": 0.0, "R": 0.0, "outcome": 0,
            "time_s": round(elapsed, 2), "error": "AS returned empty justification",
        }

    # Faithfulness: justification semantically matches the expected
    tc_f = LLMTestCase(
        input=user_question,
        actual_output=justification,
        expected_output=expected,
    )
    f_metric.measure(tc_f)
    F = float(f_metric.score or 0.0)

    # Groundedness: justification must be grounded in the pipeline context
    grounding_input = (
        f"User question: {user_question}\n"
        f"Refined query: {refined}\n"
        f"SQL: {sql}\n"
        f"Strategy: {ag_strategy}\n"
        f"Tables: {', '.join(table_names) or 'n/a'}\n"
        f"Validation iterations: {av_iterations + 1}\n"
        f"AV reasoning: {av_reasoning}\n"
        f"AE explanation: {ae_explanation}"
    )
    tc_g = LLMTestCase(
        input=grounding_input,
        actual_output=justification,
        expected_output=expected,
    )
    g_metric.measure(tc_g)
    G = float(g_metric.score or 0.0)

    R       = rouge_l_score(justification, expected)
    outcome = 1 if G >= GROUNDEDNESS_THRESHOLD_FOR_OUTCOME else 0

    return {
        "user_question": user_question, "expected": expected,
        "actual": justification, "confidence": confidence,
        "F": round(F, 3), "G": round(G, 3), "R": round(R, 3),
        "outcome": outcome, "time_s": round(time.time() - t0, 2), "error": None,
    }


def evaluate_as(model: str = "gpt-4o",
                temperature: float = 0.3,
                samples: list = None,
                verbose: bool = False) -> dict:
    from agents.AS.sustainer_agent import SustainerAgent

    if samples is None:
        samples = load_samples()

    agent = SustainerAgent()
    agent.model  = model
    agent.client = get_client_for_model(model)

    f_metric = as_faithfulness_geval()
    g_metric = as_groundedness_geval()

    F_list, G_list, R_list = [], [], []
    confidences, outcomes  = [], []
    per_sample = []

    for i, sample in enumerate(samples):
        result = _evaluate_one(agent, model, sample, f_metric, g_metric, temperature)
        per_sample.append(result)

        F_list.append(result["F"])
        G_list.append(result["G"])
        R_list.append(result["R"])
        confidences.append(result["confidence"])
        outcomes.append(result["outcome"])

        if verbose:
            tag = "OK" if result["outcome"] == 1 else "X "
            print(f"  [{i+1:>2}] {tag} F={result['F']:.3f}  G={result['G']:.3f}  "
                  f"R={result['R']:.3f}  conf={result['confidence']:.2f}  "
                  f"t={result['time_s']:.2f}s")

    n   = len(per_sample)
    avg = lambda xs: round(sum(xs) / n, 3) if n else 0.0

    F_avg = avg(F_list)
    G_avg = avg(G_list)
    R_avg = avg(R_list)

    B = round(brier_score(confidences, outcomes), 4)
    E = round(ece(confidences, outcomes, n_bins=5), 3)
    combined = round((2 * F_avg + R_avg + 2 * G_avg - B - E) / 5, 3)

    return {
        "model": model, "temperature": temperature, "n_samples": n,
        "Faithfulness": F_avg, "Groundedness": G_avg, "ROUGE-L": R_avg,
        "Brier": B, "ECE": E, "combined_score": combined,
        "avg_time_s": round(sum(r["time_s"] for r in per_sample) / n, 2) if n else 0,
        "per_sample": per_sample,
    }


def save_results(metrics: dict, tag: str = "single") -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{tag}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
