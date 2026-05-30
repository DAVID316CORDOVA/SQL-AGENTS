# -*- coding: utf-8 -*-
"""
agents/AS/phase_2/main.py - AS Fase 2: paraphrase invariance.

10 intents x 3 paraphrases = 30 inputs.
Same pipeline_state and expected_justification per intent.
Only the formulation of the meta-question varies across paraphrases.

Uses the real agentic loop (agent.process()) — consistent with Phase 1
and production behavior. get_agent_reasoning skill IS invoked.

Formula: (2F + R + 2G - B - E) / 5  (max 1.0)
Outcome Brier/ECE: G >= 0.80
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
    as_faithfulness_geval, as_groundedness_geval,
    rouge_l_score, brier_score, ece,
)
from llm_client import get_client_for_model

PARAPHRASES_PATH = os.path.join(os.path.dirname(__file__), "paraphrases.json")
RESULTS_DIR      = os.path.join(os.path.dirname(__file__), "results")

# Consistent with Phase 1
GROUNDEDNESS_THRESHOLD_FOR_OUTCOME = 0.80


def load_intents(path: str = PARAPHRASES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["intents"]


def evaluate_as_paraphrases(model: str = "claude-sonnet-4-6",
                             temperature: float = 0.3,
                             intents: list = None,
                             verbose: bool = False) -> dict:
    """
    Evaluates AS with (model, temperature) fixed over all paraphrases.

    For each intent:
      - The pipeline state (last_result) is IDENTICAL for all 3 paraphrases.
      - Only the user_question changes.

    This measures: does AS produce equivalent justifications regardless of
    how the meta-question is phrased?
    """
    from agents.AS.sustainer_agent import SustainerAgent
    from agents.AS.phase_1.main import _build_last_result

    if intents is None:
        intents = load_intents()

    agent = SustainerAgent()
    agent.model  = model
    agent.client = get_client_for_model(model)

    f_metric = as_faithfulness_geval()
    g_metric = as_groundedness_geval()

    F_list, G_list, R_list = [], [], []
    confidences, outcomes  = [], []
    per_paraphrase = []
    per_intent_stats = []

    for i_idx, intent in enumerate(intents):
        expected     = intent["expected_justification"]
        refined      = intent["refined_query"]
        sql          = intent["generated_sql"]
        tables       = [t["table_name"] for t in intent.get("selected_tables", [])]
        ag_strategy  = intent.get("ag_strategy", "simple")
        av_reasoning = intent.get("av_reasoning", "")
        ae_text      = intent.get("ae_explanation", "")

        # Build pipeline state ONCE per intent — same for all 3 paraphrases
        last_result = _build_last_result(intent)

        if verbose:
            print(f"\n  Intent #{i_idx+1} - {refined[:60]}")

        intent_F, intent_G, intent_R = [], [], []

        for p_idx, question in enumerate(intent["paraphrases"]):
            t0 = time.time()
            try:
                result        = agent.process(last_result=last_result,
                                              user_question=question,
                                              temperature=temperature)
                elapsed       = time.time() - t0
                justification = result.get("justification", "")
                confidence    = float(result.get("confidence_score", 0.75))
            except Exception as exc:
                elapsed = time.time() - t0
                per_paraphrase.append({
                    "intent_idx": i_idx + 1, "paraphrase_idx": p_idx + 1,
                    "question": question, "expected": expected,
                    "actual": "", "confidence": 0.5,
                    "F": 0.0, "G": 0.0, "R": 0.0, "outcome": 0,
                    "time_s": round(elapsed, 2), "error": str(exc),
                })
                F_list.append(0.0); G_list.append(0.0); R_list.append(0.0)
                confidences.append(0.5); outcomes.append(0)
                intent_F.append(0.0); intent_G.append(0.0); intent_R.append(0.0)
                continue

            if not justification:
                per_paraphrase.append({
                    "intent_idx": i_idx + 1, "paraphrase_idx": p_idx + 1,
                    "question": question, "expected": expected,
                    "actual": "", "confidence": confidence,
                    "F": 0.0, "G": 0.0, "R": 0.0, "outcome": 0,
                    "time_s": round(elapsed, 2), "error": "AS returned empty justification",
                })
                F_list.append(0.0); G_list.append(0.0); R_list.append(0.0)
                confidences.append(confidence); outcomes.append(0)
                intent_F.append(0.0); intent_G.append(0.0); intent_R.append(0.0)
                continue

            # Faithfulness
            tc_f = LLMTestCase(input=question, actual_output=justification,
                               expected_output=expected)
            f_metric.measure(tc_f)
            F = float(f_metric.score or 0.0)

            # Groundedness
            grounding_input = (
                f"User question: {question}\n"
                f"Refined query: {refined}\n"
                f"SQL: {sql}\n"
                f"Strategy: {ag_strategy}\n"
                f"Tables: {', '.join(tables)}\n"
                f"AV reasoning: {av_reasoning}\n"
                f"AE explanation: {ae_text}"
            )
            tc_g = LLMTestCase(input=grounding_input, actual_output=justification,
                               expected_output=expected)
            g_metric.measure(tc_g)
            G = float(g_metric.score or 0.0)

            R       = rouge_l_score(justification, expected)
            outcome = 1 if G >= GROUNDEDNESS_THRESHOLD_FOR_OUTCOME else 0

            F_list.append(F); G_list.append(G); R_list.append(R)
            confidences.append(confidence); outcomes.append(outcome)
            intent_F.append(F); intent_G.append(G); intent_R.append(R)

            per_paraphrase.append({
                "intent_idx":    i_idx + 1,
                "paraphrase_idx": p_idx + 1,
                "question":      question,
                "expected":      expected,
                "actual":        justification,
                "confidence":    confidence,
                "F": round(F, 3), "G": round(G, 3), "R": round(R, 3),
                "outcome":       outcome,
                "time_s":        round(elapsed, 2),
                "error":         None,
            })

            if verbose:
                tag = f"{i_idx+1}.P{p_idx+1}"
                print(f"    [{tag:>5}] "
                      f"F={F:.3f}  G={G:.3f}  R={R:.3f}  "
                      f"conf={confidence:.2f}  outcome={outcome}  "
                      f"t={elapsed:.2f}s")

        m = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0
        per_intent_stats.append({
            "intent_idx": i_idx + 1,
            "F_mean": m(intent_F),
            "G_mean": m(intent_G),
            "R_mean": m(intent_R),
        })

    n   = len(per_paraphrase)
    avg = lambda xs: round(sum(xs) / n, 3) if n else 0.0

    F_avg = avg(F_list)
    G_avg = avg(G_list)
    R_avg = avg(R_list)

    B = round(brier_score(confidences, outcomes), 4)
    E = round(ece(confidences, outcomes, n_bins=5), 3)
    combined = round((2 * F_avg + R_avg + 2 * G_avg - B - E) / 5, 3)

    avg_time = round(
        sum(r.get("time_s", 0) for r in per_paraphrase) / n, 2
    ) if n else 0.0

    return {
        "model":           model,
        "temperature":     temperature,
        "n_paraphrases":   n,
        "n_intents":       len(intents),
        "Faithfulness":    F_avg,
        "Groundedness":    G_avg,
        "ROUGE-L":         R_avg,
        "Brier":           B,
        "ECE":             E,
        "combined_score":  combined,
        "avg_time_s":      avg_time,
        "per_intent":      per_intent_stats,
        "per_paraphrase":  per_paraphrase,
    }


def save_results(metrics: dict, tag: str = "phase2") -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{tag}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
