# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
agents/AR/phase_2/main.py — Evaluacion del AR (Fase 2, paraphrase invariance)

Mismo conjunto de metricas que Fase 1 (Faithfulness, Groundedness, ROUGE-L,
Brier, ECE) y misma funcion objetivo (asesor 2026-05-13):

    combined_score = (2*Faithfulness + ROUGE-L + 2*Groundedness - ECE - Brier) / 5
    (max teorico = 1.0)

La diferencia con Fase 1 es el dataset:

  - Fase 1: ~10 preguntas con typos suaves; cada input tiene su propia
            respuesta esperada.
  - Fase 2: 12 intents x 3 paraphrases = 36 inputs distintos. Las 3
            paraphrases de cada intent comparten la MISMA respuesta esperada
            (el agente debe converger al mismo refinado sin importar como
            se haya escrito el input).

El campo de gold se llama `expected_refined_query`. Cada paraphrase es un
string plano dentro de la lista `paraphrases`.

Hiperparametros: FIJOS (los ganadores de Fase 1). NO se usa Optuna.

Outcome para calibracion (Brier y ECE):
    confidence_score = P(la pregunta es una query valida de BD) — devuelto por el AR.
    outcome = label fijo del dataset (ground truth):
        1 → la pregunta SI es una query valida de BD
        0 → la pregunta NO es valida (DML destructivo, saludo, opinion, etc.)

    ROUGE-L y Groundedness son metricas INDEPENDIENTES del outcome.
    No influyen en el outcome. Cada metrica mide una dimension distinta:
        ROUGE-L / F / G → calidad del texto refinado
        Brier / ECE     → calibracion del confidence_score
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

# Outcome para Brier/ECE: label fijo del dataset (ground truth).
# outcome=1 → la pregunta ES una query valida de BD.
# outcome=0 → la pregunta NO es valida (DML, saludo, opinion).
# No depende de Groundedness ni de ROUGE-L — son metricas independientes.


def load_intents(path: str = PARAPHRASES_PATH) -> list[dict]:
    """Lee paraphrases.json y devuelve la lista de intents."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


def _evaluate_paraphrase(agent, intent_idx: int, paraphrase_idx: int,
                         intent: dict, paraphrase: str,
                         f_metric, g_metric,
                         temperature: float) -> dict:
    """
    Evalua el AR sobre UNA paraphrase del intent. Devuelve dict con
    F, G, R, confidence, outcome y metadatos.

    El dataset minimalista identifica el caso invalido (intent DML) por
    expected_refined_query vacio. paraphrase es un string plano.
    """
    expected = intent.get("expected_refined_query", "")
    # is_valid es el ground truth fijo del dataset (true/false explícito).
    # Fallback: si no existe el campo, se infiere de expected_refined_query vacío.
    is_invalid_intent = not intent.get("is_valid", expected != "")

    t0 = time.time()
    try:
        r = agent.process(paraphrase, temperature=temperature)
        elapsed = time.time() - t0
        actual     = r.get("refined_query", "") or ""
        confidence = float(r.get("confidence_score", 0.5))
        is_valid   = bool(r.get("is_valid_query", False))
    except Exception as e:
        return {
            "intent_idx":     intent_idx,
            "paraphrase_idx": paraphrase_idx,
            "input":          paraphrase,
            "actual_output":  "",
            "expected_output": expected,
            "confidence":     0.5,
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome": 0,
            "time_s": 0.0,
            "error": str(e),
        }

    # Caso DML: expected_refined_query vacio. El AR debe rechazarlo.
    # outcome=0 siempre: ground truth es que NO es query valida de BD.
    # F/G/R=1.0 si rechazo correctamente (miden comportamiento del AR).
    if is_invalid_intent:
        rejected = not is_valid
        return {
            "intent_idx":     intent_idx,
            "paraphrase_idx": paraphrase_idx,
            "input":          paraphrase,
            "actual_output":  actual,
            "expected_output": "",
            "confidence":     confidence,
            "F": 1.0 if rejected else 0.0,
            "G": 1.0 if rejected else 0.0,
            "R": 1.0 if rejected else 0.0,
            "outcome": 0,  # ground truth: la query NO es valida → P(valida) deberia ser bajo
            "time_s": round(elapsed, 2),
            "error": None,
        }

    # Caso valido: DeepEval evalua actual_output vs respuesta esperada
    tc = LLMTestCase(
        input=paraphrase,
        actual_output=actual,
        expected_output=expected,
    )

    f_metric.measure(tc)
    F = float(f_metric.score or 0.0)

    g_metric.measure(tc)
    G = float(g_metric.score or 0.0)

    R = rouge_l_score(actual, expected)
    # Outcome = 1 siempre para queries validas: ground truth fijo del dataset.
    # confidence_score = P(es query valida); esta query SI es valida → outcome=1.
    # ROUGE-L y Groundedness miden calidad del texto, NO definen el outcome.
    outcome = 1

    return {
        "intent_idx":     intent_idx,
        "paraphrase_idx": paraphrase_idx,
        "input":          paraphrase,
        "actual_output":  actual,
        "expected_output": expected,
        "confidence":     confidence,
        "F": round(F, 3),
        "G": round(G, 3),
        "R": round(R, 3),
        "outcome": outcome,
        "time_s": round(elapsed, 2),
        "error": None,
    }



def evaluate_ar_paraphrases(model: str = "gpt-4o-mini",
                            temperature: float = 0.1,
                            intents: list = None,
                            verbose: bool = False) -> dict:
    """
    Evalua el AR con (model, temperature) fijos sobre las 36 paraphrases.

    Usa las mismas 5 metricas y la misma combined_score formula que Fase 1:
        combined = (2*F_avg + R_avg + 2*G_avg - E - B) / 5  (max 1.0)

    No hace busqueda de hiperparametros: model y temperature se reciben fijos
    (los ganadores de Fase 1).
    """
    from agents.AR.refiner_agent import RefinerAgent

    if intents is None:
        intents = load_intents()

    agent = RefinerAgent()
    agent.model  = model
    # Refresca el cliente segun el modelo (gpt-* OpenAI, claude-* Anthropic).
    # Imprescindible cuando Fase 3 corre con el ganador de Fase 1 si ese
    # ganador es de un proveedor distinto al default.
    agent.client = get_client_for_model(model)

    # Construccion UNA vez de las metricas LLM-juez.
    f_metric = faithfulness_geval()
    g_metric = groundedness_geval()

    F_list, G_list, R_list = [], [], []
    confidences, outcomes  = [], []
    per_paraphrase = []

    for i, intent in enumerate(intents):
        is_invalid = (intent.get("expected_refined_query", "") == "")
        if verbose:
            print(f"\n  Intent #{i+1} "
                  f"({'INVALID/DML' if is_invalid else 'valid'})")
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

            if verbose:
                tag = f"{result['intent_idx']}.P{result['paraphrase_idx']}"
                print(f"    [{tag:>5}] "
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
        "per_paraphrase": per_paraphrase,
    }


def save_results(metrics: dict, tag: str = "phase3") -> str:
    """Guarda el dict de metricas en results/<tag>_<timestamp>.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{tag}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
