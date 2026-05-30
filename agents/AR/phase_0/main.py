# -*- coding: utf-8 -*-
"""
agents/AR/phase_0/main.py — Evaluacion del agente AR (Fase 0 — grid completo)

Metodologia consensuada con el asesor (actualizada 2026-05-23):

  - Faithfulness  : GEval (LLM-juez gpt-4o) actual_output vs expected_output
                    Definicion AWS Bedrock 2024.
  - Groundedness  : GEval (LLM-juez gpt-4o) actual_output vs input
                    Definicion Microsoft Azure 2024.
  - ROUGE-L       : Scorer DeepEval determinista (sin juez).
                    Mide calidad LEXICA del refined_query vs expected_refined.
  - Brier Score   : sklearn.metrics.brier_score_loss sobre N muestras.
                    Mide calibracion del confidence_score del modelo.
  - ECE           : NumPy + binning sobre N muestras.

Funcion objetivo (consensuada con el asesor, 2026-05-13):
    combined_score = (2*Faithfulness + ROUGE-L + 2*Groundedness - ECE - Brier) / 5
    (max teorico = 1.0)

Outcome para calibracion (Brier y ECE):
    confidence_score = P(la pregunta es una query valida de BD) — devuelto por el AR.
    outcome = label fijo del dataset (ground truth):
        1 → la pregunta SI es una query valida de BD
        0 → la pregunta NO es valida (DML destructivo, saludo, opinion, etc.)

    ROUGE-L y Groundedness son metricas INDEPENDIENTES del outcome.
    No influyen en el outcome. Cada metrica mide una dimension distinta:
        ROUGE-L / F / G → calidad del texto refinado
        Brier / ECE     → calibracion del confidence_score

Uso programatico:
    from agents.AR.phase_0.main import evaluate_ar
    metrics = evaluate_ar(model="gemini-2.5-flash", temperature=0.5)
    score   = metrics["combined_score"]
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

# Apuntar al dataset Spider:concert_singer ANTES de instanciar el AR
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

# Outcome para Brier/ECE: label fijo del dataset (ground truth).
# outcome=1 → la pregunta ES una query valida de BD.
# outcome=0 → la pregunta NO es valida (DML, saludo, opinion).
# No depende de Groundedness ni de ROUGE-L — son metricas independientes.


# ─────────────────────────────────────────────────────────────────────
# Carga del dataset
# ─────────────────────────────────────────────────────────────────────

def load_questions(dataset_path: str = DATASET_PATH) -> list[dict]:
    """Lee el dataset Spider:concert_singer suavizado."""
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# ─────────────────────────────────────────────────────────────────────
# Evaluacion de UNA pregunta
# ─────────────────────────────────────────────────────────────────────

def _evaluate_one(agent, question: dict,
                  f_metric, g_metric,
                  temperature: float,
                  top_p: float | None = None) -> dict:
    """
    Evalua el AR sobre una pregunta y devuelve el detalle por pregunta.

    f_metric y g_metric son instancias YA construidas de GEval — se
    reutilizan a lo largo de todo el dataset. NO se reconstruyen aqui.

    El dataset es minimalista: cada pregunta solo trae input y
    expected_refined. Una pregunta es "invalida" (caso DML destructivo)
    cuando expected_refined viene vacio.
    """
    expected = question.get("expected_refined", "")
    # is_valid es el ground truth fijo del dataset (true/false explícito).
    # Fallback: si no existe el campo, se infiere de expected_refined vacío.
    is_invalid_intent = not question.get("is_valid", expected != "")

    t0 = time.time()
    try:
        r = agent.process(question["input"], temperature=temperature, top_p=top_p)
        elapsed = time.time() - t0
        actual     = r.get("refined_query", "") or ""
        confidence = float(r.get("confidence_score", 0.5))
        is_valid   = bool(r.get("is_valid_query", False))
    except Exception as e:
        return {
            "input": question["input"],
            "actual_output": "",
            "expected_output": expected,
            "confidence": 0.5,
            "F": 0.0, "G": 0.0, "R": 0.0,
            "outcome": 0,
            "time_s": 0.0,
            "error": str(e),
        }

    # Caso pregunta invalida (DML destructivo): expected_refined vacio.
    # outcome=0 siempre: ground truth es que NO es query valida.
    # F/G/R=1.0 si rechazo correctamente (miden comportamiento del AR).
    if is_invalid_intent:
        rejected = not is_valid
        return {
            "input": question["input"],
            "actual_output": actual,
            "expected_output": expected,
            "confidence": confidence,
            "F": 1.0 if rejected else 0.0,
            "G": 1.0 if rejected else 0.0,
            "R": 1.0 if rejected else 0.0,
            "outcome": 0,  # ground truth: la query NO es valida → P(valida) deberia ser bajo
            "time_s": round(elapsed, 2),
            "error": None,
        }

    # Caso pregunta valida — DeepEval evalua actual vs expected
    tc = LLMTestCase(
        input=question["input"],
        actual_output=actual,
        expected_output=expected,
    )

    f_metric.measure(tc)
    F = float(f_metric.score or 0.0)

    g_metric.measure(tc)
    G = float(g_metric.score or 0.0)

    R = rouge_l_score(actual, expected)
    # Outcome = 1 siempre para queries validas: ground truth fijo del dataset.
    # confidence_score = P(es query valida); esta pregunta SI es valida → outcome=1.
    # ROUGE-L y Groundedness miden calidad del texto, NO definen el outcome.
    outcome = 1

    return {
        "input": question["input"],
        "actual_output": actual,
        "expected_output": expected,
        "confidence": confidence,
        "F": round(F, 3),
        "G": round(G, 3),
        "R": round(R, 3),
        "outcome": outcome,
        "time_s": round(elapsed, 2),
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────
# Evaluacion completa (1 trial = 1 combinacion de hiperparametros)
# ─────────────────────────────────────────────────────────────────────

def evaluate_ar(model: str = "gpt-4o-mini",
                temperature: float = 0.1,
                top_p: float | None = None,
                questions: list = None,
                verbose: bool = False) -> dict:
    """
    Evalua el AR con (model, temperature) sobre el dataset Spider suavizado.

    Devuelve dict con metricas agregadas + detalle por pregunta.

    Flujo:
      1. Construye una instancia de Faithfulness GEval (LLM-juez)
      2. Construye una instancia de Groundedness GEval (LLM-juez)
         -> ambas se reusan para las N preguntas, NO se reconstruyen
      3. Para cada pregunta, ejecuta el AR y mide F, G, R, confidence, outcome
      4. Al terminar, calcula Brier y ECE sobre las listas paralelas
         (confidences, outcomes) usando sklearn/NumPy (NO DeepEval)
      5. combined_score = mean(F) + mean(G) + mean(R) - Brier - ECE
    """
    from agents.AR.refiner_agent import RefinerAgent

    if questions is None:
        questions = load_questions()

    agent = RefinerAgent()
    agent.model = model
    # Refrescamos el cliente segun el modelo. Para gpt-* usa OpenAI key,
    # para claude-* Anthropic, para gemini-* Google. Asi un solo
    # evaluate_ar maneja modelos cross-proveedor sin cambios al agente.
    agent.client = get_client_for_model(model)

    # Construccion UNA vez de las metricas LLM-juez. Si las construyeras
    # dentro del loop, DeepEval re-genera la rubrica interna en cada
    # llamada => costo y latencia inutiles.
    f_metric = faithfulness_geval()
    g_metric = groundedness_geval()

    F_list, G_list, R_list = [], [], []
    confidences, outcomes  = [], []
    per_question = []

    for idx, q in enumerate(questions):
        result = _evaluate_one(agent, q, f_metric, g_metric, temperature, top_p)
        per_question.append(result)

        F_list.append(result["F"])
        G_list.append(result["G"])
        R_list.append(result["R"])
        confidences.append(result["confidence"])
        outcomes.append(result["outcome"])

        if verbose:
            print(f"  [{idx:>2}] F={result['F']:.3f}  "
                  f"G={result['G']:.3f}  R={result['R']:.3f}  "
                  f"conf={result['confidence']:.2f}  "
                  f"outcome={result['outcome']}  "
                  f"t={result['time_s']:.2f}s")

    n = len(per_question)
    avg = lambda xs: round(sum(xs) / n, 3) if n else 0.0

    F_avg = avg(F_list)
    G_avg = avg(G_list)
    R_avg = avg(R_list)

    # Brier y ECE: sklearn / NumPy puros, NO DeepEval, NO LLM-juez.
    # Operan sobre las dos listas paralelas (confidence, outcome).
    B = round(brier_score(confidences, outcomes), 4)
    E = round(ece(confidences, outcomes, n_bins=5), 3)

    combined = round((2 * F_avg + R_avg + 2 * G_avg - E - B) / 5, 3)

    avg_time = round(
        sum(r["time_s"] for r in per_question) / n, 2
    ) if n else 0.0

    return {
        "model":          model,
        "temperature":    temperature,
        "top_p":          top_p,
        "n_questions":    n,
        "Faithfulness":   F_avg,
        "Groundedness":   G_avg,
        "ROUGE-L":        R_avg,
        "Brier":          B,
        "ECE":            E,
        "combined_score": combined,
        "avg_time_s":     avg_time,
        "per_question":   per_question,
    }


# ─────────────────────────────────────────────────────────────────────
# Persistencia
# ─────────────────────────────────────────────────────────────────────

def save_results(metrics: dict, tag: str = "single") -> str:
    """Guarda el dict de metricas en results/<tag>_<timestamp>.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{tag}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path
