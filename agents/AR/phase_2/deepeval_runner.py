# -*- coding: utf-8 -*-
"""
agents/AR/phase_3/deepeval_runner.py

Fase 3 — Validacion del AR con LLM-as-Judge (DeepEval).

Que hace:
  1. Carga la config GANADORA de Fase 2 (gpt-4o, temperature=0.1) — fija.
  2. Carga 33 paraphrases (Q1-Q11 con 3 c/u + Q12 invalida con 3 c/u).
  3. Para Q1-Q11 (validas):
     - Corre AR.process(paraphrase) -> refined_query
     - Construye LLMTestCase y aplica 4 metricas DeepEval con juez gpt-4o.
     - Guarda scores en results/scores_raw.csv (132 filas).
  4. Para Q12 (invalida):
     - Corre AR.process(paraphrase) -> refined_query
     - Auditoria binaria: ¿AR rechazo? ¿menciono keywords esperados?
     - Guarda en results/rejection_audit.csv (3 filas).
  5. Agrega por intent (mean + variance entre paraphrases) en scores_summary.csv.

Las 4 metricas de DeepEval:
  - GEval (custom)         : criterio "preserva intent sin alucinar"
  - FaithfulnessMetric     : ¿el output es fiel al input?
  - HallucinationMetric    : ¿AR invento algo que no estaba en input?
  - AnswerRelevancyMetric  : ¿el refined_query es relevante a la pregunta?

Llamadas estimadas: 33 paraphrases x 4 metricas x ~2 llamadas internas ≈ 264 a gpt-4o.
Costo estimado: ~$1.50 USD. Tiempo: ~5 min en serie.
"""

import os
import sys
import json
import csv
import time
import statistics
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

# Importa config.py para que registre OPENAI_API_KEY en el entorno
# (el proyecto guarda la clave en config.py, no en .env)
try:
    import config  # noqa: F401
except ImportError:
    pass

os.environ.setdefault("ACTIVE_DATASET", "spider:concert_singer")

# ──────────────────────────────────────────────────────────────────────
# Constantes y config
# ──────────────────────────────────────────────────────────────────────

PARAPHRASES_PATH = os.path.join(os.path.dirname(__file__), "paraphrases.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_paraphrases() -> dict:
    with open(PARAPHRASES_PATH, encoding="utf-8") as f:
        d = json.load(f)
    # Validacion: todas las paraphrases deben tener texto no-vacio
    for intent in d["intents"]:
        for p in intent["paraphrases"]:
            if not (p.get("text") or "").strip():
                raise ValueError(
                    f"Paraphrase {p['id']} esta vacia. Llenala antes de correr."
                )
    return d


def _make_ar(model: str, temperature: float):
    """Instancia el AR con la config GANADORA de Fase 2 fija."""
    from agents.AR.refiner_agent import RefinerAgent
    ar = RefinerAgent()
    ar.model = model
    ar._phase3_temperature = temperature
    return ar


def _run_ar(ar, text: str, temperature: float) -> dict:
    """Corre AR.process. Retorna dict con refined_query e is_valid_query."""
    try:
        result = ar.process(text, temperature=temperature)
    except TypeError:
        # Fallback: si process no acepta temperature, prueba sin el kwarg
        result = ar.process(text)
    return {
        "refined_query": result.get("refined_query", "") or "",
        "is_valid_query": bool(result.get("is_valid_query", True)),
        "rejection_reason": result.get("rejection_reason", "") or "",
        "confidence_score": float(result.get("confidence_score", 0.5) or 0.5),
        "raw": result,
    }


# ──────────────────────────────────────────────────────────────────────
# DeepEval setup
# ──────────────────────────────────────────────────────────────────────

def _build_metrics(judge_model: str = "gpt-4o"):
    """
    Construye las 4 metricas DeepEval. El juez NO tiene threshold relevante
    (reportamos score raw). El judge_model corre con temperature=0 implicito.
    """
    from deepeval.metrics import (
        GEval,
        FaithfulnessMetric,
        HallucinationMetric,
        AnswerRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCaseParams

    geval = GEval(
        name="IntentPreserved",
        criteria=(
            "El actual_output debe preservar todas las entidades, filtros y "
            "agregaciones del input sin inventar campos nuevos. Debe ser "
            "semanticamente equivalente al expected_output."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge_model,
        threshold=0.5,  # placeholder, no afecta el score reportado
    )

    faith = FaithfulnessMetric(model=judge_model, threshold=0.5)
    hall  = HallucinationMetric(model=judge_model, threshold=0.5)
    rel   = AnswerRelevancyMetric(model=judge_model, threshold=0.5)

    return [
        ("GEval", geval),
        ("Faithfulness", faith),
        ("Hallucination", hall),
        ("AnswerRelevancy", rel),
    ]


def _build_test_case(input_text: str, actual_output: str,
                     expected_output: str):
    """Crea un LLMTestCase con todos los campos que cualquier metrica necesita."""
    from deepeval.test_case import LLMTestCase
    return LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output,
        context=[input_text],            # para HallucinationMetric
        retrieval_context=[input_text],  # para FaithfulnessMetric
    )


# ──────────────────────────────────────────────────────────────────────
# Evaluacion principal
# ──────────────────────────────────────────────────────────────────────

def evaluate_valid_intents(ar, valid_intents: list, judge_model: str,
                           ar_temperature: float,
                           incremental_csv: str = None,
                           raw_columns: list = None) -> list[dict]:
    """Corre AR + 4 metricas DeepEval en cada paraphrase de cada intent valido.

    Si incremental_csv esta dado, escribe cada fila inmediatamente despues de
    medirla. Asi un crash a mitad no pierde datos.
    """
    metrics = _build_metrics(judge_model)
    raw_rows = []

    # Header del CSV incremental (modo append)
    if incremental_csv and raw_columns:
        write_header = not os.path.exists(incremental_csv)
        if write_header:
            with open(incremental_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=raw_columns).writeheader()

    for intent in valid_intents:
        gold = intent["gold_refined_query"]
        for p in intent["paraphrases"]:
            text = p["text"]
            print(f"\n  [{p['id']}] {text[:70]}")
            t0 = time.time()
            ar_out = _run_ar(ar, text, temperature=ar_temperature)
            ar_time = time.time() - t0
            actual = ar_out["refined_query"]
            print(f"    AR -> {actual[:70]}")

            test_case = _build_test_case(
                input_text=text,
                actual_output=actual,
                expected_output=gold,
            )

            for metric_name, metric in metrics:
                t1 = time.time()
                try:
                    metric.measure(test_case)
                    score = float(metric.score) if metric.score is not None else 0.0
                    reason = (getattr(metric, "reason", "") or "")[:300]
                    err = ""
                except Exception as exc:
                    score = 0.0
                    reason = ""
                    err = f"{type(exc).__name__}: {exc}"
                judge_time = time.time() - t1
                print(f"    {metric_name:18} score={score:.3f}  ({judge_time:.1f}s)")
                row = {
                    "intent_id":       intent["id"],
                    "case_id":         p["id"],
                    "paraphrase_type": p.get("type", ""),
                    "input":           text,
                    "actual_output":   actual,
                    "expected_output": gold,
                    "metric":          metric_name,
                    "score":           round(score, 4),
                    "reason":          reason,
                    "error":           err,
                    "ar_time_s":       round(ar_time, 2),
                    "judge_time_s":    round(judge_time, 2),
                }
                raw_rows.append(row)
                # Escritura incremental — sobrevive a crashes
                if incremental_csv and raw_columns:
                    with open(incremental_csv, "a", newline="", encoding="utf-8") as f:
                        csv.DictWriter(f, fieldnames=raw_columns).writerow(
                            {k: row.get(k, "") for k in raw_columns})

    return raw_rows


def audit_invalid_intent(ar, invalid_intent: dict,
                         ar_temperature: float) -> list[dict]:
    """Auditoria binaria de Q12 (rechazos): NO usa DeepEval."""
    keywords_lower = [k.lower() for k in invalid_intent.get("rejection_reason_keywords", [])]
    rows = []
    for p in invalid_intent["paraphrases"]:
        text = p["text"]
        print(f"\n  [{p['id']}] {text}")
        ar_out = _run_ar(ar, text, temperature=ar_temperature)
        was_rejected = not ar_out["is_valid_query"]
        rejection_reason = (ar_out["rejection_reason"] or "").lower()
        keyword_match = any(k in rejection_reason for k in keywords_lower) if keywords_lower else None
        outcome = "OK" if was_rejected else "FAIL"
        print(f"    AR rejected={was_rejected}  keyword_match={keyword_match}  -> {outcome}")
        rows.append({
            "case_id":          p["id"],
            "input":            text,
            "ar_rejected":      was_rejected,
            "rejection_reason": (ar_out["rejection_reason"] or "")[:300],
            "keyword_match":    keyword_match,
            "outcome":          outcome,
        })
    return rows


def aggregate_summary(raw_rows: list[dict]) -> list[dict]:
    """Agrega scores raw por intent_id × metric -> mean + variance entre paraphrases."""
    by_key = {}
    for r in raw_rows:
        key = (r["intent_id"], r["metric"])
        by_key.setdefault(key, []).append(r["score"])

    summary_by_intent = {}
    for (intent_id, metric_name), scores in by_key.items():
        mean = statistics.mean(scores) if scores else 0.0
        var  = statistics.pvariance(scores) if len(scores) > 1 else 0.0
        summary_by_intent.setdefault(intent_id, {"intent_id": intent_id})
        summary_by_intent[intent_id][f"{metric_name}_mean"] = round(mean, 4)
        summary_by_intent[intent_id][f"{metric_name}_var"]  = round(var, 4)

    out = list(summary_by_intent.values())
    out.sort(key=lambda r: r["intent_id"])

    # Robusto = todas las metricas tienen var < 0.05
    for row in out:
        var_keys = [k for k in row if k.endswith("_var")]
        max_var = max((row[k] for k in var_keys), default=0.0)
        row["max_variance"] = round(max_var, 4)
        row["robust"] = bool(max_var < 0.05)
    return out


def save_csv(path: str, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  FASE 3 — DeepEval Runner del AR")
    print("=" * 70)

    data = _load_paraphrases()
    meta = data["metadata"]
    valid_intents = [i for i in data["intents"] if i["valid"]]
    invalid_intents = [i for i in data["intents"] if not i["valid"]]

    print(f"  Config Fase 2 ganadora: {meta['ar_phase2_winner']}")
    print(f"  Juez DeepEval:          {meta['judge_model']} (T={meta['judge_temperature']})")
    print(f"  Intents validos:        {len(valid_intents)}  (× 3 paraphrases)")
    print(f"  Intents invalidos:      {len(invalid_intents)} (× 3 paraphrases)")
    print()

    # Verificar deepeval instalado
    try:
        import deepeval  # noqa: F401
    except ImportError:
        print("  ERROR: deepeval no esta instalado.")
        print("  Ejecuta: pip install -U deepeval")
        sys.exit(1)

    # Verificar OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("  ERROR: OPENAI_API_KEY no esta configurado en .env ni en el entorno.")
        sys.exit(1)

    ar_model = meta["ar_phase2_winner"]["model"]
    ar_temperature = float(meta["ar_phase2_winner"]["temperature"])
    judge_model = meta.get("judge_model", "gpt-4o")
    ar = _make_ar(ar_model, ar_temperature)

    # 1. Validas -> DeepEval (escritura incremental, sobrevive crashes)
    print("=" * 70)
    print("  Evaluando intents VALIDOS con DeepEval (incremental save)")
    print("=" * 70)
    raw_columns = ["intent_id", "case_id", "paraphrase_type", "input",
                   "actual_output", "expected_output", "metric", "score",
                   "reason", "error", "ar_time_s", "judge_time_s"]
    raw_path = os.path.join(RESULTS_DIR, "scores_raw.csv")
    if os.path.exists(raw_path):
        os.remove(raw_path)
    raw_rows = evaluate_valid_intents(
        ar, valid_intents, judge_model, ar_temperature,
        incremental_csv=raw_path, raw_columns=raw_columns,
    )

    # 2. Invalidas -> auditoria binaria
    print()
    print("=" * 70)
    print("  Auditando intents INVALIDOS (rechazo binario)")
    print("=" * 70)
    audit_rows = []
    for inv in invalid_intents:
        audit_rows.extend(audit_invalid_intent(ar, inv, ar_temperature))

    # 3. Resumen
    summary_rows = aggregate_summary(raw_rows)

    # 4. Persistir summary y audit (raw ya se escribio incremental)
    summary_path = os.path.join(RESULTS_DIR, "scores_summary.csv")
    audit_path = os.path.join(RESULTS_DIR, "rejection_audit.csv")

    summary_columns = ["intent_id",
                       "GEval_mean", "GEval_var",
                       "Faithfulness_mean", "Faithfulness_var",
                       "Hallucination_mean", "Hallucination_var",
                       "AnswerRelevancy_mean", "AnswerRelevancy_var",
                       "max_variance", "robust"]
    audit_columns = ["case_id", "input", "ar_rejected", "rejection_reason",
                     "keyword_match", "outcome"]

    save_csv(summary_path, summary_rows, summary_columns)
    save_csv(audit_path, audit_rows, audit_columns)

    # 5. Reporte rapido
    print()
    print("=" * 70)
    print("  RESULTADO")
    print("=" * 70)
    print(f"  scores_raw.csv       : {len(raw_rows)} filas -> {raw_path}")
    print(f"  scores_summary.csv   : {len(summary_rows)} filas -> {summary_path}")
    print(f"  rejection_audit.csv  : {len(audit_rows)} filas -> {audit_path}")
    print()

    if raw_rows:
        for metric_name in ("GEval", "Faithfulness", "Hallucination", "AnswerRelevancy"):
            scores = [r["score"] for r in raw_rows if r["metric"] == metric_name]
            if scores:
                m = statistics.mean(scores)
                s = statistics.pstdev(scores)
                print(f"  {metric_name:18} mean={m:.3f}  stdev={s:.3f}  n={len(scores)}")

    robust = sum(1 for r in summary_rows if r["robust"])
    print()
    print(f"  Intents ROBUSTOS (max_var<0.05): {robust} / {len(summary_rows)}")
    print(f"  Q12 rechazos correctos:           "
          f"{sum(1 for r in audit_rows if r['outcome']=='OK')} / {len(audit_rows)}")


if __name__ == "__main__":
    main()
