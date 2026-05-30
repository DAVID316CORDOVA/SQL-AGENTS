# -*- coding: utf-8 -*-
"""
agents/AE/phase_3/deepeval_runner.py

Fase 3 — Validacion del AE con LLM-as-Judge (DeepEval).

Que hace:
  1. Carga config GANADORA Fase 2 (gpt-4o-mini @ T=0.3) — fija.
  2. Carga 33 paraphrases de phase_3/paraphrases.json (11 intents x 3).
  3. Carga mock_full_state de phase_2/dataset.json (mismo para las 3
     paraphrases del intent).
  4. Para cada paraphrase:
     - Sobrescribe full_state.user_input con la paraphrase
     - Corre ae.process(full_state) → narrativa unificada
     - Construye LLMTestCase con context = pipeline serializado
     - Aplica 4 metricas DeepEval (juez gpt-4o, T=0).
     - Guarda fila en scores_raw.csv (incremental).
  5. Agrega por intent → scores_summary.csv.

Diferencia clave con Fase 3 AR:
  - context = serializacion del pipeline (AR refined + APS tables +
    AG SQL + AV valid). NO el input del usuario.
  - HallucinationMetric SI aplica bien (mide si AE invento tablas o
    filtros que no estan en el pipeline real).

Llamadas estimadas: 33 paraphrases x 4 metricas x ~2 llamadas internas
= 264 a gpt-4o. Costo ~$1.50 USD.
"""

import os
import sys
import json
import csv
import time
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

# Importa config para que registre OPENAI_API_KEY en el entorno
try:
    import config  # noqa: F401
except ImportError:
    pass

os.environ.setdefault("ACTIVE_DATASET", "spider:concert_singer")

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

PARAPHRASES_PATH = os.path.join(os.path.dirname(__file__), "paraphrases.json")
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "phase_2", "dataset.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_paraphrases() -> dict:
    with open(PARAPHRASES_PATH, encoding="utf-8") as f:
        d = json.load(f)
    for i, intent in enumerate(d["intents"], 1):
        for j, p in enumerate(intent["paraphrases"], 1):
            if not (p.get("text") or "").strip():
                raise ValueError(f"Paraphrase {i}.P{j} vacia")
    return d


def _load_dataset_states() -> dict:
    """Mapea intent_id -> mock_full_state."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return {q["id"]: q["mock_full_state"] for q in d["questions"]}


def _make_ae(model: str):
    from agents.AE.explainer_agent import ExplainerAgent
    ae = ExplainerAgent()
    ae.model = model
    return ae


def _run_ae(ae, full_state: dict, temperature: float) -> dict:
    """Corre AE.process. Retorna narrativa + dict completo."""
    try:
        result = ae.process(full_state, temperature=temperature)
    except TypeError:
        result = ae.process(full_state)
    explanation_block = (result.get("explanation") or {}) if isinstance(result, dict) else {}
    return {
        "narrative": (explanation_block.get("final_reasoning") or "").strip(),
        "confidence_level": explanation_block.get("confidence_level", ""),
        "raw": result,
    }


def _build_context_from_state(full_state: dict) -> list[str]:
    """
    Serializa el estado del pipeline en strings que DeepEval usa como
    fuente de verdad. AE no debe inventar nada que no este aqui.
    """
    ar = full_state.get("ar_result", {}) or {}
    aps = full_state.get("aps_result", {}) or {}
    ag = full_state.get("ag_result", {}) or {}
    av = full_state.get("av_result", {}) or {}

    intent = ar.get("intent", {}) or {}
    tables = aps.get("tables", {}) or {}
    return [
        f"AR refined query: {ar.get('refined_query', '')}",
        f"AR intent entities: {intent.get('entities', [])}, "
        f"filters: {intent.get('filter_concepts', [])}, "
        f"aggregation: {intent.get('aggregation', [])}",
        f"APS tables: {list(tables.keys())}",
        f"APS columns: " + ", ".join(
            f"{tn}={ti.get('all_columns', [])[:6]}" for tn, ti in tables.items()
        ),
        f"AG strategy: {ag.get('strategy', '')}",
        f"AG SQL: {ag.get('sql', '')}",
        f"AV valid: {av.get('is_valid', True)}, errors: {av.get('errors', [])}, "
        f"iterations: {full_state.get('iteration_count', 1)}",
        f"final_sql: {full_state.get('final_sql', '')}",
        f"overall_confidence: {full_state.get('overall_confidence', 0)}",
    ]


# ──────────────────────────────────────────────────────────────────────
# DeepEval metrics
# ──────────────────────────────────────────────────────────────────────

def _build_metrics(judge_model: str = "gpt-4o"):
    from deepeval.metrics import (
        GEval,
        FaithfulnessMetric,
        HallucinationMetric,
        AnswerRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCaseParams

    geval = GEval(
        name="PipelineNarrativeQuality",
        criteria=(
            "El actual_output es una narrativa unificada del pipeline "
            "AR/APS/AG/AV en lenguaje simple. Debe (1) reflejar lo que se "
            "entendio de la pregunta, (2) mencionar los datos usados, "
            "(3) indicar si hubo correcciones del validador, (4) cerrar "
            "con nivel de confianza. NO debe usar jerga SQL "
            "(SELECT, FROM, JOIN, WHERE, GROUP BY) ni inventar tablas o "
            "filtros que no aparezcan en el contexto. Compara con "
            "expected_output como referencia."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge_model,
        threshold=0.5,
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
                     expected_output: str, ctx: list[str]):
    from deepeval.test_case import LLMTestCase
    return LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output,
        context=ctx,
        retrieval_context=ctx,
    )


# ──────────────────────────────────────────────────────────────────────
# Evaluacion principal
# ──────────────────────────────────────────────────────────────────────

def evaluate_intents(ae, intents: list, intent_to_state: dict,
                     judge_model: str, ae_temperature: float,
                     incremental_csv: str, raw_columns: list) -> list[dict]:
    metrics = _build_metrics(judge_model)
    raw_rows = []

    if not os.path.exists(incremental_csv):
        with open(incremental_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=raw_columns).writeheader()

    for intent in intents:
        intent_id = intent["id"]
        gold_explanation = intent.get("gold_explanation", "")
        base_state = intent_to_state.get(intent_id)
        if base_state is None:
            print(f"  [skip] intent {intent_id}: sin mock_full_state")
            continue

        for p in intent["paraphrases"]:
            text = p["text"]
            print(f"\n  [{p['id']}] {text[:75]}")

            # full_state con paraphrase como user_input
            full_state = dict(base_state)
            full_state["user_input"] = text

            t0 = time.time()
            ae_out = _run_ae(ae, full_state, ae_temperature)
            ae_time = time.time() - t0
            actual = ae_out["narrative"]
            print(f"    AE -> {actual[:75]}")

            ctx = _build_context_from_state(full_state)
            test_case = _build_test_case(
                input_text=text,
                actual_output=actual,
                expected_output=gold_explanation,
                ctx=ctx,
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
                    "intent_id":       intent_id,
                    "case_id":         p["id"],
                    "paraphrase_type": p.get("type", ""),
                    "input":           text,
                    "actual_output":   actual[:600],
                    "expected_output": gold_explanation[:600],
                    "metric":          metric_name,
                    "score":           round(score, 4),
                    "reason":          reason,
                    "error":           err,
                    "ae_time_s":       round(ae_time, 2),
                    "judge_time_s":    round(judge_time, 2),
                }
                raw_rows.append(row)
                with open(incremental_csv, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=raw_columns).writerow(
                        {k: row.get(k, "") for k in raw_columns})
    return raw_rows


def aggregate_summary(raw_rows: list[dict]) -> list[dict]:
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
    print("  FASE 3 - DeepEval Runner del AE")
    print("=" * 70)

    data = _load_paraphrases()
    meta = data["metadata"]
    intents = data["intents"]
    intent_to_state = _load_dataset_states()

    print(f"  Config Fase 2 ganadora: {meta['ae_phase2_winner']}")
    print(f"  Juez DeepEval:          {meta['judge_model']} (T={meta['judge_temperature']})")
    print(f"  Intents:                {len(intents)}  (x 3 paraphrases)")
    print()

    try:
        import deepeval  # noqa
    except ImportError:
        print("  ERROR: deepeval no esta instalado. pip install deepeval")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("  ERROR: OPENAI_API_KEY no esta configurado")
        sys.exit(1)

    ae_model = meta["ae_phase2_winner"]["model"]
    ae_temperature = float(meta["ae_phase2_winner"]["temperature"])
    judge_model = meta.get("judge_model", "gpt-4o")
    ae = _make_ae(ae_model)

    print("=" * 70)
    print("  Evaluando intents con DeepEval (incremental save)")
    print("=" * 70)
    raw_columns = ["intent_id", "case_id", "paraphrase_type", "input",
                   "actual_output", "expected_output", "metric", "score",
                   "reason", "error", "ae_time_s", "judge_time_s"]
    raw_path = os.path.join(RESULTS_DIR, "scores_raw.csv")
    if os.path.exists(raw_path):
        os.remove(raw_path)

    raw_rows = evaluate_intents(
        ae, intents, intent_to_state, judge_model, ae_temperature,
        incremental_csv=raw_path, raw_columns=raw_columns,
    )

    summary_rows = aggregate_summary(raw_rows)

    summary_path = os.path.join(RESULTS_DIR, "scores_summary.csv")
    summary_columns = ["intent_id",
                       "GEval_mean", "GEval_var",
                       "Faithfulness_mean", "Faithfulness_var",
                       "Hallucination_mean", "Hallucination_var",
                       "AnswerRelevancy_mean", "AnswerRelevancy_var",
                       "max_variance", "robust"]
    save_csv(summary_path, summary_rows, summary_columns)

    print()
    print("=" * 70)
    print("  RESULTADO")
    print("=" * 70)
    print(f"  scores_raw.csv      : {len(raw_rows)} filas -> {raw_path}")
    print(f"  scores_summary.csv  : {len(summary_rows)} filas -> {summary_path}")
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


if __name__ == "__main__":
    main()
