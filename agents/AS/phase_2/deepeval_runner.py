# -*- coding: utf-8 -*-
"""
agents/AS/phase_3/deepeval_runner.py

Fase 3 — Validacion del AS con LLM-as-Judge (DeepEval).

Que hace:
  1. Carga config GANADORA Fase 2 (gpt-4o-mini @ T=0.3) — fija.
  2. Carga 39 paraphrases (13 intents x 3) de phase_3/paraphrases.json.
  3. Carga mock_last_result de phase_2/dataset.json.
  4. Inyecta MockLongTermMemory al AS (necesaria para Q12, Q13).
  5. Para cada paraphrase:
     - last_result = mock_last_result del intent
     - user_question = paraphrase.text
     - Corre as.process(last_result, user_question)
     - Construye LLMTestCase con context = pipeline serializado.
     - 4 metricas DeepEval (juez gpt-4o, T=0).
     - Guarda fila incremental.

Diferencia con AE Phase 3:
  - input = paraphrase del user_question (no del user_input)
  - context = ultimo pipeline state (last_result) — para Q12, Q13 se
    incluye ademas el resumen del MockLTM.
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

try:
    import config  # noqa
except ImportError:
    pass

os.environ.setdefault("ACTIVE_DATASET", "spider:concert_singer")

PARAPHRASES_PATH = os.path.join(os.path.dirname(__file__), "paraphrases.json")
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "phase_2", "dataset.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _load_paraphrases() -> dict:
    with open(PARAPHRASES_PATH, encoding="utf-8") as f:
        d = json.load(f)
    for intent in d["intents"]:
        for p in intent["paraphrases"]:
            if not (p.get("text") or "").strip():
                raise ValueError(f"Paraphrase {p['id']} vacia")
    return d


def _load_dataset_states() -> dict:
    with open(DATASET_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return {q["id"]: q["mock_last_result"] for q in d["questions"]}


def _make_as():
    """Reusa la fabrica de phase_2/main.py que inyecta MockLongTermMemory."""
    from agents.AS.phase_2.main import make_as
    return make_as()


def _run_as(as_agent, last_result: dict, user_question: str,
            temperature: float) -> str:
    """Corre AS.process. Retorna texto plano."""
    try:
        text = as_agent.process(last_result, user_question, temperature=temperature)
    except TypeError:
        text = as_agent.process(last_result, user_question)
    return (text or "").strip()


def _build_context_from_last_result(last_result: dict, intent: dict) -> list[str]:
    """
    Serializa last_result + datos LTM si es Q12/Q13.
    AS no debe inventar nada que no este aqui.
    """
    ar = last_result.get("ar_result", {}) or {}
    aps = last_result.get("aps_result", {}) or {}
    ag = last_result.get("ag_result", {}) or {}
    av = last_result.get("av_result", {}) or {}

    ctx = [
        f"AR refined query: {ar.get('refined_query', '')}",
        f"AR reasoning: {(ar.get('reasoning', '') or '')[:200]}",
        f"APS tables: {list((aps.get('tables', {}) or {}).keys())}",
        f"APS reasoning: {(aps.get('reasoning', '') or '')[:200]}",
        f"AG strategy: {ag.get('strategy', '')}",
        f"AG SQL: {ag.get('sql', '')}",
        f"AG reasoning: {(ag.get('reasoning', '') or '')[:200]}",
        f"AV valid: {av.get('is_valid', True)}, errors: {av.get('errors', [])}",
        f"final_sql: {last_result.get('final_sql', '')}",
    ]

    # Para LTM (Q12, Q13), incluir el contenido del MockLTM como ground truth
    if intent.get("id") in (12, 13):
        ctx.append(
            "MockLongTermMemory entries: "
            "['What is the average age of singers from France?' (-7d), "
            "'How many singers are there?' (-5d), "
            "'Show all stadiums with capacity above 5000' (-3d), "
            "'Names of singers ordered by age' (-1d)]"
        )

    return ctx


# ──────────────────────────────────────────────────────────────────────
# DeepEval
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
        name="JustificationQuality",
        criteria=(
            "El actual_output es una justificacion tecnica de las decisiones "
            "del pipeline AR/APS/AG/AV. Debe (1) responder a la pregunta de "
            "seguimiento del usuario, (2) apoyarse en el reasoning real del "
            "agente referido, (3) ser concisa (1 parrafo). NO debe inventar "
            "tablas, columnas, JOINs o filtros que no esten en el contexto, "
            "y NO debe volcar SQL crudo (SELECT *, INNER JOIN, etc.). "
            "Compara con expected_output como referencia."
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


def evaluate_intents(as_agent, intents: list, intent_to_state: dict,
                     judge_model: str, as_temperature: float,
                     incremental_csv: str, raw_columns: list) -> list[dict]:
    metrics = _build_metrics(judge_model)
    raw_rows = []

    if not os.path.exists(incremental_csv):
        with open(incremental_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=raw_columns).writeheader()

    for intent in intents:
        intent_id = intent["id"]
        gold = intent.get("gold_justification", "")
        last_result = intent_to_state.get(intent_id)
        if last_result is None:
            print(f"  [skip] intent {intent_id}: sin mock_last_result")
            continue

        for p in intent["paraphrases"]:
            text = p["text"]
            print(f"\n  [{p['id']}] {text[:75]}")

            t0 = time.time()
            actual = _run_as(as_agent, last_result, text, as_temperature)
            as_time = time.time() - t0
            print(f"    AS -> {actual[:75]}")

            ctx = _build_context_from_last_result(last_result, intent)
            test_case = _build_test_case(
                input_text=text,
                actual_output=actual,
                expected_output=gold,
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
                    "expected_output": gold[:600],
                    "metric":          metric_name,
                    "score":           round(score, 4),
                    "reason":          reason,
                    "error":           err,
                    "as_time_s":       round(as_time, 2),
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


def main():
    print("=" * 70)
    print("  FASE 3 - DeepEval Runner del AS")
    print("=" * 70)

    data = _load_paraphrases()
    meta = data["metadata"]
    intents = data["intents"]
    intent_to_state = _load_dataset_states()

    print(f"  Config Fase 2 ganadora: {meta['as_phase2_winner']}")
    print(f"  Juez DeepEval:          {meta['judge_model']} (T={meta['judge_temperature']})")
    print(f"  Intents:                {len(intents)}  (x 3 paraphrases = {len(intents)*3} casos)")
    print()

    try:
        import deepeval  # noqa
    except ImportError:
        print("  ERROR: deepeval no instalado")
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("  ERROR: OPENAI_API_KEY no configurado")
        sys.exit(1)

    as_model = meta["as_phase2_winner"]["model"]
    as_temperature = float(meta["as_phase2_winner"]["temperature"])
    judge_model = meta.get("judge_model", "gpt-4o")
    as_agent = _make_as()
    as_agent.model = as_model

    print("=" * 70)
    print("  Evaluando intents con DeepEval (incremental save)")
    print("=" * 70)
    raw_columns = ["intent_id", "case_id", "paraphrase_type", "input",
                   "actual_output", "expected_output", "metric", "score",
                   "reason", "error", "as_time_s", "judge_time_s"]
    raw_path = os.path.join(RESULTS_DIR, "scores_raw.csv")
    if os.path.exists(raw_path):
        os.remove(raw_path)

    raw_rows = evaluate_intents(
        as_agent, intents, intent_to_state, judge_model, as_temperature,
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
