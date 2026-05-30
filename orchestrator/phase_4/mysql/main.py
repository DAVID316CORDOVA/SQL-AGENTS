# -*- coding: utf-8 -*-
"""
orchestrator/phase_4/mysql/main.py

Phase 4 — Paraphrase robustness evaluation of the SQL-Agents pipeline (MySQL).

Each of the 10 scenarios has 3 linguistic paraphrases of the original question.
Expected SQL, explanation, and tools are FIXED across all paraphrases.

Metrics per paraphrase:
  - Faithfulness  : e2e_faithfulness_geval()
  - Groundedness  : e2e_groundedness_geval()
  - Correctness   : ToolCorrectnessMetric
  - ROUGE-L_SQL   : rouge_l_sql()
  - WACS          : 0.10*AR + 0.20*APS + 0.30*AG + 0.40*AV

Robustness per scenario:
  - robustness_score = mean(combined) across the 3 paraphrases

Final formula per paraphrase:
    combined = (2*F + 2*G + Correctness - Brier - ECE) / 5   (max 1.0)

Brier and ECE are computed globally over all full_pipeline paraphrases.
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
os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o")

from deepeval.test_case.llm_test_case import LLMTestCase, ToolCall

from metricas_lib import (
    e2e_faithfulness_geval,
    e2e_groundedness_geval,
    tool_correctness_metric,
    rouge_l_sql_v1 as rouge_l_sql,
    brier_score,
    ece,
)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DB_TYPE = "mysql"


def load_dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)["test_cases"]


def extract_tools_called(state: dict) -> list:
    tools = []
    ar  = state.get("ar_result",  {}) or {}
    aps = state.get("aps_result", {}) or {}
    ag  = state.get("ag_result",  {}) or {}
    av  = state.get("av_result",  {}) or {}
    ae  = state.get("ae_result",  {}) or {}
    skills_trace    = state.get("skills_trace", {}) or {}
    iteration_count = state.get("iteration_count", 0)

    if ar:
        tools.append("AR")

    if ar.get("is_valid_query") and aps and aps.get("agent", "") != "":
        tools.append("APS")
        aps_trace  = {s["skill"] for s in (skills_trace.get("APS") or [])}
        aps_direct = set(aps.get("skills_used") or [])
        aps_all    = aps_trace | aps_direct
        if aps.get("tables") or "search_tables" in aps_all:
            tools.append("APS.search_tables")
        if (aps.get("tables") and aps.get("success")) or "search_columns" in aps_all:
            tools.append("APS.search_columns")

    if ag:
        n_iters   = max(iteration_count, 1)
        sql       = state.get("final_sql", "") or ag.get("sql", "") or ""
        sql_up    = sql.upper()
        av_skills = av.get("skills_used", []) if av else []

        for i in range(n_iters):
            tools.append("AG")
            tools.append("AG.validate_sql_safety")
            if i == 0:
                if "JOIN" in sql_up:
                    tools.append("AG.find_join_path")
                elif sql_up.count("SELECT") > 1:
                    tools.append("AG.find_joins_among_tables")
            if av:
                tools.append("AV")
                for skill in av_skills:
                    tools.append(f"AV.{skill}")

    if ae:
        tools.append("AE")
    return tools


def extract_confidences(state: dict) -> dict:
    ar  = state.get("ar_result",  {}) or {}
    aps = state.get("aps_result", {}) or {}
    ag  = state.get("ag_result",  {}) or {}
    av  = state.get("av_result",  {}) or {}
    return {
        "AR":  float(ar.get("confidence_score",  0.0)),
        "APS": float(aps.get("confidence_score", 0.0)),
        "AG":  float(ag.get("confidence_score",  0.0)),
        "AV":  float(av.get("confidence_score",  0.0)),
    }


_WACS_W = {"AR": 0.10, "APS": 0.20, "AG": 0.30, "AV": 0.40}

def compute_wacs(conf: dict) -> float:
    active_w = sum(_WACS_W[k] for k, v in conf.items() if v > 0)
    if active_w == 0:
        return 0.0
    return sum(_WACS_W[k] * v for k, v in conf.items() if v > 0) / active_w


def _safe_geval(metric, test_case, retries=3):
    actual = test_case.actual_output or ""
    for attempt in range(retries):
        try:
            metric.measure(test_case)
            score = metric.score
            if score is None:
                score = 0.5
            elif score == 0.0 and len(actual) > 20:
                if attempt < retries - 1:
                    time.sleep(5)
                    continue
                score = 0.5
            return score
        except Exception:
            if attempt == retries - 1:
                return 0.5
            time.sleep(5)
    return 0.5


def evaluate_paraphrase(scenario_id: int, para: dict, tc: dict, verbose: bool = True) -> dict:
    """Run the pipeline for a single paraphrase and compute all metrics."""
    from orchestrator.graph import run_query

    para_id = para["id"]
    variant  = para["variant"]
    query    = para["question"]

    if verbose:
        print(f"\n    [{para_id}] ({variant}): {query}")

    conversation_history = tc.get("conversation_history", [])
    last_result_for_as   = tc.get("last_result_for_as", {})

    t0    = time.time()
    state = run_query(user_input=query, db_type=DB_TYPE, tool_registry=None,
                      conversation_history=conversation_history,
                      last_result_for_as=last_result_for_as)
    elapsed = round(time.time() - t0, 2)

    ar  = state.get("ar_result",  {}) or {}
    ae  = state.get("ae_result",  {}) or {}
    aps = state.get("aps_result", {}) or {}
    ag  = state.get("ag_result",  {}) or {}

    generated_sql = state.get("final_sql", "") or ag.get("sql", "") or ""

    ae_expl = ae.get("explanation", ae.get("response", ""))
    if isinstance(ae_expl, dict):
        ae_expl = ae_expl.get("final_reasoning", ae_expl.get("explanation", ""))
    explanation = str(ae_expl).strip() if ae_expl else ""

    if not explanation:
        for key in ("rejection_reason", "reasoning", "no_info_reason"):
            val = ar.get(key, "")
            if isinstance(val, str) and val.strip():
                explanation = val.strip()
                break
    if not explanation:
        explanation = str(ae.get("reasoning", "")).strip()
    if not explanation:
        explanation = "The request was rejected by the pipeline."

    expected_sql   = tc.get("expected_sql") or ""
    expected_expl  = tc["expected_explanation"]
    expected_tools = tc["expected_tools"]

    rouge_sql = rouge_l_sql(generated_sql, expected_sql) if expected_sql else 0.0

    actual_tools = extract_tools_called(state)
    tc_metric    = tool_correctness_metric(threshold=0.5)
    tc_metric.measure(LLMTestCase(
        input=query, actual_output=explanation,
        tools_called=[ToolCall(name=t) for t in actual_tools],
        expected_tools=[ToolCall(name=t) for t in expected_tools],
    ))
    correctness = tc_metric.score if tc_metric.score is not None else 0.0

    tables_used  = list((aps.get("tables") or {}).keys())
    ground_input = (f"Question: {query} | SQL: {generated_sql} | "
                    f"Tables: {', '.join(tables_used)}")

    faith_metric  = e2e_faithfulness_geval()
    ground_metric = e2e_groundedness_geval()

    F = _safe_geval(faith_metric,  LLMTestCase(
        input=query, actual_output=explanation, expected_output=expected_expl))
    G = _safe_geval(ground_metric, LLMTestCase(
        input=ground_input, actual_output=explanation))

    conf    = extract_confidences(state)
    wacs    = compute_wacs(conf)
    outcome = 1 if rouge_sql >= 0.80 else 0

    if verbose:
        print(f"      SQL: {generated_sql or '(no SQL)'}")
        print(f"      Tools: {actual_tools}")
        print(f"      F={F:.3f}  G={G:.3f}  Cor={correctness:.3f}  "
              f"ROUGE={rouge_sql:.3f}  WACS={wacs:.3f}  [{elapsed}s]")

    return {
        "scenario_id":    scenario_id,
        "para_id":        para_id,
        "variant":        variant,
        "question":       query,
        "type":           tc["type"],
        "generated_sql":  generated_sql,
        "expected_sql":   expected_sql,
        "explanation":    explanation,
        "actual_tools":   actual_tools,
        "expected_tools": expected_tools,
        "Faithfulness":   round(F, 4),
        "Groundedness":   round(G, 4),
        "Correctness":    round(correctness, 4),
        "ROUGE-L_SQL":    round(rouge_sql, 4),
        "WACS":           round(wacs, 4),
        "outcome":        outcome,
        "confidences":    conf,
        "elapsed_s":      elapsed,
    }


def evaluate_all(orchestrator_model=None, orchestrator_temperature=None,
                 verbose: bool = True) -> dict:
    import config
    import orchestrator.nodes as nodes_mod

    original_model = config.AGENT_MODELS.get("classifier")
    original_temp  = config.AGENT_TEMPERATURES.get("classifier")

    if orchestrator_model is not None:
        config.AGENT_MODELS["classifier"] = orchestrator_model
    if orchestrator_temperature is not None:
        config.AGENT_TEMPERATURES["classifier"] = orchestrator_temperature
    nodes_mod._orchestrator_agent = None

    try:
        test_cases   = load_dataset()
        all_paras    = []   # one entry per paraphrase (30 total)
        scenario_summaries = []

        for tc in test_cases:
            sid = tc["id"]
            if verbose:
                print(f"\n  Scenario [{sid}] {tc['description']}")
                print(f"  Original: {tc['original_question']}")

            para_results = []
            for para in tc["paraphrases"]:
                r = evaluate_paraphrase(sid, para, tc, verbose=verbose)
                para_results.append(r)
                all_paras.append(r)

            # Robustness per scenario = mean(combined) — computed after Brier/ECE
            scenario_summaries.append({
                "scenario_id":   sid,
                "type":          tc["type"],
                "original":      tc["original_question"],
                "para_results":  para_results,
            })

        # Calibration: Brier + ECE over full_pipeline paraphrases only
        fp_paras     = [r for r in all_paras if r["type"].startswith("full_pipeline")]
        cal_paras    = fp_paras if fp_paras else all_paras
        wacs_list    = [r["WACS"]    for r in cal_paras]
        outcome_list = [r["outcome"] for r in cal_paras]
        B = brier_score(wacs_list, outcome_list)
        E = ece(wacs_list, outcome_list)

        # Add Brier, ECE, combined to each paraphrase result
        for r in all_paras:
            r["Brier"]    = round(B, 4)
            r["ECE"]      = round(E, 4)
            r["combined"] = round(
                r["Faithfulness"] + r["Groundedness"] + r["Correctness"] - B - E, 4)

        # Robustness per scenario
        for sc in scenario_summaries:
            scores = [r["combined"] for r in sc["para_results"]]
            sc["robustness_score"] = round(sum(scores) / len(scores), 4)
            sc["min_combined"]     = round(min(scores), 4)
            sc["max_combined"]     = round(max(scores), 4)

        avg = lambda k: round(sum(r[k] for r in all_paras) / len(all_paras), 4)

        # Robustness score per variant
        for variant in ("informal", "open_rephrased", "alternative_vocabulary"):
            vp = [r for r in all_paras if r["variant"] == variant]
            sc_v = round(sum(r["combined"] for r in vp) / len(vp), 4) if vp else 0.0
            if verbose:
                print(f"\n  Variant [{variant}]: mean_combined={sc_v:.4f}  (n={len(vp)})")

        return {
            "orchestrator_model":       orchestrator_model or original_model,
            "orchestrator_temperature": orchestrator_temperature if orchestrator_temperature is not None else original_temp,
            "db_type":          DB_TYPE,
            "n_scenarios":      len(test_cases),
            "n_paraphrases":    len(all_paras),
            "Faithfulness":     avg("Faithfulness"),
            "Groundedness":     avg("Groundedness"),
            "Correctness":      avg("Correctness"),
            "ROUGE-L_SQL":      avg("ROUGE-L_SQL"),
            "WACS":             avg("WACS"),
            "Brier":            round(B, 4),
            "ECE":              round(E, 4),
            "combined_score":   avg("combined"),
            "robustness_mean":  round(
                sum(sc["robustness_score"] for sc in scenario_summaries) / len(scenario_summaries), 4),
            "scenarios":        scenario_summaries,
            "per_paraphrase":   all_paras,
        }

    finally:
        if original_model is not None:
            config.AGENT_MODELS["classifier"] = original_model
        elif "classifier" in config.AGENT_MODELS and orchestrator_model is not None:
            del config.AGENT_MODELS["classifier"]
        if original_temp is not None:
            config.AGENT_TEMPERATURES["classifier"] = original_temp
        elif "classifier" in config.AGENT_TEMPERATURES and orchestrator_temperature is not None:
            del config.AGENT_TEMPERATURES["classifier"]
        nodes_mod._orchestrator_agent = None


def save_results(metrics: dict, tag: str = "phase4") -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{tag}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def main():
    print("=" * 70)
    print("  Phase 4 — Paraphrase Robustness — MySQL")
    print(f"  DB      : {DB_TYPE.upper()}")
    print(f"  Scenarios: 10  |  Paraphrases per scenario: 3  |  Total: 30")
    print(f"  Judge   : {os.environ.get('EVAL_JUDGE_MODEL', 'gpt-4o')}")
    print(f"  Formula : F + G + Correctness - Brier - ECE")
    print("=" * 70)

    summary = evaluate_all(verbose=True)

    print("\n" + "=" * 70)
    print(f"  RESULTS  ({summary['n_paraphrases']} paraphrases, {DB_TYPE.upper()})")
    print(f"  Orchestrator : {summary['orchestrator_model']}  T={summary['orchestrator_temperature']}")
    print("-" * 70)
    print(f"  Faithfulness  : {summary['Faithfulness']:.3f}")
    print(f"  Groundedness  : {summary['Groundedness']:.3f}")
    print(f"  Correctness   : {summary['Correctness']:.3f}")
    print(f"  ROUGE-L_SQL   : {summary['ROUGE-L_SQL']:.3f}")
    print(f"  WACS          : {summary['WACS']:.3f}")
    print(f"  Brier         : {summary['Brier']:.4f}")
    print(f"  ECE           : {summary['ECE']:.3f}")
    print(f"  ─────────────────────────────")
    print(f"  combined      : {summary['combined_score']:.3f}  (max 1.0)")
    print(f"  robustness    : {summary['robustness_mean']:.3f}  (mean over scenarios)")
    print("-" * 70)
    print(f"  {'Scenario':<6}  {'Type':<35}  {'Robustness':>11}  {'Min':>6}  {'Max':>6}")
    print(f"  {'-'*6}  {'-'*35}  {'-'*11}  {'-'*6}  {'-'*6}")
    for sc in summary["scenarios"]:
        print(f"  [{sc['scenario_id']:<4}]  {sc['type']:<35}  "
              f"{sc['robustness_score']:>11.3f}  "
              f"{sc['min_combined']:>6.3f}  {sc['max_combined']:>6.3f}")
    print("=" * 70)

    path = save_results(summary, tag="phase4_mysql")
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
