# -*- coding: utf-8 -*-
"""
orchestrator/phase_4_2/postgres/run_phase4.py

Fase 4.2 de robustez (paraphrase) para el orquestador PostgreSQL.
Lee el ganador de phase_3_2/postgres/results/ y corre la evaluacion.

Checkpoint/resume: si hay un checkpoint incompleto en results/, retoma
automaticamente desde donde quedo sin repetir parafraseos ya completados.

Uso:
    .\\venv311\\Scripts\\python orchestrator/phase_4_2/postgres/run_phase4.py
"""

import os
import sys
import json
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

# Cargamos el main.py LOCAL de esta fase (phase_4_2).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "phase4_2_pg_main",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
)
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
evaluate_all = _m.evaluate_all
save_results = _m.save_results

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
_m.RESULTS_DIR = RESULTS_DIR

PHASE3_RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "phase_3_2", "postgres", "results",
))


def load_best_phase3() -> dict:
    """Lee el ganador del optuna de phase_3_2/postgres/results/."""
    winner_files = []
    if os.path.isdir(PHASE3_RESULTS_DIR):
        for fn in os.listdir(PHASE3_RESULTS_DIR):
            if fn.startswith("optuna_winner") and fn.endswith(".json"):
                winner_files.append(os.path.join(PHASE3_RESULTS_DIR, fn))

    if winner_files:
        winner_files.sort(reverse=True)
        with open(winner_files[0], encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [winner] leido de: {os.path.basename(winner_files[0])}")
        return {
            "model":       data.get("model",       "gpt-4o"),
            "temperature": data.get("temperature", 0.3),
        }

    print("  [WARN] No se encontro optuna_winner en phase_3_2. Usando: gpt-4o T=0.3")
    return {"model": "gpt-4o", "temperature": 0.3}


def main():
    best = load_best_phase3()

    print("=" * 70)
    print("  Phase 4.2 Robustez — PostgreSQL")
    print(f"  Orquestador  : {best['model']}  T={best['temperature']}")
    print(f"  Scenarios    : 10  |  Paraphrases: 3 each  |  Total: 30")
    print(f"  Judge        : {os.environ.get('EVAL_JUDGE_MODEL', 'gpt-4o')}")
    print(f"  Formula      : F + G + Correctness - Brier - ECE")
    print(f"  Results      : {RESULTS_DIR}")
    print("=" * 70)

    summary = evaluate_all(
        orchestrator_model=best["model"],
        orchestrator_temperature=best["temperature"],
        verbose=True,
    )

    # Save final JSON with timestamp
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"phase4_2_postgres_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {path}")

    print("\n" + "=" * 70)
    print(f"  PHASE 4.2 RESULTS — PostgreSQL  ({summary['n_paraphrases']} paraphrases)")
    print(f"  Orquestador  : {summary['orchestrator_model']}  T={summary['orchestrator_temperature']}")
    print(f"  Run ID       : {summary.get('run_id', 'N/A')}")
    print("-" * 70)
    print(f"  Faithfulness  : {summary['Faithfulness']:.3f}")
    print(f"  Groundedness  : {summary['Groundedness']:.3f}")
    print(f"  Correctness   : {summary['Correctness']:.3f}")
    print(f"  ROUGE-L_SQL   : {summary['ROUGE-L_SQL']:.3f}")
    print(f"  WACS          : {summary['WACS']:.3f}")
    print(f"  Brier         : {summary['Brier']:.4f}")
    print(f"  ECE           : {summary['ECE']:.3f}")
    print(f"  -------------------------------------")
    print(f"  combined      : {summary['combined_score']:.3f}")
    print(f"  robustness    : {summary['robustness_mean']:.3f}")
    print("-" * 70)
    print(f"  {'Scen':>4}  {'Type':<35}  {'Robust':>7}  {'Min':>6}  {'Max':>6}")
    print(f"  {'-'*4}  {'-'*35}  {'-'*7}  {'-'*6}  {'-'*6}")
    for sc in summary["scenarios"]:
        print(f"  [{sc['scenario_id']:<3}]  {sc['type']:<35}  "
              f"{sc['robustness_score']:>7.3f}  "
              f"{sc['min_combined']:>6.3f}  {sc['max_combined']:>6.3f}")

    all_p = summary["per_paraphrase"]
    print("-" * 70)
    print("  Breakdown por variante:")
    for variant in ("informal", "open_rephrased", "alternative_vocabulary"):
        vp  = [r for r in all_p if r["variant"] == variant]
        avg = round(sum(r["combined"] for r in vp) / len(vp), 4) if vp else 0.0
        print(f"    {variant:<30} : {avg:.3f}  (n={len(vp)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
