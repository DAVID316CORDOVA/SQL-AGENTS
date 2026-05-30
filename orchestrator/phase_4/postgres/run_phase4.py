# -*- coding: utf-8 -*-
"""
orchestrator/phase_4/postgres/run_phase4.py

Phase 4 — Paraphrase robustness runner for PostgreSQL.
Identical to mysql/run_phase4.py except it uses the postgres main and phase 3 results.

Usage:
    python -m orchestrator.phase_4.postgres.run_phase4
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

from orchestrator.phase_4.postgres.main import evaluate_all, save_results

PHASE3_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "phase_3", "postgres", "results"
)


def load_best_phase3() -> dict:
    """Load the optuna winner from phase 3 (postgres). Falls back to defaults."""
    winner_files = []
    results_dir  = os.path.normpath(PHASE3_RESULTS_DIR)
    if os.path.isdir(results_dir):
        for fn in os.listdir(results_dir):
            if fn.startswith("optuna_winner") and fn.endswith(".json"):
                winner_files.append(os.path.join(results_dir, fn))

    if winner_files:
        winner_files.sort(reverse=True)   # latest first
        with open(winner_files[0], encoding="utf-8") as f:
            data = json.load(f)
        return {
            "model":       data.get("model",       "gpt-4o-mini"),
            "temperature": data.get("temperature", 0.0),
        }

    print("  [WARN] No phase 3 optuna_winner found. Using defaults: gpt-4o-mini, T=0.0")
    return {"model": "gpt-4o-mini", "temperature": 0.0}


def main():
    best = load_best_phase3()

    print("=" * 70)
    print("  Phase 4 Robustness Runner — PostgreSQL")
    print(f"  Orchestrator : {best['model']}  T={best['temperature']}")
    print(f"  Scenarios    : 10  |  Paraphrases: 3 each  |  Total: 30")
    print(f"  Judge        : {os.environ.get('EVAL_JUDGE_MODEL', 'gpt-4o')}")
    print(f"  Formula      : F + G + Correctness - Brier - ECE")
    print("=" * 70)

    summary = evaluate_all(
        orchestrator_model=best["model"],
        orchestrator_temperature=best["temperature"],
        verbose=True,
    )

    path = save_results(summary, tag="phase4_runner_postgres")
    print(f"\n  Saved: {path}")

    print("\n" + "=" * 70)
    print(f"  PHASE 4 RESULTS — PostgreSQL  ({summary['n_paraphrases']} paraphrases)")
    print(f"  Orchestrator : {summary['orchestrator_model']}  T={summary['orchestrator_temperature']}")
    print("-" * 70)
    print(f"  Faithfulness  : {summary['Faithfulness']:.3f}")
    print(f"  Groundedness  : {summary['Groundedness']:.3f}")
    print(f"  Correctness   : {summary['Correctness']:.3f}")
    print(f"  ROUGE-L_SQL   : {summary['ROUGE-L_SQL']:.3f}")
    print(f"  WACS          : {summary['WACS']:.3f}")
    print(f"  Brier         : {summary['Brier']:.4f}")
    print(f"  ECE           : {summary['ECE']:.3f}")
    print(f"  -----------------------------")
    print(f"  combined      : {summary['combined_score']:.3f}")
    print(f"  robustness    : {summary['robustness_mean']:.3f}")
    print("-" * 70)
    print(f"  {'Scen':>4}  {'Type':<35}  {'Robust':>7}  {'Min':>6}  {'Max':>6}")
    print(f"  {'-'*4}  {'-'*35}  {'-'*7}  {'-'*6}  {'-'*6}")
    for sc in summary["scenarios"]:
        print(f"  [{sc['scenario_id']:<3}]  {sc['type']:<35}  "
              f"{sc['robustness_score']:>7.3f}  "
              f"{sc['min_combined']:>6.3f}  {sc['max_combined']:>6.3f}")

    # Variant breakdown
    all_p = summary["per_paraphrase"]
    print("-" * 70)
    print("  Breakdown by paraphrase variant:")
    for variant in ("informal", "open_rephrased", "alternative_vocabulary"):
        vp  = [r for r in all_p if r["variant"] == variant]
        avg = round(sum(r["combined"] for r in vp) / len(vp), 4) if vp else 0.0
        print(f"    {variant:<30} : {avg:.3f}  (n={len(vp)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
