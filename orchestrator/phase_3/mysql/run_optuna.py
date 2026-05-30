# -*- coding: utf-8 -*-
"""
orchestrator/phase_3/mysql/run_optuna.py

Phase 3 — Hyperparameter search for the Orchestrator (MySQL) with Optuna TPE.

Finds the best model and temperature for the classifier agent while keeping
the winning models of all other agents fixed (AR, APS, AG, AV, AE).

Search space:
    model       : 6 options (GPT-4o-mini, GPT-4o, Haiku, Sonnet, Gemini Flash, Gemini Pro)
    temperature : 5 values [0.0, 0.1, 0.3, 0.5, 0.7]

Objective:
    combined = F + G + Correctness - Brier - ECE   (max 3.0)
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

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from orchestrator.phase_3.mysql.main import evaluate_all, save_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS       = ["gpt-4o", "claude-haiku-4-5",
                "claude-sonnet-4-6", "gemini-2.5-flash"]
TEMPERATURES = [0.0, 0.3, 0.5, 0.7]
N_TRIALS     = 10


def objective(trial: optuna.Trial) -> float:
    model = trial.suggest_categorical("model",       MODELS)
    temp  = trial.suggest_categorical("temperature", TEMPERATURES)

    print(f"\n  Trial {trial.number}: {model}  T={temp}")

    metrics = evaluate_all(
        orchestrator_model=model,
        orchestrator_temperature=temp,
        verbose=True,
    )

    score = metrics["combined_score"]
    print(f"  -> combined = {score:.4f}  "
          f"(F={metrics['Faithfulness']:.3f}  G={metrics['Groundedness']:.3f}  "
          f"Cor={metrics['Correctness']:.3f}  B={metrics['Brier']:.4f}  "
          f"ECE={metrics['ECE']:.3f})")

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"trial_{trial.number:02d}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"trial": trial.number, "model": model, "temperature": temp, **metrics},
                  f, ensure_ascii=False, indent=2)

    return score


def main():
    print("=" * 70)
    print("  Optuna E2E MySQL — SQL-Agents Orchestrator")
    print(f"  DB      : MySQL")
    print(f"  Models  : gemini-2.5-flash / claude-haiku-4-5 / claude-sonnet-4-6 / gpt-4o")
    print(f"  Temps   : {TEMPERATURES}")
    print(f"  Trials  : {N_TRIALS}")
    print(f"  Judge   : {os.environ.get('EVAL_JUDGE_MODEL', 'gpt-4o')}")
    print(f"  Formula : F + G + Correctness - Brier - ECE")
    print("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=66, n_startup_trials=3),
        study_name="e2e_orchestrator_mysql",
    )
    study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))

    best = study.best_trial
    print("\n" + "=" * 70)
    print(f"  WINNER  MySQL — Orchestrator")
    print(f"  Model       : {best.params['model']}")
    print(f"  Temperature : {best.params['temperature']}")
    print(f"  combined    : {best.value:.4f}")
    print("=" * 70)

    winner = {
        "agent":          "Orchestrator",
        "backend":        "mysql",
        "model":          best.params["model"],
        "temperature":    best.params["temperature"],
        "combined_score": best.value,
        "n_trials":       N_TRIALS,
        "generated_at":   datetime.now().isoformat(),
    }
    path = save_results(winner, tag="optuna_winner")
    print(f"  Saved: {path}")

    print("\n  Trial ranking:")
    for t in sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:8]:
        val = f"{t.value:.4f}" if t.value is not None else "FAILED"
        print(f"    [{t.number:02d}] {t.params['model']:<25}  "
              f"T={t.params['temperature']}  combined={val}")


if __name__ == "__main__":
    main()
