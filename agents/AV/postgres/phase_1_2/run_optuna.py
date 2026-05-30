# -*- coding: utf-8 -*-
"""
agents/AV/postgres/phase_1.2/run_optuna.py
Segunda corrida del AV PostgreSQL (seed=42, grid completo 16 trials).

Espacio de busqueda:  4 modelos x 4 temperaturas = 16 combos  (100%)
Funcion objetivo:     (2F + R + 2G - E - B) / 5   (max 1.0)

Uso:
    .\\venv311\\Scripts\\python agents/AV/postgres/phase_1.2/run_optuna.py
    .\\venv311\\Scripts\\python agents/AV/postgres/phase_1.2/run_optuna.py --seed 42
"""

import os
import sys
import json
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o")

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from agents.AV.postgres.phase_1.main import evaluate_av

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

MODELS       = ["gpt-4o", "claude-haiku-4-5", "claude-sonnet-4-6", "gemini-2.5-flash"]
TEMPERATURES = [0.0, 0.3, 0.5, 0.7]
N_TRIALS     = 10    # mismo que Fase 1 para comparacion directa
N_STARTUP    = 3

_trials_metrics: list[dict] = []


def objective(trial: optuna.trial.Trial) -> float:
    model       = trial.suggest_categorical("model",       MODELS)
    temperature = trial.suggest_categorical("temperature", TEMPERATURES)

    print(f"\n  -- Trial #{trial.number} --  model={model}  T={temperature}")

    metrics = evaluate_av(model=model, temperature=temperature, verbose=True)

    print(f"     F={metrics['Faithfulness']:.3f}  G={metrics['Groundedness']:.3f}  "
          f"R={metrics['ROUGE-L']:.3f}  B={metrics['Brier']:.3f}  "
          f"E={metrics['ECE']:.3f}  acc={metrics['accuracy']:.3f}  "
          f"=> combined={metrics['combined_score']:.3f}")

    _trials_metrics.append({
        "trial_number": trial.number,
        "params": dict(trial.params),
        **{k: v for k, v in metrics.items() if k not in ("per_sample", "confusion")},
        "per_sample": metrics["per_sample"],
    })
    return metrics["combined_score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=66)
    args = ap.parse_args()

    print("=" * 70)
    print("  AV (PostgreSQL) Fase 1.2 - TPESampler (grid completo)")
    print(f"  Modelos      : {MODELS}")
    print(f"  Temperaturas : {TEMPERATURES}")
    print(f"  N trials     : {N_TRIALS}  (100% del espacio 4x4)")
    print(f"  Seed         : {args.seed}")
    print(f"  Juez         : {os.environ['EVAL_JUDGE_MODEL']}")
    print(f"  Resultados   : {RESULTS_DIR}")
    print("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed, n_startup_trials=N_STARTUP),
        study_name=f"AV_postgres_phase1.2_TPE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 70)
    print("  RESUMEN  AV Postgres Fase 1.2")
    print("=" * 70)
    print(f"  Mejor trial  : #{study.best_trial.number}")
    print(f"  Mejor params : {study.best_trial.params}")
    print(f"  Mejor score  : {study.best_value:.3f}")
    print()
    print(f"  {'#':<3} {'model':<22} {'T':<5} {'F':<6} {'G':<6} {'R':<6} {'acc':<6} {'B':<6} {'E':<6} {'comb':<6}")
    print("  " + "-" * 72)
    for tm in sorted(_trials_metrics, key=lambda x: -x["combined_score"]):
        p = tm["params"]
        print(f"  {tm['trial_number']:<3} {p['model']:<22} {p['temperature']:<5} "
              f"{tm['Faithfulness']:<6.3f} {tm['Groundedness']:<6.3f} "
              f"{tm['ROUGE-L']:<6.3f} {tm['accuracy']:<6.3f} "
              f"{tm['Brier']:<6.3f} {tm['ECE']:<6.3f} "
              f"{tm['combined_score']:<6.3f}")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"tpe_seed{args.seed}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "study_name":   study.study_name,
            "sampler":      "TPESampler",
            "search_space": {"models": MODELS, "temperatures": TEMPERATURES},
            "n_trials":     N_TRIALS,
            "seed":         args.seed,
            "best_trial": {
                "number": study.best_trial.number,
                "params": study.best_trial.params,
                "value":  study.best_value,
            },
            "trials": _trials_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Guardado en  : {path}")


if __name__ == "__main__":
    main()
