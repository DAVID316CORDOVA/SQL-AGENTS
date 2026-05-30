# -*- coding: utf-8 -*-
"""
agents/AG/postgres/phase_1.2/run_optuna.py
Segunda corrida del AG PostgreSQL (seed=42, grid completo 16 trials).

Espacio de busqueda:  4 modelos x 4 temperaturas = 16 combos  (100%)
Funcion objetivo:     F + G + ROUGE-L_SQL - Brier - ECE  (max 3.0)

Uso:
    .\\venv311\\Scripts\\python agents/AG/postgres/phase_1.2/run_optuna.py
    .\\venv311\\Scripts\\python agents/AG/postgres/phase_1.2/run_optuna.py --seed 42
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

# Importar solo la funcion de evaluacion (el RESULTS_DIR del main original
# no se usa aqui — guardamos en phase_1.2/results/ directamente)
from agents.AG.postgres.phase_1.main import evaluate_ag

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

    metrics = evaluate_ag(model=model, temperature=temperature, verbose=False)

    print(f"     F={metrics['Faithfulness']:.3f}  G={metrics['Groundedness']:.3f}  "
          f"R={metrics['ROUGE-L_SQL']:.3f}  B={metrics['Brier']:.3f}  "
          f"E={metrics['ECE']:.3f}  => combined={metrics['combined_score']:.3f}")

    _trials_metrics.append({
        "trial_number": trial.number,
        "params": dict(trial.params),
        **{k: v for k, v in metrics.items() if k != "per_question"},
        "per_question": metrics["per_question"],
    })
    return metrics["combined_score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=66)
    args = ap.parse_args()

    print("=" * 70)
    print("  AG (PostgreSQL) Fase 1.2 - Optuna TPE (grid completo)")
    print(f"  Modelos      : {MODELS}")
    print(f"  Temperaturas : {TEMPERATURES}")
    print(f"  Espacio      : {len(MODELS)} x {len(TEMPERATURES)} = {len(MODELS)*len(TEMPERATURES)} combos")
    print(f"  Trials       : {N_TRIALS}  (startup={N_STARTUP}) — 100% del espacio")
    print(f"  Seed         : {args.seed}")
    print(f"  Resultados   : {RESULTS_DIR}")
    print("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed, n_startup_trials=N_STARTUP),
        study_name=f"AG_postgres_phase1.2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 70)
    print("  RESUMEN  AG Postgres Fase 1.2")
    print("=" * 70)
    print(f"  Mejor trial  : #{study.best_trial.number}")
    print(f"  Mejor params : {study.best_trial.params}")
    print(f"  Mejor score  : {study.best_value:.3f}")
    print()
    print(f"  {'#':<3} {'model':<22} {'T':<5} {'F':<6} {'G':<6} {'R':<6} {'B':<6} {'E':<6} {'comb':<6}")
    print("  " + "-" * 66)
    for tm in sorted(_trials_metrics, key=lambda x: -x["combined_score"]):
        p = tm["params"]
        print(f"  {tm['trial_number']:<3} {p['model']:<22} {p['temperature']:<5} "
              f"{tm['Faithfulness']:<6.3f} {tm['Groundedness']:<6.3f} "
              f"{tm['ROUGE-L_SQL']:<6.3f} {tm['Brier']:<6.3f} "
              f"{tm['ECE']:<6.3f} {tm['combined_score']:<6.3f}")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"optuna_tpe_{ts}.json")
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
