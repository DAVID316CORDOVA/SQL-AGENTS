# -*- coding: utf-8 -*-
"""
orchestrator/phase_3_2/mysql/run_optuna.py
Segunda corrida del orquestador MySQL — seed=66, 10 trials.

Usa:
  - AG/AV mysql ganadores de phase_1_2 (prompts limpios + normalize_sql_v2)
  - rouge_l_sql_v2 (con correccion de orden de JOINs)
  - Umbral outcome = ROUGE >= 0.70
  - Prompt del orquestador corregido (Bug1 AS routing + Bug3 bypass opinion)

Uso:
    .\\venv311\\Scripts\\python orchestrator/phase_3_2/mysql/run_optuna.py
"""

import os, sys, json
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

# Cargar el main.py LOCAL (usa rouge_l_sql_v2 + threshold 0.70)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "phase3_2_mysql_main",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
)
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
evaluate_all = _m.evaluate_all

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
_m.RESULTS_DIR = RESULTS_DIR

MODELS       = ["gpt-4o", "claude-haiku-4-5", "claude-sonnet-4-6", "gemini-2.5-flash"]
TEMPERATURES = [0.0, 0.3, 0.5, 0.7]
N_TRIALS     = 10


def objective(trial: optuna.Trial) -> float:
    model = trial.suggest_categorical("model",       MODELS)
    temp  = trial.suggest_categorical("temperature", TEMPERATURES)
    print(f"\n  Trial {trial.number}: {model}  T={temp}")

    metrics = evaluate_all(orchestrator_model=model, orchestrator_temperature=temp, verbose=True)
    score   = metrics["combined_score"]

    print(f"  -> combined={score:.4f}  "
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
    print("  Optuna E2E MySQL — Orquestador Phase 3.2")
    print(f"  Models  : {MODELS}")
    print(f"  Temps   : {TEMPERATURES}")
    print(f"  Trials  : {N_TRIALS}  seed=66")
    print(f"  ROUGE   : v2 + threshold 0.70")
    print(f"  Prompt  : Bug1(AS)+Bug3(bypass) corregidos")
    print("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=66, n_startup_trials=3),
        study_name="e2e_orchestrator_mysql_3.2",
    )
    study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))

    best = study.best_trial
    print(f"\n  WINNER MySQL Phase 3.2: {best.params['model']}  T={best.params['temperature']}  combined={best.value:.4f}")

    winner = {
        "agent": "Orchestrator", "backend": "mysql",
        "model": best.params["model"], "temperature": best.params["temperature"],
        "combined_score": best.value, "n_trials": N_TRIALS,
        "generated_at": datetime.now().isoformat(),
    }
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"optuna_winner_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(winner, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path}")

    print("\n  Trial ranking:")
    for t in sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:10]:
        val = f"{t.value:.4f}" if t.value is not None else "FAILED"
        print(f"    [{t.number:02d}] {t.params['model']:<25}  T={t.params['temperature']}  combined={val}")


if __name__ == "__main__":
    main()
