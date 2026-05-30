# -*- coding: utf-8 -*-
"""
agents/AS/phase_1/run_optuna.py — Optuna TPE for the AS.

Search space: 4 models x 4 temperatures = 16 combos. TPE samples 10 (~63%).
Same setup as AE and AR: TPESampler(seed=66), N_TRIALS=10.

Formula: (2F + R + 2G - B - E) / 5   (max 1.0)
Outcome Brier/ECE: G >= 0.80

Usage:
    .\\venv311\\Scripts\\python agents/AS/phase_1/run_optuna.py
    .\\venv311\\Scripts\\python agents/AS/phase_1/run_optuna.py --seed 42
"""

import os
import sys
import json
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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

from agents.AS.phase_1.main import evaluate_as, RESULTS_DIR

# ─────────────────────────────────────────────────────────────────────
# Search space — identical to AE and AR
# ─────────────────────────────────────────────────────────────────────
MODELS = [
    "gpt-4o",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "gemini-2.5-flash",
]
TEMPERATURES = [0.0, 0.3, 0.5, 0.7]
N_TRIALS = 10

_trials_metrics: list[dict] = []


# ─────────────────────────────────────────────────────────────────────
# Objective function
# ─────────────────────────────────────────────────────────────────────

def objective(trial: optuna.trial.Trial) -> float:
    model       = trial.suggest_categorical("model",       MODELS)
    temperature = trial.suggest_categorical("temperature", TEMPERATURES)

    print(f"\n  -- Trial #{trial.number} --  model={model}  T={temperature:.1f}")

    metrics = evaluate_as(model=model, temperature=temperature, verbose=False)

    print(f"     F={metrics['Faithfulness']:.3f}  G={metrics['Groundedness']:.3f}  "
          f"R={metrics['ROUGE-L']:.3f}  B={metrics['Brier']:.3f}  "
          f"E={metrics['ECE']:.3f}  => combined={metrics['combined_score']:.3f}  "
          f"({metrics['avg_time_s']:.1f}s/q)")

    _trials_metrics.append({
        "trial_number": trial.number,
        "params":       dict(trial.params),
        **{k: v for k, v in metrics.items() if k != "per_sample"},
        "per_sample":   metrics["per_sample"],
    })
    return metrics["combined_score"]


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=66)
    args = ap.parse_args()

    print("=" * 70)
    print("  AS Fase 1 - Optuna TPESampler")
    print(f"  Modelos      : {MODELS}")
    print(f"  Temperaturas : {TEMPERATURES}")
    print(f"  N trials     : {N_TRIALS}")
    print(f"  Seed         : {args.seed}")
    print(f"  Juez         : {os.environ['EVAL_JUDGE_MODEL']}")
    print(f"  Sampler      : TPESampler")
    print("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        study_name=f"AS_phase1_TPE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    study.optimize(objective, n_trials=N_TRIALS)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)
    print(f"  Mejor trial  : #{study.best_trial.number}")
    print(f"  Mejor params : {study.best_trial.params}")
    print(f"  Mejor score  : {study.best_value:.3f}")
    print()
    print(f"  {'#':<3} {'model':<22} {'T':<5} {'F':<6} {'G':<6} {'R':<6} {'B':<6} {'E':<6} {'comb':<6}")
    print("  " + "-" * 66)
    for tm in sorted(_trials_metrics, key=lambda x: -x["combined_score"]):
        p = tm["params"]
        print(f"  {tm['trial_number']:<3} {p['model']:<22} {p['temperature']:<5.1f} "
              f"{tm['Faithfulness']:<6.3f} {tm['Groundedness']:<6.3f} "
              f"{tm['ROUGE-L']:<6.3f} {tm['Brier']:<6.3f} "
              f"{tm['ECE']:<6.3f} {tm['combined_score']:<6.3f}")
    print("=" * 70)

    # ── Save results ─────────────────────────────────────────────────
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
    print(f"  Guardado en  : {path}")

    # ── Update best_hyperparameters.json ─────────────────────────────
    best_params  = study.best_trial.params
    best_metrics = next(
        m for m in _trials_metrics if m["trial_number"] == study.best_trial.number
    )
    hp_path = os.path.join(ROOT, "agents", "AS", "best_hyperparameters.json")
    try:
        with open(hp_path, encoding="utf-8") as f:
            hp = json.load(f)
    except Exception:
        hp = {}

    hp["model"]          = best_params["model"]
    hp["temperature"]    = best_params["temperature"]
    hp["combined_score"] = best_metrics["combined_score"]
    hp["Faithfulness"]   = best_metrics["Faithfulness"]
    hp["Groundedness"]   = best_metrics["Groundedness"]
    hp["ROUGE-L"]        = best_metrics["ROUGE-L"]
    hp["Brier"]          = best_metrics["Brier"]
    hp["ECE"]            = best_metrics["ECE"]
    hp["note"] = (
        f"fase1 optuna TPE seed={args.seed} — ganador "
        f"{best_params['model']} T={best_params['temperature']}"
    )

    with open(hp_path, "w", encoding="utf-8") as f:
        json.dump(hp, f, indent=2, ensure_ascii=False)
    print(f"  Hiperparametros guardados en: {hp_path}")


if __name__ == "__main__":
    main()
