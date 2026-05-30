# -*- coding: utf-8 -*-
"""
agents/AR/phase_1/run_phase1.py — Fase 1 del AR (busqueda Bayesiana).

Sampler: TPESampler (Tree-structured Parzen Estimator) — Bayesiano.
Las primeras n_startup_trials son aleatorias (exploracion); luego TPE
modela la distribucion de buenos vs malos trials y enfoca el muestreo
en la region prometedora del espacio.

Espacio de busqueda (2 ejes, valores discretos citables):
    - model        : 6 modelos cross-proveedor (OpenAI, Anthropic, Gemini)
    - temperature  : {0.0, 0.1, 0.3, 0.5, 0.7}
                     0.0  DIN-SQL/DAIL-SQL (determinista, NL2SQL)
                     0.1  analitico conservador
                     0.3  Anthropic docs (tareas analiticas)
                     0.5  equilibrio diversidad/precision
                     0.7  LLaMA 2 / Mistral chat default
    Espacio implicito: 6 x 5 = 30 combinaciones; TPE muestrea 10 (~33%).

Nota: top_p se excluye del espacio porque la API de Anthropic (Claude 4.x)
rechaza el uso simultaneo de temperature y top_p. Para mantener el mismo
espacio de busqueda entre proveedores, se omite top_p en todos.

Funcion objetivo (asesor 2026-05-13):
    combined_score = (2*Faithfulness + ROUGE-L + 2*Groundedness - ECE - Brier) / 5
    direccion: MAXIMIZAR (cota teorica = 1.0)

Uso:
    venv/Scripts/python.exe agents/AR/phase_1/run_phase1.py
    venv/Scripts/python.exe agents/AR/phase_1/run_phase1.py --trials 10
"""

import os
import sys
import json
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o")

import optuna
from optuna.samplers import TPESampler

from agents.AR.phase_1.main import evaluate_ar, RESULTS_DIR
from llm_client import flush_observability


# ─────────────────────────────────────────────────────────────────────
# Espacio de busqueda
# ─────────────────────────────────────────────────────────────────────

MODELS = [
    "gpt-4o",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "gemini-2.5-flash",
]

TEMPERATURES = [0.0, 0.3, 0.5, 0.7]

N_TRIALS         = 10
N_STARTUP_TRIALS = 3


# ─────────────────────────────────────────────────────────────────────
# Funcion objetivo
# ─────────────────────────────────────────────────────────────────────

_trials_metrics: list[dict] = []


def objective(trial: optuna.trial.Trial) -> float:
    model       = trial.suggest_categorical("model",       MODELS)
    temperature = trial.suggest_categorical("temperature", TEMPERATURES)

    print(f"\n  -- Trial #{trial.number} --  model={model}  T={temperature:.1f}")

    metrics = evaluate_ar(
        model=model,
        temperature=temperature,
        verbose=False,
    )

    print(f"     F={metrics['Faithfulness']:.3f}  "
          f"G={metrics['Groundedness']:.3f}  "
          f"R={metrics['ROUGE-L']:.3f}  "
          f"B={metrics['Brier']:.3f}  "
          f"E={metrics['ECE']:.3f}  "
          f"=> combined={metrics['combined_score']:.3f}  "
          f"({metrics['avg_time_s']:.1f}s/q)")

    _trials_metrics.append({
        "trial_number": trial.number,
        "params": dict(trial.params),
        **{k: v for k, v in metrics.items() if k != "per_question"},
        "per_question": metrics["per_question"],
    })

    return metrics["combined_score"]


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",  type=int, default=N_TRIALS,
                    help=f"numero de trials TPE (default {N_TRIALS})")
    ap.add_argument("--startup", type=int, default=N_STARTUP_TRIALS,
                    help=f"trials de arranque aleatorio (default {N_STARTUP_TRIALS})")
    ap.add_argument("--seed",    type=int, default=66)
    args = ap.parse_args()

    print("=" * 70)
    print("  AR Fase 1 - Optuna TPE (cross-proveedor)")
    print(f"  Modelos        : {MODELS}")
    print(f"  Temperaturas   : {TEMPERATURES}")
    print(f"  Espacio total  : {len(MODELS)} x {len(TEMPERATURES)} = "
          f"{len(MODELS)*len(TEMPERATURES)} combinaciones")
    print(f"  Modelo juez    : {os.environ['EVAL_JUDGE_MODEL']}")
    print(f"  Sampler        : TPESampler (seed={args.seed}, "
          f"n_startup={args.startup})")
    print(f"  Trials         : {args.trials}  "
          f"({args.trials*100//(len(MODELS)*len(TEMPERATURES))}% del espacio)")
    print("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed, n_startup_trials=args.startup),
        study_name=f"AR_phase1_TPE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    study.optimize(objective, n_trials=args.trials)

    # ── Resumen ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)
    print(f"  Mejor trial  : #{study.best_trial.number}")
    print(f"  Mejor params : {study.best_trial.params}")
    print(f"  Mejor score  : {study.best_value:.3f}")
    print()
    print("  Todos los trials (ordenados por combined_score):")
    print(f"  {'#':<3} {'model':<22} {'T':<5} "
          f"{'F':<6} {'G':<6} {'R':<6} {'B':<6} {'E':<6} {'comb':<6}")
    print("  " + "-" * 72)
    for tm in sorted(_trials_metrics, key=lambda x: -x["combined_score"]):
        p = tm["params"]
        print(f"  {tm['trial_number']:<3} "
              f"{p['model']:<22} "
              f"{p['temperature']:<5.1f} "
              f"{tm['Faithfulness']:<6.3f} "
              f"{tm['Groundedness']:<6.3f} "
              f"{tm['ROUGE-L']:<6.3f} "
              f"{tm['Brier']:<6.3f} "
              f"{tm['ECE']:<6.3f} "
              f"{tm['combined_score']:<6.3f}")
    print("=" * 70)

    # ── Persistencia ─────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"optuna_tpe_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "study_name":  study.study_name,
            "sampler":     "TPESampler",
            "search_space": {"models": MODELS, "temperatures": TEMPERATURES},
            "n_trials":    args.trials,
            "n_startup":   args.startup,
            "seed":        args.seed,
            "best_trial": {
                "number": study.best_trial.number,
                "params": study.best_trial.params,
                "value":  study.best_value,
            },
            "trials": _trials_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Guardado en  : {path}")
    flush_observability()


if __name__ == "__main__":
    main()
