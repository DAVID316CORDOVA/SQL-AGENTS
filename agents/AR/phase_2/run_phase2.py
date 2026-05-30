# -*- coding: utf-8 -*-
"""
agents/AR/phase_2/run_phase2.py — Ejecuta Fase 2 (paraphrase invariance) con la
config ganadora de Fase 1.

Lee el modelo y temperatura ganadores de:
    experiments/winners/best_hyperparameters.json  →  winners.AR.model / .temperature

Al terminar, actualiza winners.AR.phase2 con las metricas obtenidas.

Uso:
    $env:ACTIVE_DATASET="spider:concert_singer"; $env:EVAL_JUDGE_MODEL="gpt-4o"
    python agents\\AR\\phase_2\\run_phase2.py
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o")

from agents.AR.phase_2.main import evaluate_ar_paraphrases, save_results

# ─────────────────────────────────────────────────────────────────────
# Directorio de resultados de Fase 1
# ─────────────────────────────────────────────────────────────────────
import glob as _glob

_PHASE1_RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_1", "results")

N_INTENTS = 12  # 11 SELECT validos + 1 DML invalido = 12 x 3 = 36 paraphrases


def load_winner() -> tuple:
    """
    Lee el ganador de Fase 1 del JSON mas reciente en phase_1/results/
    que contenga la clave 'best_trial'. Devuelve (model, temperature).
    """
    jsons = sorted(
        _glob.glob(os.path.join(_PHASE1_RESULTS, "*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    for path in jsons:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if "best_trial" in data:
                p = data["best_trial"]["params"]
                print(f"  [winner] leido de: {os.path.basename(path)}")
                return p["model"], float(p["temperature"])
        except Exception:
            continue
    raise FileNotFoundError(
        f"No se encontro ningun resultado de Fase 1 con 'best_trial' en:\n  {_PHASE1_RESULTS}\n"
        f"Ejecuta primero: agents/AR/phase_1/run_optuna.py"
    )


def main():
    from agents.AR.phase_2.main import load_intents

    winner_model, winner_temperature = load_winner()

    intents = load_intents()[:N_INTENTS]
    n_para  = sum(len(i["paraphrases"]) for i in intents)

    print("=" * 70)
    print("  AR Fase 2 — Paraphrase invariance")
    print(f"  Modelo AR     : {winner_model}  (ganador Fase 1, leido automaticamente)")
    print(f"  Temperatura   : {winner_temperature}  (ganador Fase 1)")
    print(f"  Modelo juez   : {os.environ['EVAL_JUDGE_MODEL']}")
    print(f"  Dataset       : {N_INTENTS} intents x 3 paraphrases = {n_para} inputs")
    print(f"  Metricas      : Faithfulness, Groundedness, ROUGE-L, Brier, ECE")
    print("=" * 70)

    metrics = evaluate_ar_paraphrases(
        model=winner_model,
        temperature=winner_temperature,
        intents=intents,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print(f"  RESULTADO FASE 2  ({metrics['n_paraphrases']} paraphrases, "
          f"{metrics['n_intents']} intents)")
    print("-" * 70)
    print(f"  Faithfulness   : {metrics['Faithfulness']:.3f}")
    print(f"  Groundedness   : {metrics['Groundedness']:.3f}")
    print(f"  ROUGE-L        : {metrics['ROUGE-L']:.3f}")
    print(f"  Brier Score    : {metrics['Brier']:.4f}")
    print(f"  ECE            : {metrics['ECE']:.3f}")
    print(f"  -------------------------")
    print(f"  combined_score : {metrics['combined_score']:.3f}  (max 1.0)")
    print(f"  avg_time/paraphrase : {metrics['avg_time_s']:.2f}s")
    print("=" * 70)

    path = save_results(metrics, tag="phase2_winner")
    print(f"  Guardado en   : {path}")


if __name__ == "__main__":
    main()
