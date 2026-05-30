# -*- coding: utf-8 -*-
"""
agents/AV/mysql/phase_2/run_phase2.py — AV MySQL Fase 2 (paraphrase invariance)
con la config ganadora de Fase 1.

El modelo y temperatura ganadores se leen automaticamente del JSON
mas reciente en phase_1/results/ (clave best_trial.params).

Uso:
    .\\venv311\\Scripts\\python agents/AV/mysql/phase_2/run_phase2.py
"""

import os
import sys
import json
import glob as _glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

from agents.AV.mysql.phase_2.main import evaluate_av_paraphrases, save_results

# ── Directorio de resultados de Fase 1 ───────────────────────────────
_PHASE1_RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_1", "results")


def load_phase1_winner() -> tuple:
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
        f"Ejecuta primero: agents/AV/mysql/phase_1/run_optuna.py"
    )


def main():
    winner_model, winner_temperature = load_phase1_winner()

    print("=" * 70)
    print("  AV (MySQL) Fase 2 - Paraphrase invariance")
    print(f"  Modelo AV     : {winner_model}  (ganador Fase 1, leido automaticamente)")
    print(f"  Temperatura   : {winner_temperature}  (ganador Fase 1)")
    print(f"  disable_db_skills = True (sin run_explain)")
    print(f"  Dataset       : 10 intents x (valid + invalid) x 3 paraphrases = 60 inputs")
    print("=" * 70)

    metrics = evaluate_av_paraphrases(
        model=winner_model,
        temperature=winner_temperature,
        verbose=True,
    )

    cm = metrics["confusion"]
    n  = metrics["n_samples"]

    print("\n" + "=" * 70)
    print(f"  RESULTADO FASE 2  ({n} muestras x 3 expected_reasonings)")
    print("-" * 70)
    print(f"  Faithfulness  : {metrics['Faithfulness']:.3f}")
    print(f"  Groundedness  : {metrics['Groundedness']:.3f}")
    print(f"  ROUGE-L       : {metrics['ROUGE-L']:.3f}")
    print(f"  F1            : {metrics['F1']:.3f}")
    print(f"  Accuracy      : {metrics['accuracy']:.3f}")
    print(f"  Brier Score   : {metrics['Brier']:.4f}")
    print(f"  ECE           : {metrics['ECE']:.3f}")
    print(f"  -------------------------")
    print(f"  combined_score: {metrics['combined_score']:.3f}  (max 1.0)")
    print(f"  Confusion: TP={cm['TP']}  TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}")
    print(f"  avg_time/q    : {metrics['avg_time_s']:.2f}s")
    print("=" * 70)

    path = save_results(metrics, tag="phase3_winner")
    print(f"  Guardado en   : {path}")


if __name__ == "__main__":
    main()
