# -*- coding: utf-8 -*-
"""
agents/AS/phase_2/run_phase2.py - AS Fase 2 (paraphrase invariance)
con la config ganadora de Fase 1.

El modelo y temperatura ganadores se leen automaticamente del JSON
mas reciente en phase_1/results/ (clave best_trial.params).

Uso:
    .\\venv311\\Scripts\\python agents/AS/phase_2/run_phase2.py
"""

import os
import sys
import json
import glob as _glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o")

from agents.AS.phase_2.main import evaluate_as_paraphrases, save_results, load_intents

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
        f"Ejecuta primero: agents/AS/phase_1/run_optuna.py"
    )


def main():
    winner_model, winner_temperature = load_phase1_winner()
    intents = load_intents()
    n_para  = sum(len(i["paraphrases"]) for i in intents)

    print("=" * 70)
    print("  AS Fase 2 - Paraphrase invariance")
    print(f"  Modelo AS     : {winner_model}  (ganador Fase 1, leido automaticamente)")
    print(f"  Temperatura   : {winner_temperature}  (ganador Fase 1)")
    print(f"  Juez          : {os.environ['EVAL_JUDGE_MODEL']}")
    print(f"  Dataset       : {len(intents)} intents x 3 paraphrases = {n_para} inputs")
    print(f"  Metricas      : Faithfulness, Groundedness, ROUGE-L, Brier, ECE")
    print(f"  Formula       : (2F + R + 2G - B - E) / 5  (max 1.0)")
    print("=" * 70)

    metrics = evaluate_as_paraphrases(
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
    print(f"  avg_time/q     : {metrics['avg_time_s']:.2f}s")
    print("=" * 70)

    path = save_results(metrics, tag="phase2_winner")
    print(f"  Guardado en   : {path}")

    print("\n  Por intent (F_mean / G_mean / R_mean):")
    for pi in metrics["per_intent"]:
        print(f"    #{pi['intent_idx']:<3} F={pi['F_mean']:.3f}  "
              f"G={pi['G_mean']:.3f}  R={pi['R_mean']:.3f}")


if __name__ == "__main__":
    main()
