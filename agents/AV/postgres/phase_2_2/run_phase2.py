# -*- coding: utf-8 -*-
"""
agents/AV/postgres/phase_2_2/run_phase2.py
Fase 2.2 de robustez (paraphrase invariance) para AV PostgreSQL.
Lee el ganador de phase_1_2/results/ y guarda en phase_2_2/results/.

Uso:
    .\\venv311\\Scripts\\python agents/AV/postgres/phase_2_2/run_phase2.py
"""

import os
import sys
import json
import glob as _glob
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o")

from agents.AV.postgres.phase_2.main import evaluate_av_paraphrases

RESULTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_PHASE1_RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "phase_1_2", "results"
)


def load_phase1_winner() -> tuple:
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
        f"No se encontro resultado de Fase 1.2 con 'best_trial' en:\n  {_PHASE1_RESULTS}\n"
        f"Ejecuta primero: agents/AV/postgres/phase_1_2/run_optuna.py"
    )


def main():
    winner_model, winner_temperature = load_phase1_winner()

    print("=" * 70)
    print("  AV (PostgreSQL) Fase 2.2 - Paraphrase invariance")
    print(f"  Modelo AV     : {winner_model}  (ganador Fase 1.2)")
    print(f"  Temperatura   : {winner_temperature}")
    print(f"  disable_db_skills = True (sin run_explain)")
    print(f"  Dataset       : 10 intents x (valid + invalid) x 3 paraphrases = 60 inputs")
    print(f"  Resultados    : {RESULTS_DIR}")
    print("=" * 70)

    metrics = evaluate_av_paraphrases(
        model=winner_model,
        temperature=winner_temperature,
        verbose=True,
    )

    cm = metrics["confusion"]
    n  = metrics["n_samples"]

    print("\n" + "=" * 70)
    print(f"  RESULTADO FASE 2.2  ({n} muestras x 3 expected_reasonings)")
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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"phase3_winner_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": winner_model, "temperature": winner_temperature, **metrics},
                  f, indent=2, ensure_ascii=False)
    print(f"  Guardado en   : {path}")


if __name__ == "__main__":
    main()
