# -*- coding: utf-8 -*-
"""
agents/AV/mysql/phase_2_2/run_phase2.py
Fase 2.2 robustez AV MySQL — lee ganador de phase_1_2/results/.

Uso:
    .\\venv311\\Scripts\\python agents/AV/mysql/phase_2_2/run_phase2.py
"""

import os, sys, json, glob as _glob
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o")

from agents.AV.mysql.phase_2.main import evaluate_av_paraphrases

RESULTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_PHASE1_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase_1_2", "results")


def load_phase1_winner():
    jsons = sorted(_glob.glob(os.path.join(_PHASE1_RESULTS, "*.json")), key=os.path.getmtime, reverse=True)
    for path in jsons:
        try:
            data = json.load(open(path, encoding="utf-8"))
            if "best_trial" in data:
                p = data["best_trial"]["params"]
                print(f"  [winner] {os.path.basename(path)}")
                return p["model"], float(p["temperature"])
        except Exception:
            continue
    raise FileNotFoundError(f"No hay resultado de Fase 1.2 en: {_PHASE1_RESULTS}")


def main():
    model, temp = load_phase1_winner()
    print("=" * 70)
    print(f"  AV (MySQL) Fase 2.2 — modelo={model}  T={temp}")
    print("=" * 70)

    metrics = evaluate_av_paraphrases(model=model, temperature=temp, verbose=True)

    cm = metrics["confusion"]
    print(f"\n  combined={metrics['combined_score']:.3f}  acc={metrics['accuracy']:.3f}  "
          f"F={metrics['Faithfulness']:.3f}  G={metrics['Groundedness']:.3f}  "
          f"TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"phase3_winner_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "temperature": temp, **metrics}, f, indent=2, ensure_ascii=False)
    print(f"  Guardado: {path}")


if __name__ == "__main__":
    main()
