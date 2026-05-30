# -*- coding: utf-8 -*-
"""
orchestrator/phase_4_2/mysql/run_phase4.py
Fase 4.2 robustez MySQL — lee ganador de phase_3_2/mysql/results/.

Uso:
    .\\venv311\\Scripts\\python orchestrator/phase_4_2/mysql/run_phase4.py
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

# Cargar main.py LOCAL (rouge_l_sql_v2 + threshold 0.70)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "phase4_2_mysql_main",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
)
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
evaluate_all = _m.evaluate_all

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
_m.RESULTS_DIR = RESULTS_DIR

PHASE3_RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "phase_3_2", "mysql", "results"
))


def load_best_phase3() -> dict:
    winner_files = []
    if os.path.isdir(PHASE3_RESULTS_DIR):
        for fn in os.listdir(PHASE3_RESULTS_DIR):
            if fn.startswith("optuna_winner") and fn.endswith(".json"):
                winner_files.append(os.path.join(PHASE3_RESULTS_DIR, fn))
    if winner_files:
        winner_files.sort(reverse=True)
        with open(winner_files[0], encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [winner] {os.path.basename(winner_files[0])}")
        return {"model": data.get("model", "gpt-4o"), "temperature": data.get("temperature", 0.3)}
    print("  [WARN] No optuna_winner en phase_3_2/mysql. Usando gpt-4o T=0.3")
    return {"model": "gpt-4o", "temperature": 0.3}


def main():
    best = load_best_phase3()
    print("=" * 70)
    print(f"  Phase 4.2 Robustez MySQL — {best['model']}  T={best['temperature']}")
    print(f"  Scenarios: 10  |  Paraphrases: 3 each  |  Total: 30")
    print(f"  ROUGE: v2 + threshold 0.70  |  Prompt corregido (Bug1+Bug3)")
    print("=" * 70)

    summary = evaluate_all(
        orchestrator_model=best["model"],
        orchestrator_temperature=best["temperature"],
        verbose=True,
    )

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"phase4_2_mysql_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved: {path}")
    print(f"  Combined={summary.get('combined_score',0):.3f}  "
          f"Brier={summary.get('Brier',0):.4f}  "
          f"Correctness={summary.get('Correctness',0):.3f}")

    for variant in ("informal", "open_rephrased", "alternative_vocabulary"):
        vp  = [r for r in summary.get("per_paraphrase",[]) if r.get("variant") == variant]
        avg = round(sum(r["combined"] for r in vp)/len(vp),4) if vp else 0.0
        print(f"  {variant:<30}: {avg:.3f}")


if __name__ == "__main__":
    main()
