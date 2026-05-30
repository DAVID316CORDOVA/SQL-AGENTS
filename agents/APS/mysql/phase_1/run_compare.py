# -*- coding: utf-8 -*-
"""
agents/APS/mysql/phase_1/run_compare.py

Compara las 3 metricas de similitud {cosine, ip, l2}. Cada similarity
se ejecuta en un SUBPROCESO Python separado para evitar el bug de
ChromaDB en Windows donde los file handles no se liberan entre
re-instanciaciones del PersistentClient.

Uso:
    venv/Scripts/python.exe agents/APS/mysql/phase_1/run_compare.py
"""

import os
import sys
import json
import subprocess
import tempfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

from agents.APS.mysql.phase_1.main import save_results

SIMILARITIES = ["cosine", "ip", "l2"]
BEST_SIMILARITY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "best_similarity.json"
)


def run_one_in_subprocess(sim: str) -> dict:
    """Ejecuta evaluate_aps(sim) en un subproceso fresco. Retorna el dict de metricas."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        out_path = tmp.name

    runner_script = os.path.join(os.path.dirname(__file__), "_run_one_sim.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("ACTIVE_DATASET", "spider:concert_singer")

    print(f"\n  >>> Subproceso similarity={sim!r}")
    result = subprocess.run(
        [sys.executable, runner_script, sim, out_path],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        print(f"  [SUBPROC ERROR] returncode={result.returncode}")
        print(f"  stdout (ultimas 20 lineas):")
        for line in (result.stdout or "").splitlines()[-20:]:
            print(f"    {line}")
        print(f"  stderr (ultimas 20 lineas):")
        for line in (result.stderr or "").splitlines()[-20:]:
            print(f"    {line}")
        raise RuntimeError(f"Subproceso fallo para similarity={sim}")

    # Echo de progreso (verbose del subproceso)
    for line in (result.stdout or "").splitlines():
        if any(tag in line for tag in ["F1_t=", "F1_tables=", "APS: Indexado"]):
            print(f"    {line}")

    with open(out_path, encoding="utf-8") as f:
        metrics = json.load(f)
    os.unlink(out_path)
    return metrics


def main():
    print("=" * 70)
    print("  APS (MySQL) Fase 1 - Comparacion de metricas de similitud")
    print(f"  Similarities probadas : {SIMILARITIES}")
    print(f"  Top-K (tablas y columnas) : 5")
    print(f"  Cada similarity corre en subproceso separado (workaround Windows)")
    print("=" * 70)

    results_per_sim = {}
    for sim in SIMILARITIES:
        metrics = run_one_in_subprocess(sim)
        results_per_sim[sim] = metrics
        print(f"    F1_tables={metrics['F1_tables']:.3f}  "
              f"F1_cols={metrics['F1_columns']:.3f}  "
              f"combined={metrics['combined_score']:.3f}")

    print("\n" + "=" * 70)
    print("  RESUMEN COMPARATIVO")
    print("=" * 70)
    print(f"  {'similarity':<10} {'P_tab':<7} {'R_tab':<7} {'F1_tab':<7} "
          f"{'P_col':<7} {'R_col':<7} {'F1_col':<7} {'comb':<7}")
    print("  " + "-" * 60)
    for sim, m in sorted(results_per_sim.items(), key=lambda x: -x[1]["combined_score"]):
        print(f"  {sim:<10} "
              f"{m['P_tables']:<7.3f} {m['R_tables']:<7.3f} {m['F1_tables']:<7.3f} "
              f"{m['P_columns']:<7.3f} {m['R_columns']:<7.3f} {m['F1_columns']:<7.3f} "
              f"{m['combined_score']:<7.3f}")
    print("=" * 70)

    winner_sim, winner_metrics = max(
        results_per_sim.items(), key=lambda x: x[1]["combined_score"]
    )
    print(f"\n  GANADOR : {winner_sim}  (combined={winner_metrics['combined_score']:.3f})")
    print(f"  Guardando en : {BEST_SIMILARITY_PATH}")

    with open(BEST_SIMILARITY_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "similarity":     winner_sim,
            "combined_score": winner_metrics["combined_score"],
            "F1_tables":      winner_metrics["F1_tables"],
            "F1_columns":     winner_metrics["F1_columns"],
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "dataset":        "spider:concert_singer (11 questions)",
        }, f, indent=2, ensure_ascii=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = save_results({
        "results_per_similarity": results_per_sim,
        "winner": {"similarity": winner_sim, **winner_metrics},
    }, tag=f"compare_{ts}")
    print(f"  Resultados completos : {full_path}")

    # ── Actualizar best_hyperparameters.json ──────────────────────────
    best_hp_path = os.path.join(ROOT, "experiments", "winners", "best_hyperparameters.json")
    try:
        with open(best_hp_path, encoding="utf-8") as f:
            hp = json.load(f)

        from config import EMBEDDING_MODEL
        entry = {
            "role": "APS_mysql", "agent": "APS", "backend": "mysql",
            "dataset": "spider:concert_singer", "available": True,
            "similarity_metric": winner_sim,
            "embedding_model": EMBEDDING_MODEL,
            "F1_tables":      winner_metrics["F1_tables"],
            "F1_columns":     winner_metrics["F1_columns"],
            "combined_score": winner_metrics["combined_score"],
        }
        hp["winners"]["APS_mysql"] = entry

        # Actualizar lookup["APS"] — reemplazar entrada mysql
        lookup_aps = hp.get("lookup", {}).get("APS", [])
        lookup_aps = [e for e in lookup_aps if e.get("backend") != "mysql"]
        lookup_aps.insert(0, {
            "backend": "mysql", "dataset": "spider:concert_singer",
            "similarity_metric": winner_sim,
            "embedding_model": EMBEDDING_MODEL,
            "F1_tables": winner_metrics["F1_tables"],
            "F1_columns": winner_metrics["F1_columns"],
            "combined_score": winner_metrics["combined_score"],
        })
        hp.setdefault("lookup", {})["APS"] = lookup_aps

        with open(best_hp_path, "w", encoding="utf-8") as f:
            json.dump(hp, f, indent=2, ensure_ascii=False)
        print(f"  best_hyperparameters.json actualizado (APS_mysql={winner_sim}  combined={winner_metrics['combined_score']:.3f})")
    except Exception as e:
        print(f"  [WARN] No se pudo actualizar best_hyperparameters.json: {e}")


if __name__ == "__main__":
    main()
