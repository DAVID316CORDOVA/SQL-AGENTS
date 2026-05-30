# -*- coding: utf-8 -*-
"""
agents/AR/phase_0/run_all_combos.py

Fase 0 — Corre TODAS las combinaciones (6 modelos × 5 temperaturas = 30)
y guarda CADA resultado inmediatamente en CSV + JSON para poder:

  1. Reanudar si falla el internet o se acaban los tokens
     (las combinaciones ya guardadas se saltan automáticamente).
  2. Luego leer el CSV y generar los gráficos de cada herramienta
     sin necesidad de volver a llamar a los LLMs.

Salidas:
  results/all_combos_YYYYMMDD_HHMMSS.csv   <- todas las combinaciones
  results/all_combos_YYYYMMDD_HHMMSS.json  <- ídem en JSON detallado

Resume:
  Si ya existe un CSV previo, las combinaciones (model, temperature)
  ya guardadas se omiten y se continúa desde la primera pendiente.

Uso:
  python agents/AR/phase_0/run_all_combos.py
  python agents/AR/phase_0/run_all_combos.py --resume  # retoma CSV más reciente
"""

import os
import sys
import csv
import json
import time
import argparse
import glob
from datetime import datetime
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

os.environ.setdefault("ACTIVE_DATASET", "spider:concert_singer")
os.environ.setdefault("EVAL_JUDGE_MODEL", "gpt-4o")

from agents.AR.phase_0.main import evaluate_ar, RESULTS_DIR

# ─────────────────────────────────────────────────────────────────────
MODELS = [
    "gpt-4o-mini", "gpt-4o",
    "claude-haiku-4-5", "claude-sonnet-4-6",
    "gemini-2.5-flash", "gemini-2.5-pro",
]
TEMPERATURES = [0.0, 0.1, 0.3, 0.5, 0.7]

CSV_FIELDS = [
    "model", "temperature",
    "Faithfulness", "Groundedness", "ROUGE-L",
    "Brier", "ECE", "combined_score",
    "avg_time_s", "n_questions",
]


def _load_done(csv_path: str) -> set:
    """Lee el CSV y retorna el set de (model, temp) ya completados."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["model"], float(row["temperature"])))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="Retomar el CSV más reciente en lugar de crear uno nuevo")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    combos = list(product(MODELS, TEMPERATURES))

    # ── Elegir archivo de salida ──────────────────────────────────────
    if args.resume:
        existing = sorted(glob.glob(os.path.join(RESULTS_DIR, "all_combos_*.csv")))
        if existing:
            csv_path  = existing[-1]
            json_path = csv_path.replace(".csv", ".json")
            print(f"  Reanudando desde: {os.path.basename(csv_path)}")
        else:
            print("  No hay CSV previo, iniciando desde cero.")
            args.resume = False

    if not args.resume:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path  = os.path.join(RESULTS_DIR, f"all_combos_{ts}.csv")
        json_path = os.path.join(RESULTS_DIR, f"all_combos_{ts}.json")

    done      = _load_done(csv_path)
    pending   = [(m, t) for m, t in combos if (m, t) not in done]
    completed = [c for c in combos if c in done]

    print("=" * 70)
    print("  AR Fase 0 — GRID COMPLETO (todas las combinaciones)")
    print(f"  Total   : {len(combos)} combos  ({len(combos)*10} evaluaciones)")
    print(f"  Hechos  : {len(completed)}")
    print(f"  Pendientes: {len(pending)}")
    print(f"  CSV     : {csv_path}")
    print("=" * 70)

    if not pending:
        print("  Todo ya completado. Usa generate_charts_*.py para los gráficos.")
        return

    # ── Abrir CSV (append si ya existe) ──────────────────────────────
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    all_results = []  # para el JSON final

    # Cargar resultados ya guardados en el JSON si existe
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            saved = json.load(f)
            all_results = saved.get("results", [])

    # ── Iterar pendientes ─────────────────────────────────────────────
    for i, (model, temp) in enumerate(pending, 1):
        remaining = len(pending) - i + 1
        print(f"\n  [{len(completed)+i:02d}/{len(combos)}] {model}  T={temp}  "
              f"(quedan {remaining-1})")

        t0 = time.time()
        try:
            metrics = evaluate_ar(model=model, temperature=temp, verbose=False)
            elapsed = time.time() - t0

            row = {
                "model":         model,
                "temperature":   temp,
                "Faithfulness":  round(metrics["Faithfulness"],  4),
                "Groundedness":  round(metrics["Groundedness"],  4),
                "ROUGE-L":       round(metrics["ROUGE-L"],       4),
                "Brier":         round(metrics["Brier"],         4),
                "ECE":           round(metrics["ECE"],           4),
                "combined_score":round(metrics["combined_score"],4),
                "avg_time_s":    round(metrics.get("avg_time_s", elapsed/10), 2),
                "n_questions":   metrics.get("n_questions", 10),
            }

            # Guardar en CSV inmediatamente
            writer.writerow(row)
            csv_file.flush()

            # Guardar en JSON con detalle por pregunta
            all_results.append({**row, "per_question": metrics.get("per_question", [])})
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({"n_total": len(combos), "n_done": len(completed)+i,
                           "results": all_results}, jf, indent=2, ensure_ascii=False)

            print(f"    F={row['Faithfulness']:.3f}  G={row['Groundedness']:.3f}  "
                  f"R={row['ROUGE-L']:.3f}  B={row['Brier']:.3f}  "
                  f"E={row['ECE']:.3f}  combined={row['combined_score']:.3f}  "
                  f"({elapsed:.0f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    ERROR ({elapsed:.0f}s): {e}")
            # Guardar error pero NO en el CSV para poder reintentar
            all_results.append({"model": model, "temperature": temp,
                                "error": str(e), "elapsed_s": round(elapsed,1)})
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({"n_total": len(combos), "n_done": len(completed)+i,
                           "results": all_results}, jf, indent=2, ensure_ascii=False)

    csv_file.close()

    # ── Resumen final ─────────────────────────────────────────────────
    done_rows = [r for r in all_results if "combined_score" in r and not r.get("error")]
    if done_rows:
        best = max(done_rows, key=lambda r: r["combined_score"])
        print("\n" + "=" * 70)
        print(f"  COMPLETADO: {len(done_rows)}/{len(combos)} combinaciones")
        print(f"  GANADOR   : {best['model']}  T={best['temperature']}  "
              f"combined={best['combined_score']:.4f}")
        print(f"  CSV       : {csv_path}")
        print(f"  JSON      : {json_path}")
        print()
        print("  Siguiente paso — generar graficos generales:")
        print("    python agents/AR/phase_0/generate_charts_all.py")
        print()
        print("  Graficos por herramienta (en observability/):")
        print("    python agents/AR/phase_0/observability/generate_charts_langsmith.py")
        print("    python agents/AR/phase_0/observability/generate_charts_langfuse.py")
        print("    python agents/AR/phase_0/observability/generate_charts_dspy.py")
        print("    python agents/AR/phase_0/observability/generate_charts_promptfoo.py")
        print("=" * 70)


if __name__ == "__main__":
    main()
