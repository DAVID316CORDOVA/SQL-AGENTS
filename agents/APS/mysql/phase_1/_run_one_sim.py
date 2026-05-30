# -*- coding: utf-8 -*-
"""
_run_one_sim.py — evalua APS para UNA similarity y guarda el resultado en JSON.

Diseñado para ser invocado como subproceso desde run_compare.py.
Cada llamada es un proceso Python fresco, lo que garantiza liberacion
de handles de ChromaDB en Windows.

Uso:
    python _run_one_sim.py <similarity> <output_json>
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)


def main():
    if len(sys.argv) != 3:
        print("Uso: _run_one_sim.py <similarity> <output_json>")
        sys.exit(1)
    similarity, out_path = sys.argv[1], sys.argv[2]

    from agents.APS.mysql.phase_1.main import evaluate_aps
    metrics = evaluate_aps(similarity=similarity, verbose=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
