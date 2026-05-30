"""
scripts/eval_bird.py

Evaluacion del sistema NL->SQL contra el benchmark BIRD (PostgreSQL).

Metrica principal: Execution Accuracy (EX)
  - Ejecuta el SQL generado por el sistema
  - Ejecuta el SQL de referencia (gold SQL)
  - Compara los resultados (conjuntos de filas)
  - EX = porcentaje de preguntas donde los resultados coinciden

Uso:
  python scripts/eval_bird.py \\
    --bird_json /ruta/mini_dev_postgresql.json \\
    --bird_path /ruta/mini_dev \\
    --limit 50          (opcional: evaluar solo N preguntas)
    --output resultados_bird.json

Requisitos:
  - PostgreSQL con las BDs de BIRD cargadas (ejecutar setup_bird_postgres.py primero)
  - Variables de entorno: ANTHROPIC_API_KEY

BIRD mini-dev se descarga desde: bird-bench.github.io
"""

import os
import sys
import json
import time
import argparse
import psycopg2
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_bird_json(json_path: str) -> list:
    """Carga el archivo JSON del BIRD mini-dev."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    # BIRD puede ser lista o dict con clave "data"
    if isinstance(data, list):
        return data
    return data.get("data", data.get("questions", []))


def execute_sql(cursor, sql: str) -> tuple:
    """
    Ejecuta una consulta SQL y retorna (rows, error).
    rows es un frozenset de tuplas para comparacion de conjuntos.
    """
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        # Normalizar a frozenset para comparacion sin importar orden
        normalized = frozenset(
            tuple(str(v).strip().lower() if v is not None else "" for v in row)
            for row in rows
        )
        return normalized, None
    except Exception as e:
        return None, str(e)


def get_pg_connection(pg_config: dict, dbname: str):
    """Conecta a la BD de BIRD en PostgreSQL."""
    cfg = pg_config.copy()
    cfg["dbname"] = dbname
    return psycopg2.connect(**cfg)


def run_evaluation(bird_json_path: str, bird_path: str, pg_config: dict,
                   prefix: str = "bird_", limit: int = None,
                   output_path: str = None, verbose: bool = False):
    """
    Ejecuta la evaluacion completa del sistema contra BIRD.
    """
    from orchestrator.graph import run_query

    questions = load_bird_json(bird_json_path)
    if limit:
        questions = questions[:limit]

    total = len(questions)
    print(f"\n{'='*60}")
    print(f"  EVALUACION BIRD - PostgreSQL")
    print(f"  Preguntas: {total} | Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    results = []
    ex_correct = 0    # Execution Accuracy
    ex_errors = 0     # SQL con errores de ejecucion
    sys_errors = 0    # Errores del sistema (no genero SQL)

    for i, item in enumerate(questions, 1):
        question = item.get("question", "")
        gold_sql  = item.get("SQL", item.get("query", ""))
        db_id     = item.get("db_id", "")
        evidence  = item.get("evidence", "")  # contexto adicional de BIRD

        pg_dbname = f"{prefix}{db_id}".lower().replace("-", "_")

        if verbose:
            print(f"[{i:3}/{total}] {db_id}: {question[:60]}")

        # --- Llamar al sistema ---
        start = time.time()
        try:
            # Si hay evidence, agregarlo a la pregunta para dar contexto
            query_input = question
            if evidence:
                query_input = f"{question} (Contexto: {evidence})"

            result = run_query(query_input, db_type="postgres")
            elapsed = time.time() - start
            generated_sql = result.get("final_sql", "")
        except Exception as e:
            elapsed = time.time() - start
            generated_sql = ""
            sys_errors += 1
            if verbose:
                print(f"       ERROR sistema: {e}")

        if not generated_sql:
            sys_errors += 1
            entry = {
                "id": i,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "generated_sql": "",
                "match": False,
                "error": "Sistema no genero SQL",
                "time_s": round(elapsed, 2),
            }
            results.append(entry)
            print(f"  [{i:3}/{total}] X SIN SQL  | {db_id}: {question[:50]}")
            continue

        # --- Ejecutar ambos SQL en PostgreSQL y comparar ---
        try:
            conn = get_pg_connection(pg_config, pg_dbname)
            conn.autocommit = True
            cur = conn.cursor()

            gold_rows, gold_err = execute_sql(cur, gold_sql)
            gen_rows,  gen_err  = execute_sql(cur, generated_sql)

            cur.close()
            conn.close()
        except Exception as e:
            results.append({
                "id": i,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "generated_sql": generated_sql,
                "match": False,
                "error": f"Conexion BD: {e}",
                "time_s": round(elapsed, 2),
            })
            ex_errors += 1
            print(f"  [{i:3}/{total}] X BD ERR   | {db_id}")
            continue

        if gen_err:
            ex_errors += 1
            match = False
            error_msg = f"SQL invalido: {gen_err}"
        elif gold_err:
            match = False
            error_msg = f"Gold SQL fallo: {gold_err}"
        else:
            match = (gen_rows == gold_rows)
            error_msg = None
            if match:
                ex_correct += 1

        status = "OK" if match else "X"
        label  = "MATCH" if match else ("SQL_ERR" if gen_err else "MISMATCH")
        print(f"  [{i:3}/{total}] {status} {label:<9} | {db_id}: {question[:45]}")

        results.append({
            "id": i,
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "gold_sql": gold_sql,
            "generated_sql": generated_sql,
            "match": match,
            "error": error_msg,
            "gold_rows": len(gold_rows) if gold_rows else 0,
            "gen_rows": len(gen_rows) if gen_rows else 0,
            "time_s": round(elapsed, 2),
        })

    # --- Calcular metricas finales ---
    ex_score = ex_correct / total if total > 0 else 0
    valid_sql = total - sys_errors
    exec_acc_valid = ex_correct / valid_sql if valid_sql > 0 else 0

    print(f"\n{'='*60}")
    print(f"  RESULTADOS FINALES")
    print(f"{'='*60}")
    print(f"  Total preguntas:        {total}")
    print(f"  SQL generado:           {total - sys_errors} ({(total-sys_errors)/total:.1%})")
    print(f"  SQL con error ejecucion:{ex_errors}")
    print(f"  Correctos (EX):         {ex_correct}")
    print(f"")
    print(f"  Execution Accuracy (EX):        {ex_score:.1%}  ({ex_correct}/{total})")
    print(f"  EX sobre SQL validos:           {exec_acc_valid:.1%}  ({ex_correct}/{valid_sql})")
    print(f"{'='*60}\n")

    # Analisis por dominio de BD
    from collections import defaultdict
    by_db = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_db[r["db_id"]]["total"] += 1
        if r["match"]:
            by_db[r["db_id"]]["correct"] += 1

    print(f"  Accuracy por base de datos:")
    for db_name, stats in sorted(by_db.items()):
        acc = stats["correct"] / stats["total"]
        print(f"    {db_name:<30} {acc:.1%}  ({stats['correct']}/{stats['total']})")

    # Guardar resultados
    output = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "BIRD mini-dev PostgreSQL",
        "total": total,
        "ex_correct": ex_correct,
        "ex_score": round(ex_score, 4),
        "ex_score_valid": round(exec_acc_valid, 4),
        "sys_errors": sys_errors,
        "exec_errors": ex_errors,
        "by_db": dict(by_db),
        "questions": results,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n  Resultados guardados en: {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evalua el sistema NL->SQL contra BIRD benchmark (PostgreSQL)"
    )
    parser.add_argument("--bird_json", required=True,
                        help="Ruta al archivo mini_dev_postgresql.json de BIRD")
    parser.add_argument("--bird_path", required=True,
                        help="Ruta a la carpeta raiz del BIRD mini-dev")
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int, default=5432)
    parser.add_argument("--user",     default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--prefix",   default="bird_",
                        help="Prefijo de BDs en PostgreSQL (default: bird_)")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Evaluar solo N primeras preguntas (para pruebas rapidas)")
    parser.add_argument("--output",   default="resultados_bird.json",
                        help="Archivo JSON donde guardar resultados")
    parser.add_argument("--verbose",  action="store_true",
                        help="Mostrar detalles de cada pregunta")
    args = parser.parse_args()

    pg_config = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "password": args.password,
    }

    run_evaluation(
        bird_json_path=args.bird_json,
        bird_path=args.bird_path,
        pg_config=pg_config,
        prefix=args.prefix,
        limit=args.limit,
        output_path=args.output,
        verbose=args.verbose,
    )
