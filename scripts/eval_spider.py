"""
scripts/eval_spider.py

Evaluacion del sistema NL->SQL contra Spider benchmark.
Soporta MySQL y PostgreSQL.

Metrica principal: Execution Accuracy (EX)
  - Ejecuta el SQL generado por el sistema
  - Ejecuta el gold SQL
  - Compara resultados como conjuntos de filas (sin importar orden)

Uso:
  python scripts/eval_spider.py --engine mysql   --limit 10
  python scripts/eval_spider.py --engine postgres --limit 10
  python scripts/eval_spider.py --engine mysql   --limit 10 --questions metadata/spider/experiment_questions.json
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────
# Conexion a BD
# ──────────────────────────────────────────────

def get_mysql_conn(db_name: str, host="localhost", port=3306, user="root", password="123"):
    import mysql.connector
    return mysql.connector.connect(
        host=host, port=port, user=user, password=password, database=db_name
    )

def get_pg_conn(db_name: str, host="localhost", port=5433, user="postgres", password="123"):
    import psycopg2
    return psycopg2.connect(host=host, port=port, user=user, password=password, dbname=db_name)


def execute_sql(cursor, sql: str, engine: str) -> tuple:
    """Ejecuta SQL y retorna (frozenset_rows, error)."""
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        normalized = frozenset(
            tuple(str(v).strip().lower() if v is not None else "" for v in row)
            for row in rows
        )
        return normalized, None
    except Exception as e:
        return None, str(e)


# ──────────────────────────────────────────────
# Carga de preguntas
# ──────────────────────────────────────────────

def load_questions(questions_path: str, limit: int = None) -> list:
    with open(questions_path, encoding="utf-8") as f:
        items = json.load(f)
    # Soporta formato Spider validation.json y experiment_questions.json
    if isinstance(items, list) and items and "SQL" not in items[0] and "sql_query" in items[0]:
        # formato experiment_questions: {"id", "db_id", "question", "sql_query"}
        items = [{"db_id": x["db_id"], "question": x["question"],
                  "query": x["sql_query"], "id": x.get("id", i)}
                 for i, x in enumerate(items)]
    if limit:
        items = items[:limit]
    return items


# ──────────────────────────────────────────────
# Evaluacion principal
# ──────────────────────────────────────────────

def run_evaluation(questions_path: str, engine: str, limit: int = None,
                   output_path: str = None, verbose: bool = False,
                   mysql_cfg: dict = None, pg_cfg: dict = None):

    from orchestrator.graph import run_query

    questions = load_questions(questions_path, limit)
    total = len(questions)

    print(f"\n{'='*65}")
    print(f"  EVALUACION SPIDER — {engine.upper()}")
    print(f"  Preguntas: {total} | Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}\n")

    results = []
    ex_correct = 0
    ex_errors  = 0
    sys_errors = 0

    for i, item in enumerate(questions, 1):
        question = item.get("question", "")
        gold_sql  = item.get("query", item.get("SQL", ""))
        db_id     = item.get("db_id", "")
        q_id      = item.get("id", i)

        # Configurar dataset activo antes de cada query
        dataset_key = f"spider:{db_id}"
        os.environ["ACTIVE_DATASET"] = dataset_key

        import importlib
        import config as _cfg
        importlib.reload(_cfg)
        try:
            import orchestrator.nodes as _nodes
            _nodes._aps_cache.clear()
        except Exception:
            pass
        try:
            import agents.AR.prompt as _ar_prompt
            importlib.reload(_ar_prompt)
        except Exception:
            pass

        if verbose:
            print(f"[{i:3}/{total}] {db_id}: {question[:60]}")

        # ── Llamar al sistema ──
        start = time.time()
        generated_sql = ""
        ar_valid = False
        aps_ok = False
        confidence = 0.0
        error_sys = None

        try:
            result = run_query(question, db_type=engine)
            elapsed = time.time() - start
            generated_sql = result.get("final_sql", "")
            ar_valid    = result.get("ar_result", {}).get("is_valid_query", False)
            aps_ok      = result.get("aps_result", {}).get("success", False)
            confidence  = round(result.get("confidence_score", 0.0), 3)
        except Exception as e:
            elapsed = time.time() - start
            sys_errors += 1
            error_sys = str(e)
            if verbose:
                print(f"       ERROR sistema: {e}")

        if not generated_sql:
            if not error_sys:
                sys_errors += 1
            label = "AR_REJECT" if not ar_valid else "SIN_SQL"
            print(f"  [{i:3}/{total}] X {label:<10} | {db_id}: {question[:45]}")
            results.append({
                "id": q_id, "db_id": db_id, "question": question,
                "gold_sql": gold_sql, "gen_sql": "",
                "exact_match": False, "exec_acc": False,
                "ar_valid": ar_valid, "aps_ok": aps_ok,
                "confidence": confidence, "time_s": round(elapsed, 2),
                "error": error_sys or "Sin SQL generado",
            })
            continue

        # ── Ejecutar ambos SQL y comparar ──
        db_name = f"spider_{db_id}"  # en MySQL/PG las BDs tienen prefijo spider_
        match = False
        gen_err = None

        try:
            if engine == "mysql":
                cfg = mysql_cfg or {}
                conn = get_mysql_conn(db_name, **cfg)
                conn.autocommit = True
                cur = conn.cursor()
            else:
                cfg = pg_cfg or {}
                conn = get_pg_conn(db_name, **cfg)
                conn.autocommit = True
                cur = conn.cursor()

            gold_rows, gold_err = execute_sql(cur, gold_sql, engine)
            # Si el Gold SQL falla por only_full_group_by en MySQL, reintentarlo
            # desactivando ese modo en la sesion (el Gold SQL del benchmark fue
            # escrito para SQLite que no tiene esta restriccion).
            if gold_err and engine == "mysql" and "only_full_group_by" in str(gold_err):
                try:
                    cur.execute("SET SESSION sql_mode = (SELECT REPLACE(@@sql_mode, 'ONLY_FULL_GROUP_BY', ''))")
                    gold_rows, gold_err = execute_sql(cur, gold_sql, engine)
                    cur.execute("SET SESSION sql_mode = @@GLOBAL.sql_mode")
                except Exception:
                    pass
            gen_rows,  gen_err  = execute_sql(cur, generated_sql, engine)
            cur.close()
            conn.close()

            if gen_err:
                ex_errors += 1
                match = False
            elif gold_err:
                match = False
                gen_err = f"Gold SQL fallo: {gold_err}"
            else:
                match = (gen_rows == gold_rows)
                # Fallback: para resultados de una sola fila, comparar valores
                # sin importar el orden de columnas (cubre casos donde el Gold SQL
                # tiene distinto orden de columnas que el enunciado de la pregunta).
                if not match and len(gold_rows) == 1 and len(gen_rows) == 1:
                    gold_vals = sorted(str(v) for v in list(gold_rows)[0])
                    gen_vals  = sorted(str(v) for v in list(gen_rows)[0])
                    match = (gold_vals == gen_vals)
                if match:
                    ex_correct += 1

        except Exception as e:
            ex_errors += 1
            gen_err = f"Conexion BD: {e}"

        status = "OK" if match else "X"
        label  = "MATCH    " if match else ("SQL_ERR  " if gen_err and "Gold" not in str(gen_err) else "MISMATCH ")
        print(f"  [{i:3}/{total}] {status} {label} | conf={confidence} | {db_id}: {question[:40]}")
        if not match and verbose and gen_err:
            print(f"       err: {gen_err}")

        results.append({
            "id": q_id, "db_id": db_id, "question": question,
            "gold_sql": gold_sql, "gen_sql": generated_sql,
            "exact_match": generated_sql.strip().lower() == gold_sql.strip().lower(),
            "exec_acc": match,
            "ar_valid": ar_valid, "aps_ok": aps_ok,
            "confidence": confidence, "time_s": round(elapsed, 2),
            "error": gen_err,
        })

    # ── Métricas finales ──
    ex_score = ex_correct / total if total else 0
    valid_sql = total - sys_errors
    ex_valid  = ex_correct / valid_sql if valid_sql else 0

    print(f"\n{'='*65}")
    print(f"  RESULTADOS FINALES — Spider / {engine.upper()}")
    print(f"{'='*65}")
    print(f"  Total preguntas :      {total}")
    print(f"  SQL generado    :      {total - sys_errors}  ({(total-sys_errors)/total:.0%})")
    print(f"  Errores sistema :      {sys_errors}")
    print(f"  Errores SQL exec:      {ex_errors}")
    print(f"  Correctos (EX)  :      {ex_correct}")
    print(f"")
    print(f"  Execution Accuracy (EX):          {ex_score:.1%}  ({ex_correct}/{total})")
    print(f"  EX sobre SQL validos:             {ex_valid:.1%}  ({ex_correct}/{valid_sql})")
    print(f"{'='*65}")

    by_db = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_db[r["db_id"]]["total"] += 1
        if r["exec_acc"]:
            by_db[r["db_id"]]["correct"] += 1

    print(f"\n  Accuracy por base de datos:")
    for db_name, stats in sorted(by_db.items()):
        acc = stats["correct"] / stats["total"]
        bar = "+" * stats["correct"] + "-" * (stats["total"] - stats["correct"])
        print(f"    {db_name:<35} {bar}  {acc:.0%}  ({stats['correct']}/{stats['total']})")

    output = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "Spider",
        "engine": engine,
        "total": total,
        "ex_correct": ex_correct,
        "ex_score": round(ex_score, 4),
        "ex_score_valid": round(ex_valid, 4),
        "sys_errors": sys_errors,
        "exec_errors": ex_errors,
        "by_db": {k: dict(v) for k, v in by_db.items()},
        "results": results,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n  Guardado en: {output_path}")

    return output


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine",    default="mysql", choices=["mysql", "postgres"])
    parser.add_argument("--questions", default="metadata/spider/experiment_questions.json")
    parser.add_argument("--limit",     type=int, default=10)
    parser.add_argument("--output",    default=None)
    parser.add_argument("--verbose",   action="store_true")
    # MySQL
    parser.add_argument("--mysql_host",     default="localhost")
    parser.add_argument("--mysql_port",     type=int, default=3306)
    parser.add_argument("--mysql_user",     default="root")
    parser.add_argument("--mysql_password", default="123")
    # PostgreSQL
    parser.add_argument("--pg_host",     default="localhost")
    parser.add_argument("--pg_port",     type=int, default=5433)
    parser.add_argument("--pg_user",     default="postgres")
    parser.add_argument("--pg_password", default="123")
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        args.output = f"resultados_spider_{args.engine}_{ts}.json"

    mysql_cfg = {"host": args.mysql_host, "port": args.mysql_port,
                 "user": args.mysql_user, "password": args.mysql_password}
    pg_cfg    = {"host": args.pg_host, "port": args.pg_port,
                 "user": args.pg_user, "password": args.pg_password}

    run_evaluation(
        questions_path=args.questions,
        engine=args.engine,
        limit=args.limit,
        output_path=args.output,
        verbose=args.verbose,
        mysql_cfg=mysql_cfg,
        pg_cfg=pg_cfg,
    )
