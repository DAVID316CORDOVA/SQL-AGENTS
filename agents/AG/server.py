# -*- coding: utf-8 -*-
"""
agents/AG/server.py — FastAPI microservice para el Agente Generador SQL.
Puerto: 8003
"""
import os, sys, importlib
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
try:
    from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional
import time

app = FastAPI(title="AG - Agente Generador SQL", version="1.0")


class AGRequest(BaseModel):
    aps_result: Dict[str, Any]
    db_type: str = "mysql"
    feedback: Optional[str] = None
    ar_result: Optional[Dict[str, Any]] = None
    user_input: str = ""
    current_sql: str = ""
    active_dataset: str = "demo_db"


@app.get("/health")
def health():
    return {"agent": "AG", "status": "ok"}


@app.post("/invoke")
def invoke(req: AGRequest) -> Dict[str, Any]:
    t0 = time.time()
    # Aplicar dataset activo para que SCHEMA_PATH apunte al schema correcto
    os.environ["ACTIVE_DATASET"] = req.active_dataset
    import config as _cfg
    importlib.reload(_cfg)

    db_type = req.db_type
    aps_result = req.aps_result

    if not aps_result.get("success"):
        return {
            "ag_result": {"success": False, "agent": "AG-SQLGenerator",
                          "confidence_score": 0.0, "reasoning": "APS fallo", "sql": ""},
            "current_sql": "", "final_sql": "",
            "all_reasoning": ["[AG]\nNo ejecutado - APS fallo"],
            "agent_times": {"AG": round(time.time() - t0, 2)},
        }

    if db_type == "postgres":
        from agents.AG.postgres.sql_generator_agent import SQLGeneratorAgent
    else:
        from agents.AG.mysql.sql_generator_agent import SQLGeneratorAgent

    # Garantizar input_query
    if not aps_result.get("input_query"):
        ar = req.ar_result or {}
        aps_result["input_query"] = ar.get("refined_query") or req.user_input

    # Garantizar JOINs si hay múltiples tablas
    if not aps_result.get("joins") and len(aps_result.get("tables", {})) > 1:
        try:
            import json
            from config import SCHEMA_PATH
            with open(SCHEMA_PATH) as f:
                schema = json.load(f)
            joins = [{"join_hint": r["join_hint"]} for r in schema.get("relationships", []) if r.get("join_hint")]
            aps_result["joins"] = joins
        except Exception:
            pass

    if not aps_result.get("original_intent") and req.ar_result:
        aps_result["original_intent"] = req.ar_result.get("intent", {})

    from config import AGENT_TEMPERATURES
    temp = AGENT_TEMPERATURES.get("AG", 0.3)
    agent = SQLGeneratorAgent()
    result = agent.process(aps_result, feedback_from_av=req.feedback, temperature=temp)
    sql = result.get("sql", "")
    return {
        "ag_result": result,
        "current_sql": sql,
        "final_sql": sql if result.get("success") else "",
        "all_reasoning": [f"[AG]\n{result.get('reasoning', '')}"],
        "agent_times": {"AG": round(time.time() - t0, 2)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
