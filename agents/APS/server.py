# -*- coding: utf-8 -*-
"""
agents/APS/server.py — FastAPI microservice para el Agente de Proximidad Semántica.
Puerto: 8002
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

app = FastAPI(title="APS - Agente Proximidad Semántica", version="1.0")

_aps_cache: Dict[str, Any] = {}


def _get_aps(db_type: str, active_dataset: str):
    """Obtiene (o crea) el agente APS para el dataset+backend dado."""
    # config ya fue recargado antes de llamar aquí; los imports capturan el valor nuevo
    from config import SCHEMA_PATH, VECTOR_DB_PATH
    key = f"{db_type}::{active_dataset}::{SCHEMA_PATH}"
    if key not in _aps_cache:
        from agents.APS import SchemaMatcherAgent
        _aps_cache[key] = SchemaMatcherAgent(
            schema_path=SCHEMA_PATH,
            vector_db_path=VECTOR_DB_PATH,
            db_type=db_type,
        )
    return _aps_cache[key]


class APSRequest(BaseModel):
    ar_result: Dict[str, Any]
    db_type: str = "mysql"
    last_result_for_as: Optional[Dict[str, Any]] = None
    active_dataset: str = "demo_db"


@app.get("/health")
def health():
    return {"agent": "APS", "status": "ok"}


@app.post("/invoke")
def invoke(req: APSRequest) -> Dict[str, Any]:
    t0 = time.time()
    # Aplicar dataset activo antes de resolver los paths de schema/chroma
    os.environ["ACTIVE_DATASET"] = req.active_dataset
    import config as _cfg
    importlib.reload(_cfg)

    ar_result = req.ar_result
    if not ar_result.get("is_valid_query"):
        return {
            "aps_result": {"success": False, "agent": "APS-SchemaMatcher",
                           "confidence_score": 0.0, "reasoning": "AR rechazo"},
            "all_reasoning": ["[APS]\nNo ejecutado - consulta rechazada"],
            "agent_times": {"APS": round(time.time() - t0, 2)},
        }
    agent = _get_aps(req.db_type, req.active_dataset)
    result = agent.process(ar_result)
    return {
        "aps_result": result,
        "all_reasoning": [f"[APS]\n{result.get('reasoning', '')}"],
        "agent_times": {"APS": round(time.time() - t0, 2)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
