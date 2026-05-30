# -*- coding: utf-8 -*-
"""
agents/AV/server.py — FastAPI microservice para el Agente Validador SQL.
Puerto: 8004
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

app = FastAPI(title="AV - Agente Validador SQL", version="1.0")


class AVRequest(BaseModel):
    ag_result: Dict[str, Any]
    aps_result: Dict[str, Any]
    db_type: str = "mysql"
    iteration_count: int = 0
    ar_result: Optional[Dict[str, Any]] = None
    active_dataset: str = "demo_db"


@app.get("/health")
def health():
    return {"agent": "AV", "status": "ok"}


@app.post("/invoke")
def invoke(req: AVRequest) -> Dict[str, Any]:
    t0 = time.time()
    os.environ["ACTIVE_DATASET"] = req.active_dataset
    import config as _cfg
    importlib.reload(_cfg)
    db_type = req.db_type
    ag_result = req.ag_result

    if not ag_result.get("success"):
        return {
            "av_result": {"success": False, "agent": "AV-SQLValidator",
                          "is_valid": False, "confidence_score": 0.0,
                          "reasoning": "AG fallo"},
            "all_reasoning": ["[AV]\nNo ejecutado - AG fallo"],
            "agent_times": {"AV": round(time.time() - t0, 2)},
        }

    if db_type == "postgres":
        from agents.AV.postgres.sql_validator_agent import SQLValidatorAgent
    else:
        from agents.AV.mysql.sql_validator_agent import SQLValidatorAgent

    from config import AGENT_TEMPERATURES
    temp = AGENT_TEMPERATURES.get("AV", 0.3)
    agent = SQLValidatorAgent()
    result = agent.process(
        ag_result=ag_result,
        ar_result=req.ar_result or {},
        aps_result=req.aps_result,
        temperature=temp,
    )
    return {
        "av_result": result,
        "all_reasoning": [f"[AV]\n{result.get('reasoning', '')}"],
        "agent_times": {"AV": round(time.time() - t0, 2)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
