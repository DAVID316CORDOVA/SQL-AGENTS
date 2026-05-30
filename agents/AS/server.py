# -*- coding: utf-8 -*-
"""
agents/AS/server.py — FastAPI microservice para el Agente Sustentador.
Puerto: 8006
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

app = FastAPI(title="AS - Agente Sustentador", version="1.0")


class ASRequest(BaseModel):
    user_input: str
    last_result_for_as: Optional[Dict[str, Any]] = None
    active_dataset: str = "demo_db"


@app.get("/health")
def health():
    return {"agent": "AS", "status": "ok"}


@app.post("/invoke")
def invoke(req: ASRequest) -> Dict[str, Any]:
    t0 = time.time()
    os.environ["ACTIVE_DATASET"] = req.active_dataset
    import config as _cfg
    importlib.reload(_cfg)
    last_result = req.last_result_for_as or {}
    if not last_result or not last_result.get("final_sql"):
        return {
            "ae_result": {"explanation": {"final_reasoning": "No hay consulta anterior para explicar."}},
            "is_complete": True,
            "all_reasoning": ["[AS] Sin resultado anterior"],
            "agent_times": {"AS": round(time.time() - t0, 2)},
        }
    from agents.AS.sustainer_agent import SustainerAgent
    from config import AGENT_TEMPERATURES
    temp = AGENT_TEMPERATURES.get("AS", 0.0)
    agent = SustainerAgent()
    answer = agent.process(last_result, req.user_input, temperature=temp)
    return {
        "ae_result": {"explanation": {"final_reasoning": answer}},
        "final_sql": last_result.get("final_sql", ""),
        "is_complete": True,
        "all_reasoning": [f"[AS]\n{answer}"],
        "agent_times": {"AS": round(time.time() - t0, 2)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
