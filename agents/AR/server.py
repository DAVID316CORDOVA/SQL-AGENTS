# -*- coding: utf-8 -*-
"""
agents/AR/server.py — FastAPI microservice para el Agente Refinador.
Puerto: 8001
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

app = FastAPI(title="AR - Agente Refinador", version="1.0")


class ARRequest(BaseModel):
    user_input: str
    db_type: str = "mysql"
    active_dataset: str = "demo_db"


@app.get("/health")
def health():
    return {"agent": "AR", "status": "ok"}


@app.post("/invoke")
def invoke(req: ARRequest) -> Dict[str, Any]:
    t0 = time.time()
    # Aplicar dataset activo para que el agente use el contexto correcto de BD
    os.environ["ACTIVE_DATASET"] = req.active_dataset
    import config as _cfg
    importlib.reload(_cfg)

    from agents.AR.refiner_agent import RefinerAgent
    from config import AGENT_TEMPERATURES
    temp = AGENT_TEMPERATURES.get("AR", 0.7)
    agent = RefinerAgent()
    result = agent.process(req.user_input, temperature=temp)
    return {
        "ar_result": result,
        "all_reasoning": [f"[AR-Refiner]\n{result.get('reasoning', '')}"],
        "iteration_count": 0,
        "iteration_history": [],
        "current_feedback": None,
        "current_sql": "",
        "final_sql": "",
        "agent_times": {"AR": round(time.time() - t0, 2)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
