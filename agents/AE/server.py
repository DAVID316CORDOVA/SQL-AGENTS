# -*- coding: utf-8 -*-
"""
agents/AE/server.py — FastAPI microservice para el Agente Explicador.
Puerto: 8005
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
from typing import Any, Dict, List, Optional
import time

app = FastAPI(title="AE - Agente Explicador", version="1.0")

CONFIDENCE_WEIGHTS = {"AR": 0.10, "APS": 0.20, "AG": 0.30, "AV": 0.40}


class AERequest(BaseModel):
    ar_result: Optional[Dict[str, Any]] = None
    aps_result: Optional[Dict[str, Any]] = None
    ag_result: Optional[Dict[str, Any]] = None
    av_result: Optional[Dict[str, Any]] = None
    final_sql: str = ""
    db_type: str = "mysql"
    iteration_count: int = 0
    conversation_history: Optional[List[Dict[str, Any]]] = None
    overall_confidence: float = 0.0
    active_dataset: str = "demo_db"


@app.get("/health")
def health():
    return {"agent": "AE", "status": "ok"}


@app.post("/invoke")
def invoke(req: AERequest) -> Dict[str, Any]:
    t0 = time.time()
    os.environ["ACTIVE_DATASET"] = req.active_dataset
    import config as _cfg
    importlib.reload(_cfg)
    ar_result  = req.ar_result  or {}
    aps_result = req.aps_result or {}
    ag_result  = req.ag_result  or {}
    av_result  = req.av_result  or {}

    arc  = ar_result.get("confidence_score", 0)
    apsc = aps_result.get("confidence_score", 0)
    agc  = ag_result.get("confidence_score", 0)
    avc  = av_result.get("confidence_score", 0)

    ar_valid   = ar_result.get("is_valid_query")
    aps_ok     = aps_result.get("success", False)
    ag_ok      = ag_result.get("success", False)

    if not ar_valid:
        overall = 1.0 - arc
    elif not aps_ok or aps_result.get("below_threshold"):
        overall = 1.0 - apsc
    elif not ag_ok:
        overall = 0.0
    else:
        overall = (arc * CONFIDENCE_WEIGHTS["AR"] + apsc * CONFIDENCE_WEIGHTS["APS"] +
                   agc * CONFIDENCE_WEIGHTS["AG"] + avc * CONFIDENCE_WEIGHTS["AV"])
    overall = round(overall, 3)

    # Rechazos simples sin LLM
    if not ar_valid:
        return {"ae_result": {"explanation": {"final_reasoning": ""}},
                "final_sql": "", "overall_confidence": overall,
                "is_complete": True,
                "all_reasoning": ["[AE]\nRechazada por AR"],
                "agent_times": {"AE": round(time.time() - t0, 2)}}

    full_state = {
        "ar_result": ar_result, "aps_result": aps_result,
        "ag_result": ag_result, "av_result": av_result,
        "final_sql": req.final_sql, "db_type": req.db_type,
        "overall_confidence": overall,
        "iteration_count": req.iteration_count,
        "conversation_history": req.conversation_history or [],
    }

    from agents.AE.explainer_agent import ExplainerAgent
    from config import AGENT_TEMPERATURES
    temp = AGENT_TEMPERATURES.get("AE", 0.5)
    agent = ExplainerAgent()
    ae_result = agent.process(full_state, temperature=temp)
    return {
        "ae_result": ae_result,
        "final_sql": req.final_sql,
        "overall_confidence": overall,
        "is_complete": True,
        "all_reasoning": [f"[AE]\n{ae_result.get('explanation', {}).get('final_reasoning', '')}"],
        "agent_times": {"AE": round(time.time() - t0, 2)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
