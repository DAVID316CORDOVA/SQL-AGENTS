"""
agents/AR/refiner_agent.py

AR - Agente Refinador (primer filtro del pipeline NL→SQL).

Responsabilidad: recibe la pregunta cruda del usuario y decide si es una
consulta de base de datos valida. Si lo es, la refina (corrige ambiguedades,
normaliza entidades) y produce una pregunta enriquecida para el resto del pipeline.
Si no lo es (saludo, pregunta general, pregunta sin relacion con datos), la rechaza.

Umbral de decision: confidence_score > CONFIDENCE_THRESHOLDS["AR"] (default 0.60).
Por debajo del umbral, is_valid_query = False y el pipeline termina en AE sin
pasar por APS ni AG.

Contexto de base de datos: si el servidor MCP esta disponible, AR recibe
una descripcion en lenguaje natural del esquema activo para entender mejor
que entidades son validas en la BD actual.
"""

import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_client import get_model, get_client_for_model
from agents.base_skill_agent import SkillAgent
from agents.AR.prompt import SYSTEM_PROMPT as _SYSTEM_PROMPT
from config import ACTIVE_DATASET, CONFIDENCE_THRESHOLDS


class RefinerAgent(SkillAgent):

    def __init__(self):
        self.model = get_model("AR")
        self.client = get_client_for_model(self.model)
        self._agent_label = "AR-Refiner"

    def process(self, user_input: str, temperature: float = 0.1,
                top_p: float | None = None) -> dict:

        system_prompt = _SYSTEM_PROMPT

        try:
            # En Docker cada contenedor fija su dataset via env var; se lee aqui
            # en tiempo de ejecucion para no quedar atado al valor del import inicial.
            _active_ds = os.environ.get("ACTIVE_DATASET", ACTIVE_DATASET)
            # Seleccionar la funcion de contexto segun modo de despliegue:
            # - Modo HTTP (Docker): MCP corre en su propio contenedor, se accede por URL.
            # - Modo local: MCP se invoca directamente como modulo Python.
            if os.environ.get("AGENT_MCP_URL"):
                from agents.MCP.client import get_database_description_client
                db_context = get_database_description_client(_active_ds)
            else:
                from agents.MCP.tools.ar_tools import get_database_description_server
                db_context = get_database_description_server(_active_ds)
            # Inyectar la descripcion de la BD en el system prompt solo si hay contenido real.
            # Esto permite que AR sepa, por ejemplo, que "alumnos" existe en la BD
            # aunque el usuario haya escrito "students" (matching semantico).
            if db_context and "sin descripcion disponible" not in db_context.lower():
                system_prompt = system_prompt.replace(
                    "\n=== INSTRUCTIONS ===",
                    f"\n=== ACTIVE DATABASE CONTEXT ===\n{db_context}\n\n=== INSTRUCTIONS ==="
                )
        except Exception:
            pass

        # Ejecutar el loop ReAct. AR no usa skills; produce directamente un JSON
        # con refined_query, confidence_score y reasoning.
        raw = self._run_agent_loop(
            system_prompt=system_prompt,
            user_message=user_input,
            temperature=temperature,
            top_p=top_p,
        )

        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            result = {}

        confidence = result.get("confidence_score", 0.0)
        threshold  = CONFIDENCE_THRESHOLDS.get("AR", 0.80)
        # La query es valida para el pipeline solo si la confianza supera el umbral.
        # Por debajo: el usuario escribio algo que no es una consulta de BD.
        is_valid   = confidence > threshold

        return {
            "success":        True,
            "agent":          "AR-Refiner",
            "is_valid_query": is_valid,
            "refined_query":  result.get("refined_query", user_input).strip() if is_valid else "",
            "confidence_score": confidence,
            "reasoning":      result.get("reasoning", ""),
            "original_input": user_input,
            "skills_used":    []
        }

    def get_info(self):
        return {"name": "AR-Refiner", "model": self.model, "skills": []}
