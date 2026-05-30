"""
explainer_agent.py

AE - Agente Explicador con Agent Skills.

Arquitectura Agent Skills:
  - format_sql_readable : formatea el SQL final con saltos de linea para
    mostrarlo legible al usuario (skill invocable por el LLM).

ROL del AE (alineado con la tesis):
- Genera una explicacion UNIFICADA del pipeline completo (AR/APS/AG/AV)
  en lenguaje simple, para usuarios sin conocimientos tecnicos.
- Recibe el estado completo de todos los agentes (full_state).
- Sintetiza: que se entendio, con que datos se respondio, con que
  nivel de confianza, y si hubo correcciones del validador.

Caso especial — rechazo tras encadenamiento exitoso:
- Si APS rechaza una pregunta de seguimiento (la BD no tiene la columna
  necesaria) tras un encadenamiento exitoso, el AE consulta su
  context_history y construye un mensaje que menciona el ultimo SQL
  util generado, indicando el alcance maximo del sistema dado el
  esquema actual.

DIFERENCIA con AS (Sustainer):
- AE narra el pipeline en lenguaje cotidiano (audiencia: usuario final).
- AS justifica las decisiones tecnicas de cada agente bajo demanda
  (audiencia: evaluador / academico).

Ubicacion: agents/AE/explainer_agent.py
"""

import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_client import get_model, get_client_for_model
from agents.base_skill_agent import SkillAgent
from config import EXPLAINER_MAX_TOKENS


class ExplainerAgent(SkillAgent):

    def __init__(self):
        self.model = get_model("AE")
        self.client = get_client_for_model(self.model)
        self._agent_label = "AE-Explainer"

        self.max_tokens = EXPLAINER_MAX_TOKENS

        # Historial conversacional. Cada entrada: {"question": ..., "sql": ...}
        # Lo usa el caso especial de rechazo tras encadenamiento exitoso.
        self.context_history = []

        # Rastreo de invocaciones REALES de skills (se resetea en cada process())
        self._last_skills_invoked: list[str] = []

    # ------------------------------------------------------------------
    # Skills del agente
    # ------------------------------------------------------------------

    def _get_skills(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "format_sql_readable",
                    "description": (
                        "Formats the SQL with line breaks before each major "
                        "keyword (SELECT, FROM, WHERE, JOIN, GROUP BY, etc.) "
                        "for readability. Returns sql_formatted. "
                        "Invoke ONCE with the final pipeline SQL."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL crudo a formatear"
                            }
                        },
                        "required": ["sql"]
                    }
                }
            }
        ]

    def _execute_skill(self, skill_name: str, args: dict) -> str:
        # Registrar invocacion real (independiente del auto-reporte del modelo)
        if skill_name not in self._last_skills_invoked:
            self._last_skills_invoked.append(skill_name)

        if skill_name == "format_sql_readable":
            from agents.AE.skills import format_sql_readable
            result = format_sql_readable(args.get("sql", ""))
            return json.dumps(result)
        return json.dumps({"error": f"Skill '{skill_name}' no reconocida en AE"})

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def process(self, full_state, temperature: float = 0.3):
        """
        Genera explicacion unificada del pipeline usando agentic loop.

        Args:
          full_state: dict con ar_result, aps_result, ag_result, av_result,
                      final_sql, overall_confidence, iteration_history,
                      user_input. Puede incluir tambien:
                        - chain_rejection (bool): True si APS rechazo una
                          pregunta de seguimiento tras un encadenamiento.
                        - last_useful_sql (str): ultimo SQL util previo
                          (puede tomarse del state o del context_history).
          temperature: temperatura del LLM.
        """
        # Resetear tracking de skills para esta invocacion
        self._last_skills_invoked = []

        user_input = full_state.get("user_input", "")
        ar = full_state.get("ar_result", {}) or {}
        aps = full_state.get("aps_result", {}) or {}
        ag = full_state.get("ag_result", {}) or {}
        av = full_state.get("av_result", {}) or {}
        final_sql = full_state.get("final_sql", "")
        overall_confidence = full_state.get("overall_confidence", 0)
        iteration_history = full_state.get("iteration_history", [])

        # Caso especial: rechazo tras encadenamiento exitoso.
        chain_rejection = bool(full_state.get("chain_rejection", False))
        last_useful_sql = full_state.get("last_useful_sql", "")
        if chain_rejection and not last_useful_sql and self.context_history:
            last_useful_sql = self.context_history[-1].get("sql", "")

        # AE siempre en ingles — SYSTEM_PROMPT_EN para todos los datasets
        from agents.AE.prompt import SYSTEM_PROMPT_EN as AE_SYSTEM_PROMPT
        system_prompt = AE_SYSTEM_PROMPT

        # ── Detectar escenario real usando campos reales de los agentes ──
        # AR: success siempre es True, el fallo se detecta por is_valid_query=False
        ar_rejected  = not ar.get("is_valid_query", True)
        # APS: success=False cuando ChromaDB no encontro tablas/columnas suficientes
        aps_failed   = not aps.get("success", True)
        # AV: unfixable=True cuando los reintentos se agotaron
        av_exhausted = bool(av.get("unfixable", False))

        # ── Construir tabla de tablas usadas (solo para SUCCESS/AV_EXHAUSTED) ──
        tables_dict = aps.get("tables", {}) or {}
        tables_info = []
        for tn, ti in tables_dict.items():
            cols = ti.get("matched_columns") or list(ti.get("all_columns", []))[:4]
            tables_info.append(f"{tn} (cols: {', '.join(cols[:4])})")

        # ── AR block ──────────────────────────────────────────────────────
        if ar_rejected:
            ar_block = (
                f"AR  - status: REJECTED\n"
                f"AR  - reason: \"{ar.get('reasoning', 'not a valid database query')}\"\n"
                "APS - not reached (AR rejected)\n"
                "AG  - not reached\n"
                "AV  - not reached\n"
            )
        else:
            ar_block = (
                f"AR  - status: accepted\n"
                f'AR  - refined query: "{ar.get("refined_query", user_input)}"\n'
            )

        # ── APS block ─────────────────────────────────────────────────────
        if ar_rejected:
            aps_block = ""   # already included in ar_block
        elif aps_failed:
            aps_block = (
                f"APS - status: REJECTED (schema mismatch)\n"
                f"APS - schema limitation: \"{aps.get('no_info_reason', 'No related tables or columns found.')}\"\n"
                "AG  - not reached\n"
                "AV  - not reached\n"
            )
        else:
            aps_block = (
                f"APS - tables found: {', '.join(tables_info) or 'none'}\n"
            )

        # ── AG + AV block ──────────────────────────────────────────────────
        av_errors    = av.get("errors", []) or []
        av_reasoning = av.get("reasoning", "") or ""
        if ar_rejected or aps_failed:
            ag_av_block = ""   # already handled above
        elif av_exhausted:
            ag_av_block = (
                f"AG  - strategy: {ag.get('strategy', 'unknown')}\n"
                f"AV  - status: EXHAUSTED after {len(iteration_history)} attempt(s)\n"
                f"AV  - last error: \"{av_errors[-1] if av_errors else 'unknown error'}\"\n"
                f"AV  - reasoning: {av_reasoning}\n"
            )
        else:
            ag_av_block = (
                f"AG  - strategy: {ag.get('strategy', 'unknown')}\n"
                f"AV  - valid: yes, iterations: {len(iteration_history)}\n"
                f"AV  - reasoning: {av_reasoning}\n"
            )

        # ── SQL block ──────────────────────────────────────────────────────
        sql_block = f"Raw SQL: {final_sql[:400]}\n" if final_sql else ""

        # ── Chain rejection block ──────────────────────────────────────────
        chain_block = ""
        if chain_rejection:
            chain_block = (
                "\nSPECIAL CASE: chain_rejection=true.\n"
                "The user's follow-up asks for data the schema does NOT have.\n"
                f"Last useful SQL: {last_useful_sql or '(none)'}\n"
                "Acknowledge the limitation and indicate what the system CAN provide.\n"
            )

        # ── Instruction line ───────────────────────────────────────────────
        if final_sql:
            action_line = (
                "Invoke format_sql_readable with the raw SQL, "
                "then write the technical explanation."
            )
        else:
            action_line = (
                "Do NOT invoke format_sql_readable (no SQL was generated). "
                "Identify the failure scenario and write the appropriate explanation."
            )

        user_message = (
            f'Original question: "{user_input}"\n\n'
            f"=== PIPELINE STATE ===\n"
            f"{ar_block}"
            f"{aps_block}"
            f"{ag_av_block}"
            f"{sql_block}"
            f"Overall confidence: {overall_confidence:.2f}\n"
            f"{chain_block}\n"
            f"{action_line}"
        )

        try:
            raw = self._run_agent_loop(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                json_output=True,
                force_tools=bool(final_sql),   # forzar skill solo cuando hay SQL
            )
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {}
            explanation_text = parsed.get("explanation", "") or ""
            formatted_sql = parsed.get("final_sql", "") or final_sql
            confidence_level = parsed.get("confidence_level", "")
            # Usar invocaciones REALES rastreadas por _execute_skill
            skills_used = list(self._last_skills_invoked)

            return {
                "success": True,
                "agent": "AE-Explainer",
                "requires_reformulation": False,
                "explanation": {
                    "final_reasoning": explanation_text,
                    "user_question": user_input,
                    "final_sql": formatted_sql,
                    "overall_confidence": overall_confidence,
                    "iteration_count": len(iteration_history),
                    "confidence_level": confidence_level,
                },
                "confidence_score": float(overall_confidence) if overall_confidence else 1.0,
                "skills_used": skills_used,
            }
        except Exception as exc:
            return {
                "success": False,
                "agent": "AE-Explainer",
                "requires_reformulation": False,
                "explanation": {
                    "final_reasoning": f"Error generando explicacion: {exc}",
                    "user_question": user_input,
                    "final_sql": final_sql,
                    "overall_confidence": overall_confidence,
                    "iteration_count": len(iteration_history),
                },
                "confidence_score": 0.0,
            }

    # ------------------------------------------------------------------
    # Historial conversacional (para chain_rejection)
    # ------------------------------------------------------------------

    def add_to_context(self, question, sql):
        self.context_history.append({"question": question, "sql": sql})

    def reset_context(self):
        self.context_history = []

    def get_info(self):
        return {
            "name": "AE-Explainer",
            "model": self.model,
            "max_tokens": self.max_tokens,
            "context_entries": len(self.context_history),
            "skills": ["format_sql_readable"],
        }
