"""
sustainer_agent.py

AS - Sustainer Agent with Agent Skills.

Architecture:
  - get_agent_reasoning : retrieves the detailed reasoning of a specific pipeline
    agent (AR, APS, AG, AV) from the current pipeline state (programmatic, no LLM).

DIFFERENCE from AE (Explainer):
- AE explains WHAT the query does (SQL → natural language for the end user)
- AS explains WHY it was generated that way (justifies pipeline decisions for an evaluator)

The LLM may invoke get_agent_reasoning to inspect a specific agent's reasoning
before producing its justification.

Location: agents/AS/sustainer_agent.py
"""

import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_client import get_model, get_client_for_model
from agents.base_skill_agent import SkillAgent
from config import SUSTAINER_MAX_TOKENS


class SustainerAgent(SkillAgent):

    def __init__(self):
        self.model = get_model("AS") or "gpt-4o"   # fallback para evitar None en init
        self.client = get_client_for_model(self.model)
        self._agent_label = "AS-Sustainer"

        self.max_tokens = SUSTAINER_MAX_TOKENS

        self.context_history = []
        self._last_result_cache = {}

    # ------------------------------------------------------------------
    # Agent skills
    # ------------------------------------------------------------------

    def _get_skills(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_agent_reasoning",
                    "description": (
                        "Retrieves the detailed reasoning of a specific pipeline agent "
                        "(AR, APS, AG, AV) FROM THE CURRENT TURN. "
                        "Use this skill when the user asks about the most recently processed "
                        "query ('why those tables?', 'why that JOIN?', 'how confident were you?')."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent": {
                                "type": "string",
                                "enum": ["AR", "APS", "AG", "AV"],
                                "description": "Name of the agent whose reasoning to retrieve"
                            }
                        },
                        "required": ["agent"]
                    }
                }
            }
        ]

    def _execute_skill(self, skill_name: str, args: dict) -> str:
        if skill_name == "get_agent_reasoning":
            from agents.AS.skills import get_agent_reasoning
            result = get_agent_reasoning(self._last_result_cache, args.get("agent", ""))
            return json.dumps(result)
        return json.dumps({"error": f"Skill '{skill_name}' not recognized in AS"})

    def process(self, last_result, user_question, temperature: float = 0.3):
        """
        Generates a justification using the agentic loop with skills.

        The LLM may invoke get_agent_reasoning to inspect specific agents'
        reasoning before producing its final justification.
        """
        # Load result into cache so skills can access it
        self._last_result_cache = last_result

        ar = last_result.get("ar_result", {})
        aps = last_result.get("aps_result", {})
        ag = last_result.get("ag_result", {})
        av = last_result.get("av_result", {})
        ae = last_result.get("ae_result", {}) or {}
        final_sql = last_result.get("final_sql", "")
        user_input = last_result.get("user_input", "")
        iterations = last_result.get("iteration_count", 0)

        ae_text = (
            (ae.get("explanation") or {}).get("final_reasoning", "")
            if isinstance(ae.get("explanation"), dict)
            else ""
        )

        context_text = ""
        if self.context_history:
            lines = [f"  - {e['question'][:60]}" for e in self.context_history[-3:]]
            context_text = "\nPREVIOUS QUESTIONS:\n" + "\n".join(lines)

        from agents.AS.prompt import SYSTEM_PROMPT_EN
        system_prompt = SYSTEM_PROMPT_EN

        user_message = (
            f'User asks: "{user_question}"\n\n'
            f"AVAILABLE DATA:\n"
            f'- Original question: "{user_input}"\n'
            f'- Interpreted as: "{ar.get("refined_query", "")}"\n'
            f"- SQL: {final_sql[:200]}\n"
            f"- Strategy: {ag.get('strategy', '?')}\n"
            f"- Tables: {', '.join(list(aps.get('tables', {}).keys()))}\n"
            f"- Validation: {'OK' if av.get('is_valid') else 'Failed'} in {iterations} attempt(s)\n"
            f"- Errors: {av.get('errors', []) or 'None'}\n"
            f"- AE explanation (already shown to the user): {ae_text[:300] or '(no AE)'}\n"
            f"{context_text}\n\n"
            f"Use get_agent_reasoning if you need more detail from a specific pipeline agent."
        )

        try:
            raw = self._run_agent_loop(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                json_output=True,
            )
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                return {
                    "justification":    parsed.get("justification", str(raw)),
                    "confidence_score": float(parsed.get("confidence_score", 0.75)),
                }
            except Exception:
                return {"justification": str(raw).strip(), "confidence_score": 0.75}
        except Exception as exc:
            return {"justification": f"Error: {exc}", "confidence_score": 0.5}

    def add_to_context(self, question, sql):
        self.context_history.append({"question": question, "sql": sql})

    def reset_context(self):
        self.context_history = []

    def get_info(self):
        return {
            "name": "AS-Sustainer",
            "model": self.model,
            "max_tokens": self.max_tokens,
            "context_entries": len(self.context_history),
            "skills": ["get_agent_reasoning"],
        }
