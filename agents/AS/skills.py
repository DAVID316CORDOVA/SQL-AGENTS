"""
agents/AS/skills.py

Skills for the AS (Sustainer Agent).
  - get_agent_reasoning : retrieves the reasoning of a specific pipeline agent
                          from the current session result.
"""


def get_agent_reasoning(last_result: dict, agent: str) -> dict:
    """
    Retrieves the detailed reasoning of a specific pipeline agent.
    agent: "AR", "APS", "AG", or "AV"
    """
    if not last_result:
        return {"error": "No previous result available"}

    key = f"{agent.lower()}_result"
    result = last_result.get(key, {})
    if not result:
        return {"error": f"No data available for agent {agent}"}

    return {
        "agent": agent.upper(),
        "reasoning": result.get("reasoning", "No reasoning available"),
        "confidence_score": result.get("confidence_score", 0),
        "key_data": {
            k: v for k, v in result.items()
            if k in ("refined_query", "tables", "sql", "strategy",
                     "is_valid", "errors", "intent", "is_valid_query")
        }
    }
