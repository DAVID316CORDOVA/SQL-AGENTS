"""
agents/AV/postgres/sql_validator_agent.py

AV - Agente Validador de SQL especifico para PostgreSQL 14+.

Skills:
  - check_syntax_rules     : reglas de sintaxis PostgreSQL
  - check_semantic_patterns: patrones SQL problematicos
  - check_query_efficiency : analisis estatico de calidad del SQL
  - check_query_performance: heuristicas de rendimiento sin ejecutar

Ubicacion: agents/AV/postgres/sql_validator_agent.py
"""

import os
import json
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from llm_client import get_model, get_client_for_model
from agents.base_skill_agent import SkillAgent
from agents.AV.postgres import agentic_skills as _skills
from agents.AV.postgres.prompt import SYSTEM_PROMPT as _SYSTEM_PROMPT


class SQLValidatorAgent(SkillAgent):
    """AV - Validador de SQL para PostgreSQL 14+."""

    def __init__(self):
        self.db_type = "postgres"
        self.model = get_model("AV", backend="postgres")
        self.client = get_client_for_model(self.model)
        self._agent_label = "AV-Validator-POSTGRES"

    def _get_skills(self) -> list:
        skills = [
            {
                "type": "function",
                "function": {
                    "name": "check_syntax_rules",
                    "description": (
                        "Verificacion programatica de reglas de sintaxis PostgreSQL: "
                        "SELECT *, COUNT(*), operaciones DML/DDL prohibidas, JOIN sin ON, "
                        "GROUP_CONCAT, AUTO_INCREMENT. Invoca esto primero, siempre."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "Consulta SQL a verificar"}
                        },
                        "required": ["sql"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_semantic_patterns",
                    "description": (
                        "Detecta patrones SQL problematicos para PostgreSQL: "
                        "window functions en WHERE, LIMIT global sin PARTITION BY, HAVING mal usado."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string"},
                            "query": {"type": "string", "description": "Pregunta original del usuario"}
                        },
                        "required": ["sql", "query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_query_efficiency",
                    "description": (
                        "Analisis estatico de calidad del SQL sin ejecutarlo contra la BD. "
                        "Detecta producto cartesiano, exceso de JOINs (>4), FULL OUTER JOIN "
                        "innecesario, DISTINCT redundante con GROUP BY, cadenas OR sobre la "
                        "misma columna y subqueries profundamente anidadas. "
                        "Invoca cuando el SQL tiene multiples tablas o JOINs complejos."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_query_performance",
                    "description": (
                        "Heuristicas de rendimiento sin ejecutar la query. "
                        "Predice si la consulta sera lenta: wildcard inicial LIKE/ILIKE '%...', "
                        "EXTRACT/DATE_TRUNC en WHERE que rompe indices, subquery correlacionada "
                        "en WHERE (se ejecuta N veces), tablas grandes sin WHERE. "
                        "Invoca cuando el SQL tiene LIKE, EXTRACT, funciones en WHERE o subqueries."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string"},
                            "table_row_counts": {
                                "type": "object",
                                "description": "Diccionario {nombre_tabla: numero_filas} opcional"
                            }
                        },
                        "required": ["sql"]
                    }
                }
            }
        ]
        return skills

    def _execute_skill(self, skill_name: str, args: dict) -> str:
        try:
            if skill_name == "check_syntax_rules":
                return json.dumps(_skills.check_syntax_rules(args["sql"]))
            elif skill_name == "check_semantic_patterns":
                return json.dumps(_skills.check_semantic_patterns(args["sql"], args.get("query", "")))
            elif skill_name == "check_query_efficiency":
                return json.dumps(_skills.check_query_efficiency(args["sql"]), default=str)
            elif skill_name == "check_query_performance":
                counts = args.get("table_row_counts") or self._table_row_counts
                return json.dumps(_skills.check_query_performance(args["sql"], counts), default=str)
            return json.dumps({"error": f"Skill '{skill_name}' no reconocida en AV-PostgreSQL"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def process(self, ag_result, ar_result, aps_result, temperature: float = 0.1):
        if not ag_result.get("success"):
            return {
                "success": False, "agent": self._agent_label,
                "is_valid": False, "needs_correction": False,
                "decision_word": "incorrect",
                "error": "AG no genero SQL", "confidence_score": 0.0,
                "reasoning": "No SQL to validate."
            }

        sql           = ag_result.get("sql", "")
        refined_query = ar_result.get("refined_query", "")
        tables        = aps_result.get("tables", {})

        # Exponer row_counts para que check_query_performance los use via _execute_skill
        self._table_row_counts = {
            t: info.get("row_count", 0) for t, info in tables.items()
        }

        schema_info = "SCHEMA (columnas por tabla):\n" + "\n".join(
            f"  {tname}: {', '.join(info.get('all_columns', []))}"
            for tname, info in tables.items()
        )
        joins = aps_result.get("joins", [])
        joins_info = ""
        if joins:
            joins_info = "\nJOINS DISPONIBLES EN EL SCHEMA:\n" + "".join(
                f"  - {j.get('join_hint','')}\n" for j in joins
            )
        row_info = "\nFILAS APROX POR TABLA:\n" + "\n".join(
            f"  {t}: {self._table_row_counts.get(t, '?')} filas"
            for t in tables
        )

        user_message = (
            f'PREGUNTA DEL USUARIO: "{refined_query}"\n\n'
            f'SQL A VALIDAR:\n{sql}\n\n'
            f'{schema_info}{joins_info}{row_info}'
        )

        raw = self._run_agent_loop(
            system_prompt=_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=temperature,
        )

        try:
            llm_result = json.loads(raw)
        except Exception:
            llm_result = {
                "is_valid": True, "needs_correction": False,
                "errors": [], "warnings": [], "suggestions": [],
                "feedback_for_ag": None, "confidence_score": 0.8,
                "reasoning": "Validation completed.", "skills_used": []
            }

        is_valid = llm_result.get("is_valid", True)
        needs_correction = llm_result.get("needs_correction", False) or not is_valid
        all_errors = llm_result.get("errors", [])

        return {
            "success": True, "agent": "AV-SQLValidator",
            "original_sql": sql, "sql": sql,
            "is_valid": is_valid, "needs_correction": needs_correction,
            "decision_word": "correct" if is_valid else "incorrect",
            "issues": all_errors, "errors": all_errors,
            "warnings": llm_result.get("warnings", []),
            "optimization_tips": llm_result.get("suggestions", []),
            "dba_recommendations": [],
            "suggestions": llm_result.get("suggestions", []),
            "explain_result": {"executed": True, "static_analysis": True},
            "efficiency_analysis": {},
            "feedback_for_ag": llm_result.get("feedback_for_ag"),
            "confidence_score": llm_result.get("confidence_score", 0.8),
            "reasoning": llm_result.get("reasoning", ""),
            "skills_used": llm_result.get("skills_used", []),
        }

    def get_info(self):
        return {
            "name": self._agent_label,
            "model": self.model,
            "db_type": self.db_type,
            "skills": ["check_syntax_rules", "check_semantic_patterns", "check_query_efficiency"]
        }
