"""
agents/AG/postgres/sql_generator_agent.py

AG - Agente Generador de SQL especifico para PostgreSQL 14+.

Skills:
  - validate_sql_safety : verificacion programatica de seguridad
  - fix_reserved_words  : agrega comillas dobles a palabras reservadas PostgreSQL

Ubicacion: agents/AG/postgres/sql_generator_agent.py
"""

import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from llm_client import get_model, get_client_for_model
from agents.base_skill_agent import SkillAgent
from agents.AG.postgres import agentic_skills as _skills
from agents.AG.postgres import functions as _fn
from agents.AG.postgres.prompt import SYSTEM_PROMPT as _SYSTEM_PROMPT


class SQLGeneratorAgent(SkillAgent):
    """AG - Generador de SQL para PostgreSQL 14+."""

    def __init__(self):
        self.db_type = "postgres"
        self.model = get_model("AG", backend="postgres")
        self.client = get_client_for_model(self.model)
        self._agent_label = "AG-SQLGenerator-POSTGRES"

    def _get_skills(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "validate_sql_safety",
                    "description": (
                        "Verificacion programatica de seguridad del SQL. "
                        "Verifica que no contenga DROP, DELETE, UPDATE, INSERT, "
                        "SELECT * ni COUNT(*). "
                        "Invoca esto siempre ANTES de retornar el SQL final."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "La consulta SQL a verificar"}
                        },
                        "required": ["sql"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fix_reserved_words",
                    "description": (
                        "Agrega comillas dobles a aliases que coincidan con palabras reservadas "
                        "de PostgreSQL (user, order, group, rank, etc.). "
                        "Invoca esto despues de validate_sql_safety."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "La consulta SQL a corregir"}
                        },
                        "required": ["sql"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_join_path",
                    "description": (
                        "Calcula el camino mas corto (BFS) entre DOS tablas en el grafo "
                        "de claves foraneas del esquema y devuelve el path completo + "
                        "los hints de JOIN. Usalo SOLO cuando dos tablas que necesitas "
                        "JOINear no comparten FK directa (caso 'tabla puente'). "
                        "Si comparten FK directa, usa find_joins_among_tables."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from_table": {"type": "string"},
                            "to_table":   {"type": "string"},
                            "max_hops":   {"type": "integer", "default": 3}
                        },
                        "required": ["from_table", "to_table"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_joins_among_tables",
                    "description": (
                        "Devuelve todas las relaciones FK directas que existen entre un "
                        "conjunto de tablas, con los hints de JOIN listos para pegar en "
                        "el SQL. Invocalo cuando tengas mas de una tabla y necesites "
                        "saber como unirlas (sin tablas puente)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tables": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["tables"]
                    }
                }
            }
        ]

    def _execute_skill(self, skill_name: str, args: dict) -> str:
        if skill_name == "validate_sql_safety":
            return json.dumps(_skills.validate_sql_safety(args.get("sql", "")))
        elif skill_name == "fix_reserved_words":
            return json.dumps(_skills.fix_reserved_words(args.get("sql", "")))
        elif skill_name == "find_join_path":
            return json.dumps(_fn.find_join_path_skill(
                args.get("from_table", ""),
                args.get("to_table", ""),
                int(args.get("max_hops", 3)),
            ))
        elif skill_name == "find_joins_among_tables":
            return json.dumps(_fn.find_joins_among_tables(args.get("tables", [])))
        return json.dumps({"error": f"Skill '{skill_name}' no reconocida en AG-PostgreSQL"})

    def process(self, aps_result, feedback_from_av=None, previous_sql=None,
                last_successful_sql=None, temperature: float = 0.1):
        if not aps_result.get("success"):
            return {"success": False, "agent": self._agent_label,
                    "error": "Sin contexto de APS", "sql": "",
                    "confidence_score": 0.0, "reasoning": "APS fallo"}

        tables = aps_result.get("tables", {})
        joins = aps_result.get("joins", [])
        intent = aps_result.get("original_intent", {})
        query = aps_result.get("input_query", "")

        if not tables:
            return {"success": False, "agent": self._agent_label,
                    "error": "Sin tablas", "sql": "",
                    "confidence_score": 0.0, "reasoning": "No hay tablas"}

        is_correction = bool(feedback_from_av and previous_sql)
        user_message = self._build_prompt(
            query, tables, joins, intent,
            feedback=feedback_from_av if is_correction else None,
            previous_sql=previous_sql if is_correction else None,
            last_successful_sql=last_successful_sql
        )

        try:
            raw = self._run_agent_loop(
                system_prompt=_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=temperature
            )
            result = self._parse_response(raw)
            if result is None:
                raw = self._run_agent_loop(
                    system_prompt=_SYSTEM_PROMPT,
                    user_message=user_message + "\n\nRESPONDE UNICAMENTE CON JSON VALIDO.",
                    temperature=0
                )
                result = self._parse_response(raw)
            if result is None:
                return {"success": False, "agent": self._agent_label,
                        "error": "Respuesta invalida del LLM tras reintento", "sql": "",
                        "confidence_score": 0.0, "reasoning": "Respuesta invalida del LLM"}

            sql = result.get("sql", "")

            fix_result = _skills.fix_reserved_words(sql)
            sql = fix_result.get("sql", sql)

            reasoning = result.get("reasoning", "SQL generado")
            if is_correction:
                changes = result.get("changes_made", [])
                reasoning = f"[CORRECCION] {', '.join(changes) if changes else 'Regenerado'}\n{reasoning}"

            return {
                "success": True,
                "agent": self._agent_label,
                "input_query": query,
                "sql": sql,
                "db_type": self.db_type,
                "confidence_score": result.get("confidence_score", 0.85),
                "reasoning": reasoning,
                "strategy": result.get("strategy", "unknown"),
                "tables_used": list(tables.keys()),
                "is_correction": is_correction,
                "changes_made": result.get("changes_made", [])
            }

        except Exception as exc:
            return {"success": False, "agent": self._agent_label,
                    "error": str(exc), "sql": "",
                    "confidence_score": 0.0, "reasoning": f"Error: {exc}"}

    def _build_prompt(self, query, tables, joins, intent, feedback=None, previous_sql=None,
                      last_successful_sql=None):
        t = "=== SCHEMA ===\n"
        for name, info in tables.items():
            cols = info.get("all_columns", [])
            pk = info.get("primary_key", "")
            cat_semantic = info.get("categorical_semantic", {})
            t += f"\nTABLA: {name}"
            if pk:
                t += f" (PK: {pk})"
            t += f"\n  COLUMNAS: {', '.join(cols)}\n"
            if cat_semantic:
                for col_name, desc in cat_semantic.items():
                    if desc and isinstance(desc, str):
                        t += f"  {col_name}: {desc}\n"

        j = ""
        if joins:
            j = "\n=== JOINS DISPONIBLES ===\n"
            for jj in joins:
                j += f"  - {jj.get('join_hint', '')}\n"

        i = ""
        if intent:
            i = "\n=== INTENT ===\n"
            if intent.get("action"):
                i += f"  Accion: {intent['action']}\n"
            if intent.get("entities"):
                i += f"  Entidades: {', '.join(intent['entities'])}\n"

        fb = ""
        if feedback and previous_sql:
            fb = (
                f"\n=== CORRECCION REQUERIDA ===\n"
                f"SQL anterior (con errores):\n{previous_sql}\n\n"
                f"Feedback del validador:\n{feedback}\n\n"
                f"Corrige el SQL resolviendo los errores indicados.\n"
            )

        chain = ""
        if last_successful_sql and not feedback:
            chain = (
                f"\n=== SQL EXITOSO ANTERIOR ===\n"
                f"{last_successful_sql}\n\n"
                f"REGLA DE ENCADENAMIENTO: Reutiliza los mismos filtros y formato "
                f"del SQL anterior como base. Solo agrega o modifica lo que la nueva "
                f"pregunta pide. NO cambies el estilo de filtros existentes.\n"
            )

        return (
            f'CONSULTA: "{query}"\n\n=== MOTOR ===\n  POSTGRESQL\n{t}{j}{i}{fb}{chain}\n'
            f"Genera el SQL, luego usa validate_sql_safety y fix_reserved_words "
            f"antes de retornar tu respuesta final en JSON."
        )

    def _parse_response(self, raw: str):
        if not raw or not raw.strip():
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def get_info(self):
        return {
            "name": self._agent_label,
            "model": self.model,
            "db_type": self.db_type,
            "skills": ["validate_sql_safety", "fix_reserved_words"]
        }
