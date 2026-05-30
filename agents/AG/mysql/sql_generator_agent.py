"""
agents/AG/mysql/sql_generator_agent.py

AG - Agente Generador de SQL para MySQL 8.0.

Responsabilidad: recibe el schema reducido de APS (solo las tablas/columnas
relevantes para la pregunta) y genera una query SELECT valida para MySQL 8.0.
Si AV detecta errores en la query generada, AG recibe el feedback y corrige
la query en la siguiente iteracion del ciclo AG<->AV.

Skills (herramientas deterministas que el LLM puede invocar):
  - validate_sql_safety      : rechaza DML/DDL y SELECT */COUNT(*) prohibidos
  - fix_reserved_words       : agrega backticks a palabras reservadas MySQL
  - find_joins_among_tables  : devuelve FKs directas entre un conjunto de tablas
  - find_join_path           : BFS en el grafo de FKs para encontrar el camino
                               mas corto entre dos tablas (tablas puente incluidas)

El LLM usa las skills como oraculo: no inventa condiciones JOIN, sino que
llama a find_joins_among_tables para obtener los JOIN hints verificados.
"""

import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from llm_client import get_model, get_client_for_model
from agents.base_skill_agent import SkillAgent
from agents.AG.mysql import agentic_skills as _skills
from agents.AG.mysql import functions as _fn
from agents.AG.mysql.prompt import SYSTEM_PROMPT as _SYSTEM_PROMPT


class SQLGeneratorAgent(SkillAgent):
    """AG - Generador de SQL para MySQL 8.0."""

    def __init__(self):
        self.db_type = "mysql"
        self.model = get_model("AG", backend="mysql")
        self.client = get_client_for_model(self.model)
        self._agent_label = "AG-SQLGenerator-MYSQL"

    def _get_skills(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "validate_sql_safety",
                    "description": (
                        "Programmatic SQL safety check. "
                        "Verifies the query contains no DROP, DELETE, UPDATE, INSERT, "
                        "SELECT * or COUNT(*). "
                        "Always call this BEFORE returning the final SQL."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "The SQL query to verify"}
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
                        "Adds backticks to aliases or column names that clash with MySQL "
                        "reserved words (rank, status, name, type, level, date, etc.). "
                        "Call this after validate_sql_safety."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "The SQL query to fix"}
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
                        "Computes the shortest BFS path between TWO tables in the FK graph "
                        "and returns the full path plus ready-to-use JOIN hints. "
                        "Use this ONLY when two tables you need to JOIN have no direct FK "
                        "(bridge-table scenario). "
                        "If they share a direct FK, use find_joins_among_tables instead."
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
                        "Returns all direct FK relationships among a set of tables, "
                        "with JOIN hints ready to paste into the SQL. "
                        "Call this when you have more than one table and need to know "
                        "how to join them (no bridge tables needed)."
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
        return json.dumps({"error": f"Skill '{skill_name}' no reconocida en AG-MySQL"})

    def process(self, aps_result, feedback_from_av=None, previous_sql=None,
                last_successful_sql=None, temperature: float = 0.1):
        if not aps_result.get("success"):
            return {"success": False, "agent": self._agent_label,
                    "error": "No APS context", "sql": "",
                    "confidence_score": 0.0, "reasoning": "APS failed"}

        tables = aps_result.get("tables", {})
        joins  = aps_result.get("joins", [])
        intent = aps_result.get("original_intent", {})
        query  = aps_result.get("input_query", "")

        if not tables:
            return {"success": False, "agent": self._agent_label,
                    "error": "No tables", "sql": "",
                    "confidence_score": 0.0, "reasoning": "No tables returned by APS"}

        # Detectar si es una correccion: AV encontro errores y solicita re-generacion
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
            # Si el LLM devuelve JSON malformado, reintentar con temperatura=0
            # para forzar una respuesta mas determinista y bien estructurada
            if result is None:
                raw = self._run_agent_loop(
                    system_prompt=_SYSTEM_PROMPT,
                    user_message=user_message + "\n\nRESPOND ONLY WITH VALID JSON.",
                    temperature=0
                )
                result = self._parse_response(raw)
            if result is None:
                return {"success": False, "agent": self._agent_label,
                        "error": "Invalid LLM response after retry", "sql": "",
                        "confidence_score": 0.0, "reasoning": "Invalid LLM response"}

            sql = result.get("sql", "")

            # Aplicar fix_reserved_words como post-procesamiento determinista.
            # El LLM a veces omite backticks en palabras como `rank` o `status`;
            # esta funcion los agrega programaticamente sin otra llamada LLM.
            fix_result = _skills.fix_reserved_words(sql)
            sql = fix_result.get("sql", sql)

            reasoning = result.get("reasoning", "SQL generated")
            if is_correction:
                changes = result.get("changes_made", [])
                reasoning = f"[CORRECTION] {', '.join(changes) if changes else 'Regenerated'}\n{reasoning}"

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
                    "confidence_score": 0.0, "reasoning": f"Exception: {exc}"}

    def _build_prompt(self, query, tables, joins, intent, feedback=None, previous_sql=None,
                      last_successful_sql=None):
        # Serializar el schema reducido: solo las tablas y columnas que APS selecciono.
        # Incluir clave primaria y descripciones semanticas de columnas categoricas
        # para ayudar al LLM a elegir correctamente entre columnas similares.
        t = "=== SCHEMA ===\n"
        for name, info in tables.items():
            cols = info.get("all_columns", [])
            pk   = info.get("primary_key", "")
            cat_semantic = info.get("categorical_semantic", {})
            t += f"\nTABLE: {name}"
            if pk:
                t += f" (PK: {pk})"
            t += f"\n  COLUMNS: {', '.join(cols)}\n"
            # Descripcion semantica de columnas categoricas (ej. "country: pais de origen")
            # reduce la ambiguedad cuando hay columnas con nombres parecidos
            if cat_semantic:
                for col_name, desc in cat_semantic.items():
                    if desc and isinstance(desc, str):
                        t += f"  {col_name}: {desc}\n"

        # Los JOIN hints precomputados por APS (o por find_joins_among_tables)
        # evitan que el LLM invente condiciones de join incorrectas
        j = ""
        if joins:
            j = "\n=== AVAILABLE JOINS ===\n"
            for jj in joins:
                j += f"  - {jj.get('join_hint', '')}\n"

        # Incluir el intent estructurado de AR para dar mas contexto al tipo de query
        i = ""
        if intent:
            i = "\n=== INTENT ===\n"
            if intent.get("action"):
                i += f"  Action: {intent['action']}\n"
            if intent.get("entities"):
                i += f"  Entities: {', '.join(intent['entities'])}\n"

        # Modo correccion: AV rechazo el SQL anterior y provee feedback detallado.
        # El bloque CORRECTION REQUIRED indica al LLM exactamente que cambiar.
        fb = ""
        if feedback and previous_sql:
            fb = (
                f"\n=== CORRECTION REQUIRED ===\n"
                f"Previous SQL (with errors):\n{previous_sql}\n\n"
                f"Validator feedback:\n{feedback}\n\n"
                f"Fix the SQL by resolving the reported errors.\n"
            )

        # Encadenamiento de consultas conversacionales: si el usuario hace una pregunta
        # de seguimiento ("y de esos, cuantos son de berlin?"), se provee el SQL exitoso
        # anterior como base para que AG reutilice los mismos filtros y solo agregue los nuevos.
        chain = ""
        if last_successful_sql and not feedback:
            chain = (
                f"\n=== PREVIOUS SUCCESSFUL SQL ===\n"
                f"{last_successful_sql}\n\n"
                f"CHAINING RULE: Reuse the same filters and format from the previous SQL "
                f"as a base. Only add or modify what the new question requires. "
                f"Do NOT change the style of existing filters.\n"
            )

        return (
            f'QUERY: "{query}"\n\n=== ENGINE ===\n  MYSQL\n{t}{j}{i}{fb}{chain}\n'
            f"Generate the SQL, then call validate_sql_safety and fix_reserved_words "
            f"before returning your final JSON response."
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
