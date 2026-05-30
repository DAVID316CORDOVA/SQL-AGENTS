"""
registry.py

Control de acceso centralizado para las skills (herramientas) del sistema SQL-Agents.

Problema que resuelve: en un sistema multi-agente donde cada agente puede invocar
herramientas via tool-calling del LLM, existe riesgo de prompt injection: un LLM
adversarialmente instruido podria intentar llamar skills de otro agente (ej. AS
intentando llamar search_tables, que pertenece a APS). El registry bloquea
cualquier invocacion no autorizada antes de que llegue al handler.

Funcionamiento:
  1. Al arrancar el sistema, setup_registry() registra las 13 skills del pipeline
     y asigna a cada agente exactamente las skills que tiene autorización de usar.
  2. En cada invocacion de skill, _dispatch_skill_safely() (base_skill_agent.py)
     consulta can_use() antes de ejecutar el handler real.
  3. Si el agente no tiene permiso, se lanza PermissionError y queda en el log
     de auditoria, recuperable con get_denied_executions().

Nota sobre handlers: en setup_registry() todos los handlers son `lambda **kw: None`.
El registry aqui sirve para control de acceso y definicion de schemas OpenAI.
Los handlers reales (la implementacion concreta de cada skill) viven en los
archivos de skills de cada agente: agents/APS/mysql/skills.py, agents/AV/mysql/
agentic_skills.py, etc. Son los que se pasan a _execute_skill() dentro de cada agente.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any


@dataclass
class ToolDefinition:
    """
    Descriptor de una skill del sistema.

    name               : identificador unico de la skill (debe coincidir con el nombre
                         que el LLM usa en tool-calling)
    description        : texto legible para logs y depuracion
    handler            : funcion Python que implementa la skill (puede ser placeholder
                         lambda en el registry global; el handler real vive en el agente)
    requires_input     : si la skill necesita argumentos para ejecutarse
    allowed_operations : tipo de operacion que realiza ("read" = solo lectura,
                         nunca escritura a BD)
    openai_schema      : schema JSON en formato OpenAI tool-calling; si se define,
                         el agente puede exponer esta skill al LLM via el parametro
                         'tools' de la API
    """
    name: str
    description: str
    handler: Callable
    requires_input: bool = True
    allowed_operations: List[str] = field(default_factory=lambda: ["read"])
    openai_schema: Optional[dict] = None


class ToolRegistry:
    """
    Registro central de skills con control de acceso por agente.

    Garantias de seguridad que implementa:
      - Ningun agente puede invocar skills de otro agente.
      - Ningun agente tiene acceso de escritura a la BD (todas las skills son read-only).
      - Toda invocacion — autorizada o denegada — queda en el log de auditoria.
      - Si un LLM intenta invocar una skill no registrada o no autorizada,
        se lanza PermissionError antes de ejecutar cualquier codigo.
    """

    def __init__(self):
        # Diccionario de todas las skills registradas: nombre → ToolDefinition
        self._tools: Dict[str, ToolDefinition] = {}
        # Permisos por agente: nombre_agente → lista de nombres de skills permitidas
        self._agent_permissions: Dict[str, List[str]] = {}
        # Log de auditoria: cada invocacion (OK, DENIED o ERROR) queda registrada
        self._execution_log: List[Dict] = []

    def register_tool(self, tool: ToolDefinition):
        """Registra una skill en el sistema. Falla si el nombre ya existe (evita solapamiento)."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' ya esta registrada")
        self._tools[tool.name] = tool

    def allow_agent_tools(self, agent_name: str, tool_names: List[str]):
        """
        Define exactamente que skills puede invocar un agente.
        Valida que todas las skills existan antes de asignar permisos,
        para detectar errores de configuracion en el arranque.
        """
        for tn in tool_names:
            if tn not in self._tools:
                raise ValueError(f"Tool '{tn}' no esta registrada — registrarla antes de asignar permisos")
        self._agent_permissions[agent_name] = tool_names

    def can_use(self, agent_name: str, tool_name: str) -> bool:
        """Consulta de permiso: True si el agente tiene la skill en su lista autorizada."""
        allowed = self._agent_permissions.get(agent_name, [])
        return tool_name in allowed

    def execute(self, agent_name: str, tool_name: str, **kwargs) -> Any:
        """
        Ejecuta una skill con validacion de permisos previa.

        Flujo:
          1. Verificar que la skill existe en el registry.
          2. Verificar que el agente tiene permiso para usarla.
          3. Si hay permiso: ejecutar handler y registrar OK en el log.
          4. Si no hay permiso: registrar DENIED en el log y lanzar PermissionError.
          5. Si el handler falla: registrar ERROR en el log y re-lanzar la excepcion.
        """
        # La skill debe existir en el registry antes de verificar permisos
        if tool_name not in self._tools:
            raise PermissionError(f"Tool '{tool_name}' no existe en el registry")

        if not self.can_use(agent_name, tool_name):
            allowed = self._agent_permissions.get(agent_name, [])
            self._execution_log.append({
                "agent":  agent_name,
                "tool":   tool_name,
                "status": "DENIED",
                "reason": f"No autorizado. Skills permitidas para '{agent_name}': {allowed}"
            })
            raise PermissionError(
                f"Agente '{agent_name}' no tiene permiso para usar '{tool_name}'. "
                f"Skills permitidas: {allowed}"
            )

        tool = self._tools[tool_name]
        try:
            result = tool.handler(**kwargs)
            self._execution_log.append({
                "agent":       agent_name,
                "tool":        tool_name,
                "status":      "OK",
                "kwargs_keys": list(kwargs.keys())
            })
            return result
        except Exception as e:
            self._execution_log.append({
                "agent":  agent_name,
                "tool":   tool_name,
                "status": "ERROR",
                "error":  str(e)
            })
            raise

    def get_agent_tools(self, agent_name: str) -> List[ToolDefinition]:
        """Retorna los objetos ToolDefinition de las skills autorizadas para un agente."""
        allowed = self._agent_permissions.get(agent_name, [])
        return [self._tools[tn] for tn in allowed if tn in self._tools]

    def get_agent_tool_descriptions(self, agent_name: str) -> str:
        """Texto de descripcion de las skills de un agente, util para incluir en prompts."""
        tools = self.get_agent_tools(agent_name)
        if not tools:
            return "Sin herramientas asignadas."
        return "\n".join(f"- {t.name}: {t.description}" for t in tools)

    def get_openai_tools(self, agent_name: str) -> List[dict]:
        """
        Schemas de las skills del agente en formato OpenAI tool-calling.
        Solo incluye skills que tienen openai_schema definido.
        Se usa en el agentic loop para pasar 'tools' a la API de OpenAI/Anthropic.
        """
        allowed_tools = self.get_agent_tools(agent_name)
        return [
            {"type": "function", "function": t.openai_schema}
            for t in allowed_tools
            if t.openai_schema is not None
        ]

    def get_execution_log(self) -> List[Dict]:
        """Log completo de invocaciones (OK + DENIED + ERROR) para auditoria."""
        return self._execution_log.copy()

    def get_denied_executions(self) -> List[Dict]:
        """Solo las invocaciones denegadas. Una lista no vacia indica posible prompt injection."""
        return [e for e in self._execution_log if e["status"] == "DENIED"]

    def get_info(self) -> Dict:
        """Resumen del estado del registry para depuracion y tests."""
        return {
            "total_tools":       len(self._tools),
            "tools":             list(self._tools.keys()),
            "agents":            {k: v for k, v in self._agent_permissions.items()},
            "total_executions":  len(self._execution_log),
            "denied_executions": len(self.get_denied_executions())
        }


# ================================================================
# CONFIGURACION GLOBAL DEL REGISTRY
# ================================================================

def setup_registry() -> ToolRegistry:
    """
    Inicializa el registry con las 13 skills del sistema y sus permisos por agente.
    Se llama una sola vez al arrancar el sistema (graph.py o main.py).

    Los handlers son `lambda **kw: None` porque el registry aqui cumple dos roles:
      1. Control de acceso: can_use() verifica permisos antes de ejecutar.
      2. Definicion de schemas OpenAI: openai_schema permite exponer las skills al LLM.
    Los handlers reales (implementacion concreta) viven en los archivos de skills
    de cada agente y son invocados por _execute_skill() dentro del agente.
    """
    registry = ToolRegistry()

    # ── AR (Refinador) ────────────────────────────────────────────
    # AR es un agente LLM puro: no invoca skills via tool-calling en produccion.
    # check_forbidden_keywords actua como gate procedural en nodes.py (ar_node)
    # para verificar permisos del agente antes de procesarlo.
    registry.register_tool(ToolDefinition(
        name="check_forbidden_keywords",
        description="Verifica si la pregunta contiene operaciones prohibidas (DELETE, DROP) o es conversacion sin datos",
        handler=lambda **kw: None,
        allowed_operations=["read"],
        openai_schema={
            "name": "check_forbidden_keywords",
            "description": "Verificacion rapida de palabras clave prohibidas. Llama esto PRIMERO antes de analizar la intencion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Pregunta del usuario a verificar"}
                },
                "required": ["question"]
            }
        }
    ))

    # ── APS (Schema Matcher) ──────────────────────────────────────
    # APS es el unico agente que consulta ChromaDB directamente.
    # Sus dos skills son las unicas que realizan busqueda vectorial.
    registry.register_tool(ToolDefinition(
        name="search_tables",
        description="Busca tablas relevantes por similitud semantica en ChromaDB",
        handler=lambda **kw: None,
        allowed_operations=["read"],
        openai_schema={
            "name": "search_tables",
            "description": "Busca en ChromaDB las tablas mas relevantes usando similitud semantica. Siempre llama esto primero.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Terminos derivados de la pregunta del usuario"},
                    "n":     {"type": "integer", "description": "Numero maximo de tablas a retornar (default 5)", "default": 5}
                },
                "required": ["query"]
            }
        }
    ))
    registry.register_tool(ToolDefinition(
        name="search_columns",
        description="Busca columnas relevantes por similitud semantica, opcionalmente filtradas por tabla",
        handler=lambda **kw: None,
        allowed_operations=["read"],
        openai_schema={
            "name": "search_columns",
            "description": "Busca columnas relevantes para la consulta. Llama esto despues de search_tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string", "description": "Terminos de busqueda para las columnas"},
                    "tables": {"type": "array", "items": {"type": "string"}, "description": "Tablas para filtrar (opcional)"},
                    "n":      {"type": "integer", "description": "Numero maximo de columnas (default 10)", "default": 10}
                },
                "required": ["query"]
            }
        }
    ))

    # ── AG (Generador SQL) ────────────────────────────────────────
    # AG invoca estas dos skills en cada generacion de SQL.
    # find_join_path y find_joins_among_tables son helpers deterministas
    # llamados directamente desde AG.process() via functions.py, NO via
    # tool-calling del LLM, por eso no se registran aqui.
    registry.register_tool(ToolDefinition(
        name="validate_sql_safety",
        description="Verificacion programatica de seguridad: rechaza DML/DDL, SELECT *, COUNT(*)",
        handler=lambda **kw: None,
        allowed_operations=["read"],
        openai_schema={
            "name": "validate_sql_safety",
            "description": "Valida que el SQL sea solo SELECT sin operaciones peligrosas. Llama siempre antes de retornar el SQL final.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "Consulta SQL a verificar"}
                },
                "required": ["sql"]
            }
        }
    ))
    registry.register_tool(ToolDefinition(
        name="fix_reserved_words",
        description="Agrega backticks (MySQL) o comillas dobles (PostgreSQL) a palabras reservadas usadas como identificadores",
        handler=lambda **kw: None,
        allowed_operations=["read"],
        openai_schema={
            "name": "fix_reserved_words",
            "description": "Corrige el SQL agregando backticks a aliases que coinciden con palabras reservadas (rank, status, name, type...). Llama despues de validate_sql_safety.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "Consulta SQL a corregir"}
                },
                "required": ["sql"]
            }
        }
    ))

    # ── AV (Validador SQL) ────────────────────────────────────────
    # AV invoca estas cuatro skills en orden segun el tipo de SQL recibido.
    # check_syntax_rules siempre primero; las demas segun el contenido del SQL.
    registry.register_tool(ToolDefinition(
        name="check_syntax_rules",
        description="Verificacion de reglas de sintaxis SQL: SELECT *, COUNT(*), DML/DDL, JOIN sin ON",
        handler=lambda **kw: None,
        allowed_operations=["read"],
        openai_schema={
            "name": "check_syntax_rules",
            "description": "Verificacion programatica rapida de sintaxis SQL. Llama esto primero siempre.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "Consulta SQL a verificar"}
                },
                "required": ["sql"]
            }
        }
    ))
    registry.register_tool(ToolDefinition(
        name="check_semantic_patterns",
        description="Detecta patrones SQL problematicos: window functions en WHERE, LIMIT global sin PARTITION BY",
        handler=lambda **kw: None,
        allowed_operations=["read"],
        openai_schema={
            "name": "check_semantic_patterns",
            "description": "Detecta patrones SQL semanticamente incorrectos. Invocar cuando el SQL tiene LIMIT, HAVING o window functions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql":   {"type": "string"},
                    "query": {"type": "string", "description": "Pregunta original del usuario"}
                },
                "required": ["sql", "query"]
            }
        }
    ))
    registry.register_tool(ToolDefinition(
        name="check_query_efficiency",
        description="Analisis estatico de calidad: producto cartesiano, exceso de JOINs (>4), DISTINCT redundante con GROUP BY",
        handler=lambda **kw: None,
        allowed_operations=["read"],
        openai_schema={
            "name": "check_query_efficiency",
            "description": "Detecta problemas de eficiencia estructural. Invocar cuando el SQL tiene multiples tablas o JOINs complejos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"}
                },
                "required": ["sql"]
            }
        }
    ))
    registry.register_tool(ToolDefinition(
        name="check_query_performance",
        description="Heuristicas de rendimiento sin ejecutar: wildcard inicial LIKE, funciones en WHERE, subqueries correlacionadas",
        handler=lambda **kw: None,
        allowed_operations=["read"],
        openai_schema={
            "name": "check_query_performance",
            "description": "Predice si la consulta sera lenta. Invocar cuando hay LIKE, funciones en WHERE o subqueries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql":              {"type": "string"},
                    "table_row_counts": {"type": "object", "description": "Diccionario {tabla: filas} opcional"}
                },
                "required": ["sql"]
            }
        }
    ))

    # ── AE (Explicador) ───────────────────────────────────────────
    # AE usa una sola skill: formatear el SQL para mostrarlo legible al usuario.
    registry.register_tool(ToolDefinition(
        name="format_sql_readable",
        description="Formatea SQL con saltos de linea ante cada keyword principal para mostrarlo al usuario",
        handler=lambda **kw: None,   # handler real en agents/AE/skills.py
        allowed_operations=["read"],
        openai_schema={
            "name": "format_sql_readable",
            "description": "Formatea el SQL con saltos de linea ante SELECT, FROM, WHERE, JOIN, GROUP BY, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL crudo a formatear"}
                },
                "required": ["sql"]
            }
        }
    ))

    # ── AS (Sustentador) ──────────────────────────────────────────
    # AS solo puede acceder al razonamiento de los agentes del turno actual.
    # No puede invocar search_tables ni ninguna skill de otro agente.
    registry.register_tool(ToolDefinition(
        name="get_agent_reasoning",
        description="Recupera el razonamiento detallado de un agente del pipeline del turno actual",
        handler=lambda **kw: None,   # handler real en agents/AS/skills.py
        allowed_operations=["read"],
        openai_schema={
            "name": "get_agent_reasoning",
            "description": "Extrae razonamiento de un agente especifico (AR/APS/AG/AV) del turno actual.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": ["AR", "APS", "AG", "AV"],
                        "description": "Nombre del agente cuyo razonamiento recuperar"
                    }
                },
                "required": ["agent"]
            }
        }
    ))

    # ── Orquestador (memoria) ─────────────────────────────────────
    # Estas no son skills agentic: son funciones procedurales que el orquestador
    # llama directamente en memory_functions.py, antes de activar el pipeline.
    # Se registran aqui para que nodes.py pueda verificar permisos via registry.execute().
    registry.register_tool(ToolDefinition(
        name="check_short_term_memory",
        description="Busca en cache RAM de la sesion actual (funcion procedural del orquestador, no skill agentic)",
        handler=lambda **kw: None,
        allowed_operations=["read"]
    ))
    registry.register_tool(ToolDefinition(
        name="check_long_term_memory",
        description="Busca en ChromaDB persistente entre sesiones (funcion procedural del orquestador, no skill agentic)",
        handler=lambda **kw: None,
        allowed_operations=["read"]
    ))

    # ================================================================
    # MATRIZ DE PERMISOS — un agente = exactamente sus skills autorizadas
    # ================================================================
    registry.allow_agent_tools("AR",  ["check_forbidden_keywords"])
    registry.allow_agent_tools("APS", ["search_tables", "search_columns"])
    registry.allow_agent_tools("AG",  ["validate_sql_safety", "fix_reserved_words"])
    registry.allow_agent_tools("AV",  ["check_syntax_rules", "check_semantic_patterns",
                                        "check_query_efficiency", "check_query_performance"])
    registry.allow_agent_tools("AE",  ["format_sql_readable"])
    registry.allow_agent_tools("AS",  ["get_agent_reasoning"])
    registry.allow_agent_tools("orchestrator", ["check_short_term_memory",
                                                "check_long_term_memory"])

    # ================================================================
    # GARANTIAS DE SEGURIDAD
    # ================================================================
    # 1. Ninguna skill tiene permiso de escritura a BD (todas son read-only).
    # 2. Ningun agente puede invocar skills de otro agente (matriz cerrada).
    # 3. Si un LLM recibe un prompt injection que intente invocar una skill no
    #    autorizada, _dispatch_skill_safely() (base_skill_agent.py) llama can_use()
    #    y lanza PermissionError sin ejecutar ningun codigo. El intento queda en
    #    get_denied_executions() para auditoria posterior.
    # 4. find_join_path y find_joins_among_tables son helpers deterministicos
    #    llamados directamente desde AG.process() via functions.py, fuera del
    #    bucle agentic — no necesitan pasar por el registry.

    return registry
