"""
agents/MCP/client.py

Cliente MCP — soporta dos transportes:

  stdio (local/dev): lanza el server como subprocess JSON-RPC.
  SSE   (Docker):    conecta al servidor HTTP via AGENT_MCP_URL.

Seleccion automatica: si AGENT_MCP_URL esta definido en el entorno
se usa SSE; de lo contrario stdio.

Convencion de nombres:
  - Funciones publicas: <name>_client    (lo que llaman los agentes)
  - Tools en server.py: <name>_server    (decoradas con @mcp.tool())
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parent.parent.parent

_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "agents.MCP.server"],
    cwd=str(ROOT),
)


async def _request_to_server(tool_name: str, args: dict) -> str:
    """Invoca una tool MCP. Usa SSE si AGENT_MCP_URL esta definido, sino stdio."""
    mcp_url = os.environ.get("AGENT_MCP_URL", "")

    async def _call(read, write) -> str:
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            content = result.content or []
            if not content:
                return ""
            if len(content) == 1:
                return getattr(content[0], "text", "") or ""
            # FastMCP 1.27+ puede serializar un list de dicts como un TextContent
            # por elemento. Intentamos reconstruir el JSON array desde los trozos.
            texts = [getattr(c, "text", "") or "" for c in content]
            # Caso 1: unico item no vacío
            non_empty = [t for t in texts if t.strip()]
            if len(non_empty) == 1:
                return non_empty[0]
            # Caso 2: cada texto es un JSON objeto/valor → reconstruir array
            items = []
            for t in non_empty:
                try:
                    parsed = json.loads(t)
                    if isinstance(parsed, list):
                        items.extend(parsed)
                    else:
                        items.append(parsed)
                except json.JSONDecodeError:
                    pass
            if items:
                return json.dumps(items)
            # Fallback: concatenar con saltos de línea (comportamiento anterior)
            return "\n".join(non_empty)

    if mcp_url:
        from mcp.client.sse import sse_client
        async with sse_client(mcp_url) as (read, write):
            return await _call(read, write)
    else:
        async with stdio_client(_SERVER_PARAMS) as (read, write):
            return await _call(read, write)


# ─── Wrappers publicos para los agentes ────────────────────────────────────
# Cada wrapper:
#   1. Recibe argumentos como kwargs naturales.
#   2. Llama via asyncio.run() a _request_to_server con el nombre _server.
#   3. Parsea la respuesta segun el tipo de retorno.

def get_database_description_client(db_name: str) -> str:
    return asyncio.run(_request_to_server(
        "get_database_description_server", {"db_name": db_name}))


def list_available_databases_client() -> list:
    raw = asyncio.run(_request_to_server("list_available_databases_server", {}))
    return [s.strip() for s in raw.split("\n") if s.strip()] if raw else []


def find_join_path_client(from_table: str, to_table: str, db_name: str,
                           max_hops: int = 3,
                           backend: str = "mysql") -> dict:
    raw = asyncio.run(_request_to_server("find_join_path_server", {
        "from_table": from_table, "to_table": to_table,
        "db_name": db_name, "max_hops": max_hops, "backend": backend,
    }))
    return json.loads(raw) if raw else {}


def find_column_for_concept_client(concept: str, db_name: str,
                                    tables: list = None,
                                    threshold: float = 0.6,
                                    top_n: int = 5,
                                    backend: str = "mysql") -> dict:
    raw = asyncio.run(_request_to_server("find_column_for_concept_server", {
        "concept": concept, "db_name": db_name, "tables": tables,
        "threshold": threshold, "top_n": top_n, "backend": backend,
    }))
    return json.loads(raw) if raw else {}


def get_table_relationships_client(table: str, db_name: str,
                                    backend: str = "mysql") -> dict:
    raw = asyncio.run(_request_to_server(
        "get_table_relationships_server",
        {"table": table, "db_name": db_name, "backend": backend}))
    return json.loads(raw) if raw else {}


def get_pk_client(table: str, db_name: str,
                  backend: str = "mysql") -> dict:
    raw = asyncio.run(_request_to_server(
        "get_pk_server",
        {"table": table, "db_name": db_name, "backend": backend}))
    return json.loads(raw) if raw else {}


def classify_table_role_client(table: str, db_name: str,
                                backend: str = "mysql") -> dict:
    raw = asyncio.run(_request_to_server(
        "classify_table_role_server",
        {"table": table, "db_name": db_name, "backend": backend}))
    return json.loads(raw) if raw else {}


def search_tables_client(query: str, db_name: str, n: int = 5,
                          backend: str = "mysql") -> list:
    raw = asyncio.run(_request_to_server("search_tables_server", {
        "query": query, "db_name": db_name, "n": n, "backend": backend,
    }))
    return json.loads(raw) if raw else []


def search_columns_client(query: str, db_name: str, tables: list = None,
                           n: int = 10, backend: str = "mysql") -> list:
    raw = asyncio.run(_request_to_server("search_columns_server", {
        "query": query, "db_name": db_name,
        "tables": tables, "n": n, "backend": backend,
    }))
    return json.loads(raw) if raw else []


