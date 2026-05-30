"""
agents/MCP/tools

Modulos de tools del servidor MCP, organizados por consumidor:

  - ar_tools.py     : tools que consume el Agente Refinador (AR)
  - aps_tools.py    : tools que consume el Agente de Proximidad Semantica (APS)
  - shared_tools.py : tools de uso transversal (no especificas a un agente)

Importar este paquete registra todas las tools en la instancia compartida de
FastMCP definida en agents/MCP/_mcp_instance.py.
"""
from agents.MCP.tools import ar_tools, aps_tools, shared_tools

__all__ = ["ar_tools", "aps_tools", "shared_tools"]
