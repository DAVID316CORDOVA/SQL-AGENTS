"""
agents/MCP/server.py

Servidor MCP del sistema SQL-Agents. ENTRYPOINT minimalista.

La instancia `mcp` vive en `_mcp_instance.py` (no aqui) para que NO se duplique
cuando server.py se lanza como __main__ via subprocess. Si la instancia
viviera aqui, las tools harian `from agents.MCP.server import mcp` lo cual
carga server.py una SEGUNDA vez (con otro nombre) y las tools se
registrarian en otra instancia distinta a la que el server corre.

Las tools se definen en `tools/` y se registran al importar el paquete
(efecto secundario de los decoradores @mcp.tool()).

Para arrancar standalone (transporte stdio del protocolo MCP):
    venv/Scripts/python.exe -m agents.MCP.server
"""
import sys
from pathlib import Path

# Cuando server.py se ejecuta como subproceso, Python no conoce la raiz
# del proyecto. La agregamos para que los imports `agents.MCP.*` funcionen.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os

from agents.MCP._mcp_instance import mcp
from agents.MCP import tools  # noqa: F401  -- el import registra las tools


def main():
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        # Docker mode: HTTP/SSE — MCP como microservicio independiente.
        # host/port se configuran en FastMCP.__init__ via MCP_HOST/MCP_PORT.
        # mcp.run() en v1.27+ solo acepta transport y mount_path.
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
