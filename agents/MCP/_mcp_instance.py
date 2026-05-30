"""
agents/MCP/_mcp_instance.py

Componentes compartidos por tools y server. Necesario porque server.py
se ejecuta como __main__ cuando se lanza por subprocess; si la instancia
`mcp` viviera en server.py, las tools la importarian via
`agents.MCP.server` (segunda carga) y se registrarian en otra instancia.

Vivir en _mcp_instance.py garantiza una sola instancia sin importar como se
lance server.py.
"""
import json
from functools import lru_cache
from pathlib import Path

import networkx as nx
from mcp.server.fastmcp import FastMCP


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DESCRIPTIONS_DIR = HERE / "db_descriptions"
METADATA_DIR = HERE / "metadata"   # ahora vive dentro de agents/MCP/


# Instancia unica del servidor. Todos los modulos de tools la comparten.
# host/port se leen de env vars para que funcione tanto en local (stdio)
# como en Docker (SSE en 0.0.0.0:8010).
import os as _os
mcp = FastMCP(
    "sql-agents-mcp",
    host=_os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(_os.environ.get("MCP_PORT", "8010")),
)


# ─── Resolvers de paths ────────────────────────────────────────────────────

def resolve_description_path(db_name: str) -> Path:
    """Convierte 'bird:financial' en db_descriptions/bird_financial.md"""
    if ":" in db_name:
        family, db_id = db_name.split(":", 1)
        filename = f"{family}_{db_id}.md"
    else:
        filename = f"{db_name}.md"
    return DESCRIPTIONS_DIR / filename


def resolve_schema_path(db_name: str, backend: str = "mysql") -> Path:
    """Convierte ('bird:financial', 'mysql') en
    metadata/mysql/bird/financial/schema.json.

    El schema (DDL) puede divergir entre backends si se agregan/modifican
    tablas en uno sin actualizar el otro, por eso se separa fisicamente."""
    backend = (backend or "mysql").lower()
    if ":" in db_name:
        family, db_id = db_name.split(":", 1)
        return METADATA_DIR / backend / family / db_id / "schema.json"
    return METADATA_DIR / backend / db_name / "schema.json"


def resolve_diccionario_path(db_name: str, backend: str = "mysql") -> Path:
    """Convierte ('bird:financial', 'mysql') en
    metadata/mysql/bird/financial/diccionario_datos.json."""
    backend = (backend or "mysql").lower()
    if ":" in db_name:
        family, db_id = db_name.split(":", 1)
        return METADATA_DIR / backend / family / db_id / "diccionario_datos.json"
    return METADATA_DIR / backend / db_name / "diccionario_datos.json"


def resolve_vector_db_path(db_name: str, backend: str = "mysql") -> Path:
    """Devuelve el path al directorio ChromaDB del dataset indicado.

    Estructura: agents/APS/<backend>/chroma/<family>/<db_id>/
    Para datasets sin family (demo_db, wikisql) la ruta es
    agents/APS/<backend>/chroma/<db_name>/.
    """
    backend = (backend or "mysql").lower()
    base = ROOT / "agents" / "APS" / backend / "chroma"
    if ":" in db_name:
        family, db_id = db_name.split(":", 1)
        return base / family / db_id
    return base / db_name


# ─── Loaders cacheados ─────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def load_schema(db_name: str, backend: str = "mysql") -> dict:
    """Carga el schema.json de la BD especificada para el backend dado.
    Cachea por (db_name, backend) — cada combinacion tiene su propio archivo."""
    path = resolve_schema_path(db_name, backend)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=64)
def build_graph(db_name: str, backend: str = "mysql") -> nx.Graph:
    """
    Grafo no dirigido con tablas como nodos y FKs como aristas.
    Cada arista guarda la metadata de la relacion para reconstruir las
    condiciones de JOIN al recorrer un camino.
    """
    schema = load_schema(db_name, backend)
    g = nx.Graph()
    for tname in schema.get("available_entities", {}).keys():
        g.add_node(tname)
    for rel in schema.get("relationships", []):
        ft, tt = rel.get("from_table"), rel.get("to_table")
        if ft and tt:
            g.add_edge(ft, tt, rel=rel)
    return g
