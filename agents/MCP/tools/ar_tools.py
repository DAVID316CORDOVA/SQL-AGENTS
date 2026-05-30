"""
agents/MCP/tools/ar_tools.py

Tools MCP que consume el Agente Refinador (AR).

El AR las invoca al inicio de su pipeline para tener contexto de dominio
sobre la base de datos activa antes de validar la pregunta del usuario.

Nota: get_database_description tambien la consume el APS dentro del LLM en
zona gris (cuando debe decidir compatibilidad de un concepto). Vive aqui
porque su consumidor principal y su proposito original es contextualizar
al AR.
"""
from agents.MCP._mcp_instance import mcp, resolve_description_path


@mcp.tool()
def get_database_description_server(db_name: str) -> str:
    """
    Obtiene la descripcion de dominio de una base de datos.

    Devuelve un texto en formato Markdown que describe de que trata la base
    de datos: su proposito, los usuarios tipicos, las preguntas frecuentes
    y las caracteristicas relevantes del dominio. Este texto sirve como
    contexto para que un agente entienda sobre que datos esta operando.

    Args:
        db_name: identificador de la base de datos. Corresponde al valor de
                 ACTIVE_DATASET en config.py.

    Returns:
        Texto Markdown con la descripcion de la base de datos. Si no hay
        descripcion registrada para el identificador recibido, devuelve un
        mensaje indicandolo en lugar de fallar.
    """
    path = resolve_description_path(db_name)
    if not path.exists():
        return (
            f"(sin descripcion disponible para '{db_name}'. "
            f"Se esperaba el archivo {path.name} en db_descriptions/)"
        )
    return path.read_text(encoding="utf-8")
