# agents/MCP

Servidor [Model Context Protocol](https://modelcontextprotocol.io) que expone
descripciones de dominio de las bases de datos a los agentes del sistema.
Lo consume principalmente el **Agente Refinador (AR)** para validar mejor
preguntas con terminologia especifica de cada base de datos.

---

## Estructura

```
agents/MCP/
├── __init__.py
├── README.md                          ← este archivo
├── server.py                          ← servidor MCP (expone tools)
├── client.py                          ← cliente que el AR usa para llamar al servidor
├── test_mcp.py                        ← prueba end-to-end de la conexion
└── db_descriptions/
    ├── demo_db.md
    ├── bird_financial.md
    └── bird_european_football_2.md
```

---

## Por que existen dos archivos: server.py y client.py

MCP es un protocolo **cliente-servidor** (similar conceptualmente a HTTP, pero
sobre stdio o websockets en vez de HTTP). En todo protocolo cliente-servidor
hay siempre dos lados:

- **`server.py`** — el lado que **expone** las herramientas. Define las
  funciones disponibles con `@mcp.tool()` y se queda esperando que alguien
  las invoque. No hace nada por si solo: necesita ser arrancado y necesita
  que un cliente le envie peticiones.

- **`client.py`** — el lado que **consume** las herramientas. Es la pieza que
  el agente (AR) usa para pedir la descripcion de una base de datos.
  Internamente arranca al servidor como subproceso, abre una sesion MCP,
  invoca la tool y recibe el resultado.

Cualquier cliente compatible con MCP puede consumir `server.py`: Claude
Desktop, Cursor, otros agentes de SQL-Agents, etc. Nuestro `client.py` es
solo el cliente que usa el AR del proyecto.

---

## Como funciona mcp.run()

La linea `mcp.run(transport="stdio")` al final de `server.py` activa el
servidor MCP. Funciona asi:

1. Cuando se ejecuta `python agents/MCP/server.py`, Python carga el modulo
   y al llegar al bloque `if __name__ == "__main__":` invoca `main()`.

2. `main()` llama `mcp.run(transport="stdio")`, que pone al proceso en modo
   escucha sobre stdin (entrada estandar). El proceso queda **bloqueado**
   esperando que llegue un mensaje JSON-RPC por stdin.

3. Un cliente MCP (por ejemplo `client.py` o Claude Desktop) lanza este
   archivo como subproceso, abriendo pipes a stdin y stdout del proceso.

4. El cliente envia mensajes JSON por stdin del servidor (ej. la peticion
   `tools/call` con `name=get_database_description` y `arguments={"db_name":
   "demo_db"}`). El servidor responde por stdout con el resultado.

5. La conversacion sigue mientras el cliente quiera. Cuando el cliente
   cierra stdin, el servidor termina.

El servidor no se "activa" en ningun otro sentido: es un proceso normal de
Python que se queda escuchando en sus pipes. Por eso decimos que MCP usa
**transporte stdio**: stdin y stdout son el canal de comunicacion entre
servidor y cliente, en lugar de un puerto HTTP.

---

## Que ve el LLM cuando consume una tool MCP

Cuando un cliente MCP se conecta al servidor, lo primero que hace es pedir
la lista de tools disponibles. El servidor responde con el nombre, los
parametros y la **docstring** de cada funcion decorada con `@mcp.tool()`.
El LLM del cliente recibe esa informacion como descripcion de cada
herramienta y decide cuando invocarla a partir de su docstring.

Por eso las docstrings de `get_database_description` y
`list_available_databases` en `server.py` son detalladas: son el contrato
que el LLM lee para decidir cuando llamar a cada tool y con que argumentos.

---

## Probar la conexion MCP

```bash
venv/Scripts/python.exe agents/MCP/test_mcp.py
```

El test ejecuta seis comprobaciones:

1. `list_available_databases()` en modo directo (mismo proceso).
2. `get_database_description("demo_db")` en modo directo.
3. Comportamiento ante una base de datos inexistente.
4. Conexion completa al protocolo MCP real (subproceso stdio + JSON-RPC).
5. Verifica que el contenido devuelto por el modo directo y el protocolo
   MCP real es el mismo.
6. Lectura de la descripcion de `bird:financial`.

Si los seis tests pasan, MCP esta correctamente integrado.

---

## Formato de los archivos de descripcion

Cada Markdown bajo `db_descriptions/` describe de que trata la base de
datos. Tiene dos secciones:

1. `ARCHIVO DE DESCRIPCION DE LA BASE DE DATOS` — descripcion narrativa del
   dominio: proposito, usuarios tipicos, preguntas frecuentes y
   caracteristicas relevantes.

2. `SCRIPT DE PYTHON QUE DESCRIBE LA BASE DE DATOS` — espacio reservado para
   un script independiente que se conecte a la base y reporte estadisticas
   programaticas. Pendiente hasta que se integre con el modulo CDC.

Convencion de nombres de archivo:

- `demo_db.md` para bases de datos propias del proyecto.
- `bird_<db_id>.md` para bases del benchmark BIRD; corresponde al
  identificador `bird:<db_id>` de `ACTIVE_DATASET`.
- `spider_<db_id>.md` para bases del benchmark Spider.

---

## Tools expuestas

| Tool | Argumentos | Devuelve |
|------|------------|----------|
| `get_database_description` | `db_name: str` | Markdown con la descripcion de la base de datos solicitada |
| `list_available_databases` | (ninguno) | Lista de identificadores de bases de datos con descripcion disponible |

---

## Uso desde el AR

```python
from agents.MCP.client import get_db_description
from config import ACTIVE_DATASET

db_context = get_db_description(ACTIVE_DATASET)
system_prompt += f"\n\n=== CONTEXTO DE LA BASE DE DATOS ===\n{db_context}\n"
```

---

## Arrancar el servidor MCP de forma autonoma

Para que clientes externos como Claude Desktop lo consuman, agregar a
`%AppData%/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sql-agents-db-descriptions": {
      "command": "C:/Users/DAVID/Desktop/agent_skills/venv/Scripts/python.exe",
      "args": ["C:/Users/DAVID/Desktop/agent_skills/agents/MCP/server.py"]
    }
  }
}
```
