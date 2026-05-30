# Dockerfile — imagen base para todos los agentes del sistema SQL-Agents
#
# Un solo Dockerfile sirve para los 7 servicios (MCP, AR, APS, AG, AV, AE, AS,
# orquestador). El comportamiento especifico se controla con dos argumentos:
#
#   ARG REQS   → selecciona el conjunto de dependencias a instalar:
#                  requirements.txt       (PESADO ~2.3 GB) — MCP y APS
#                  requirements-light.txt (LIVIANO ~570 MB) — el resto
#
#   ENV AGENT_ROLE → indica que servidor arrancar al iniciar el contenedor:
#                  AR | APS | AG | AV | AE | AS | orchestrator
#
# La separacion PESADO/LIVIANO existe porque MCP y APS necesitan
# sentence-transformers (torch CPU ~1 GB) para calcular embeddings vectoriales.
# Los demas agentes no usan torch y se benefician de una imagen mucho mas ligera.

FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema operativo:
#   gcc/g++  → necesarios para compilar algunas extensiones de Python (ej. tokenizers)
#   curl     → usado por los healthchecks de docker-compose para verificar /health
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# ARG configurable en tiempo de build (docker-compose lo pasa via args.REQS).
# Por defecto usa requirements.txt (pesado) si no se especifica.
ARG REQS=requirements.txt
COPY requirements.txt requirements-light.txt ./

# --timeout 300 evita que pip corte la descarga de paquetes grandes como
# torch (~190 MB CPU), sentence-transformers o deepeval con conexion lenta.
RUN pip install --no-cache-dir --timeout 300 -r ${REQS}

# Copiar el codigo fuente completo al contenedor
COPY . .

# AGENT_ROLE selecciona que servidor Python arrancar.
# docker-compose sobreescribe esta variable por servicio (ej. AGENT_ROLE=AV).
ENV AGENT_ROLE=AR

# Arrancar el servidor del agente indicado por AGENT_ROLE.
# Cada agente tiene su propio agents/<ROL>/server.py con un servidor FastAPI
# que expone el endpoint POST /invoke para recibir llamadas del orquestador.
CMD python -m agents.${AGENT_ROLE}.server
