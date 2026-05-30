# SQL-Agents: Sistema Multi-Agente para Generación de SQL desde Lenguaje Natural

**Tesis de Maestría en Inteligencia Artificial**  
Félix David Córdova García — Pontificia Universidad Javeriana, Bogotá D.C.  
Asesor: Edison Leonardo Neira Espitia

---

## Descripción

SQL-Agents convierte preguntas en lenguaje natural en consultas SQL ejecutables sobre bases de datos relacionales MySQL y PostgreSQL, sin que el usuario necesite conocer SQL. El sistema implementa el patrón **Agent Skills** de Anthropic y el protocolo **MCP (Model Context Protocol)** dentro de un pipeline multi-agente orquestado con LangGraph.

A diferencia de sistemas existentes como DIN-SQL, DAIL-SQL o MAC-SQL, SQL-Agents no recibe el DDL completo como entrada, no requiere datos de entrenamiento por dominio, e integra validación proactiva, gestión de contexto conversacional y memoria entre sesiones — características que ningún sistema publicado combina simultáneamente.

**Resultado principal (Fase 4 — robustez ante 30 paráfrasis):**

| Base de datos | Score normalizado | Combined |
|---------------|-----------------|----------|
| MySQL | **82.9 %** | 2.488 / 3.0 |
| PostgreSQL | **83.2 %** | 2.495 / 3.0 |

---

## Arquitectura del sistema

El pipeline sigue el flujo:

```
Usuario
  │
  ▼
Orquestador ── STM (RAM) / LTM (ChromaDB)
  │
  ├─[conversacional]──► respuesta directa
  ├─[sustentación]────► AS → respuesta
  ├─[cache hit]───────► resultado sin pipeline
  │
  └─[sql_query]
       │
       AR ──[rechaza]──────────────────────────► AE
       │
       APS ──[sin tablas]──────────────────────► AE
       │
       AG ◄──────────────────────────────────────┐
       │                                   retry < 3
       AV ──[needs_correction]────────────────────┘
       │
       AE → WACS → respuesta final al usuario
```

### Los seis agentes

| Agente | Nombre | Rol | Skills internas |
|--------|--------|-----|-----------------|
| **AR** | Refinador | Valida que la pregunta sea una consulta de BD legítima y la reformula | — |
| **APS** | Schema Matcher | Recupera las tablas y columnas relevantes por similitud vectorial en ChromaDB | `search_tables`, `search_columns` |
| **AG** | Generador SQL | Genera SQL (MySQL o PostgreSQL) a partir del schema reducido de APS | `validate_sql_safety`, `fix_reserved_words`, `find_joins_among_tables`, `find_join_path` |
| **AV** | Validador | Verifica sintaxis, compliance con el schema y semántica del SQL; retroalimenta a AG si hay errores | `check_syntax_rules`, `check_semantic_patterns`, `check_query_efficiency`, `check_query_performance` |
| **AE** | Explicador | Sintetiza el resultado en lenguaje natural y calcula el índice WACS | `format_sql_readable` |
| **AS** | Sustentador | Responde bajo demanda preguntas sobre el proceso (¿por qué esas tablas?, ¿cuántas iteraciones?) | `get_agent_reasoning` |

### WACS — Weighted Agent Confidence Score

Índice de confianza global del pipeline, ponderado por la posición causal de cada agente:

```
WACS = 0.10·c_AR + 0.20·c_APS + 0.30·c_AG + 0.40·c_AV
```

### Componentes auxiliares

- **Orquestador**: clasifica el intent (conversacional / sustentación / sql_query), detecta cambios de contexto, resuelve referencias anafóricas ("de ellos", "los mismos") y consulta la memoria antes de activar el pipeline.
- **MCP Server**: externaliza el contexto de dominio (descripción narrativa de la BD) y las operaciones sobre el schema (grafo de FKs, similitud coseno), desacopladas del proveedor LLM.
- **Tool Registry** (`registry.py`): registro central que controla qué skills puede invocar cada agente. Si un LLM recibe un prompt injection e intenta llamar una skill de otro agente, el registry lo bloquea antes de ejecutar cualquier código y lo registra en el log de auditoría.
- **Memoria STM**: caché RAM de la sesión activa. Evita reprocesar preguntas idénticas o muy similares.
- **Memoria LTM**: ChromaDB persistente por usuario / base de datos / dataset. Si el orquestador encuentra un acierto, retorna el resultado sin invocar ningún agente.
- **jobs/**: módulo CDC (Change Data Capture) que conecta a MySQL o PostgreSQL, extrae el DDL completo, índices, FKs y estadísticas, y lo serializa en JSON para APS y MCP.

---

## Modelos LLM evaluados

Seis modelos de tres proveedores, cubriendo el espectro eficiencia–capacidad:

| Modelo | Proveedor | Fase 0 | Fases 1–4 |
|--------|-----------|:------:|:---------:|
| GPT-4o-mini | OpenAI | Sí | — |
| GPT-4o | OpenAI | Sí | Sí |
| Claude Haiku 4.5 | Anthropic | Sí | Sí |
| Claude Sonnet 4.6 | Anthropic | Sí | Sí |
| Gemini 2.5 Flash | Google | Sí | Sí |
| Gemini 2.5 Pro | Google | Sí | — |

**Juez de evaluación:** GPT-4o con T = 0.0 (fijo en todas las fases, garantiza determinismo).  
**Embeddings:** `intfloat/multilingual-e5-base` (109 M parámetros, 100 idiomas, local sin API de pago).

---

## Protocolo experimental (cuatro fases)

### Fase 0 — Grilla exhaustiva y selección de herramienta

Se ejecutó una grilla de **6 modelos × 5 temperaturas = 30 combinaciones** sobre el Agente Refinador (AR) con dos objetivos simultáneos:

1. Comparar cinco herramientas de optimización y observabilidad (Optuna, LangSmith, Langfuse, DSPy, Promptfoo) bajo condiciones idénticas.
2. Determinar el espacio reducido de hiperparámetros para Fases 1–4.

**Resultado:** se seleccionó **Optuna TPE** por ser la única herramienta con búsqueda bayesiana automática y visualizaciones reproducibles sin dependencias cloud. GPT-4o-mini y Gemini 2.5 Pro se excluyeron de las fases siguientes por menor rendimiento frente a sus pares del mismo proveedor. T = 0.1 se descartó (Δ < 0.005 vs T = 0.0 en todos los agentes). El espacio final fue: {gpt-4o, claude-haiku-4-5, claude-sonnet-4-6, gemini-2.5-flash} × {0.0, 0.3, 0.5, 0.7} = 16 combinaciones.

### Fase 1 — Optimización independiente por agente con Optuna TPE

Cada agente se optimizó de forma **independiente** con Optuna TPE (`seed=66`, 10 trials) sobre 10 preguntas de prueba del benchmark Spider 1.0 `concert_singer`. Las métricas se adaptan al rol de cada agente:

| Agente | Función objetivo | Máximo |
|--------|-----------------|--------|
| AR, AV | `(2F + R + 2G − E − B) / 5` | 1.0 |
| AE, AS | `(2F + 2G + 0.5·R − E − B) / 4.5` | 1.0 |
| AG | `F + G + R_SQL − B − E` | 3.0 |
| APS | `(F1_tablas + F1_columnas) / 2` | 1.0 |

Donde: **F** = Faithfulness (GEval LLM-as-judge), **G** = Groundedness (GEval), **R** = ROUGE-L, **B** = Brier Score ↓, **E** = ECE ↓, **R_SQL** = ROUGE-L sobre SQL normalizado con `normalize_sql_v2`.

**Ganadores Fase 1:**

| Agente | Modelo | T | Score |
|--------|--------|---|-------|
| AR | gemini-2.5-flash | 0.5 | 0.962 |
| APS | cosine / multilingual-e5-base | — | 0.842 |
| AG MySQL | claude-haiku-4-5 | 0.3 | 2.700 |
| AG PostgreSQL | claude-haiku-4-5 | 0.3 | 2.687 |
| AV MySQL | claude-sonnet-4-6 | 0.7 | 0.956 |
| AV PostgreSQL | claude-haiku-4-5 | 0.0 | 0.955 |
| AE | gpt-4o | 0.3 | 0.792 |
| AS | claude-sonnet-4-6 | 0.7 | 0.870 |

> **Sub-fases 1\_1 / 1\_2 (AG y AV):** la Fase 1\_2 corrige tres problemas metodológicos: data leakage en prompts, penalización injusta de JOINs semánticamente equivalentes en ROUGE-L (`normalize_sql_v2`), y umbral de outcome reducido de 0.80 a 0.70. El modelo ganador fue el mismo en ambas sub-fases.

### Fase 2 — Robustez a paráfrasis por agente

Los ganadores de Fase 1 se fijaron y se evaluaron sobre 3 reformulaciones lingüísticas por intent (informal, reformulada, vocabulario alternativo), generando **30 entradas por agente**.

- **AV es el más robusto:** ΔCombined = 0 en MySQL.
- **AG es el más sensible:** Postgres pierde Δ = −0.400 en combined.
- **AE y AS son estables:** caídas ≤ 0.033.

### Fase 3 — Búsqueda de hiperparámetros del orquestador

El pipeline completo se evaluó sobre **10 escenarios** que cubren todas las rutas posibles. `ToolCorrectnessMetric` (DeepEval) verifica que el orquestador activa los agentes y skills correctos.

**Función objetivo:** `combined_orch = F + G + C − B − E` (máx ≈ 3.0)  
**Ganador (ambas bases de datos):** `claude-haiku-4-5`, T = 0.3

| Métrica | MySQL | PostgreSQL |
|---------|------:|----------:|
| Faithfulness | 0.857 | 0.810 |
| Groundedness | 0.948 | 0.875 |
| ToolCorrectness | 0.991 | 0.800 |
| Brier ↓ | 0.005 | 0.005 |
| ECE ↓ | 0.071 | 0.067 |
| **Combined** | **2.744** | **2.554** |

### Fase 4 — Robustez del orquestador (30 paráfrasis)

Los 10 escenarios de Fase 3 se reformularon en 3 variantes cada uno.

| | MySQL | PostgreSQL |
|--|------:|----------:|
| Combined Fase 3 | 2.744 | 2.554 |
| Combined Fase 4 | 2.488 | 2.495 |
| Caída relativa | −9.3 % | −2.3 % |
| **Score normalizado** | **82.9 %** | **83.2 %** |

---

## Conclusiones principales

1. **No existe un único LLM óptimo para todo el pipeline.** Cada rol converge a un modelo diferente: AG a `claude-haiku-4-5`, AE a `gpt-4o`, AS a `claude-sonnet-4-6`, orquestador a `claude-haiku-4-5`.
2. **La calidad de la métrica es tan crítica como la del agente.** Corregir data leakage, normalización de JOINs en ROUGE-L y el umbral de outcome redujo el Brier de PostgreSQL en más del 96 %.
3. **Alta robustez lingüística.** Caídas menores al 10 % al pasar de 10 preguntas a 30 paráfrasis en todas las bases de datos.
4. **Agent Skills + MCP es viable arquitectónicamente.** Permite sustituir el LLM de cualquier agente sin modificar los demás.

---

## Estructura del repositorio

```
agent_skills/
│
├── agents/
│   ├── AR/                        # Refinador
│   │   ├── refiner_agent.py
│   │   ├── prompt.py
│   │   ├── server.py              # servidor FastAPI del agente (POST /invoke)
│   │   ├── phase_0/               # Grilla exhaustiva 30 combinaciones
│   │   ├── phase_1/               # Optuna TPE — búsqueda de hiperparámetros
│   │   └── phase_2/               # Robustez a paráfrasis
│   ├── APS/                       # Schema Matcher (mysql/ y postgres/)
│   ├── AG/                        # Generador SQL
│   │   ├── mysql/
│   │   │   ├── phase_1/           # Fase 1_1 (línea base)
│   │   │   ├── phase_1_2/         # Fase 1_2 (correcciones metodológicas)
│   │   │   ├── phase_2/
│   │   │   └── phase_2_2/
│   │   └── postgres/              # Misma estructura que mysql/
│   ├── AV/                        # Validador (misma estructura que AG)
│   ├── AE/                        # Explicador
│   ├── AS/                        # Sustentador
│   ├── MCP/                       # Servidor MCP: tools, client, server
│   └── base_skill_agent.py        # Clase base con loop ReAct (OpenAI y Anthropic)
│
├── orchestrator/
│   ├── graph.py                   # Definición del DAG LangGraph
│   ├── nodes.py                   # Implementación de cada nodo del grafo
│   ├── state.py                   # GraphState compartido entre nodos
│   ├── orchestrator_agent.py      # Routing, STM/LTM, historial conversacional
│   ├── memory_functions.py        # check_short_term_memory / check_long_term_memory
│   ├── prompt.py                  # Prompt del clasificador de intent
│   ├── phase_3/                   # Evaluación E2E Fase 3 (mysql/ y postgres/)
│   ├── phase_3_2/                 # Fase 3_2: correcciones AS routing + bypass opiniones
│   ├── phase_4/                   # Robustez 30 paráfrasis Fase 4
│   └── phase_4_2/                 # Fase 4_2: correcciones completas
│
├── memory/
│   ├── long_term_memory.py        # ChromaDB persistente por usuario/base de datos/dataset
│   └── short_term.py              # Caché RAM de sesión
│
├── metricas_lib/
│   ├── sql_metrics.py             # normalize_sql, rouge_l_sql (v1 y v2)
│   ├── deepeval_metrics.py        # Faithfulness, Groundedness (GEval LLM-as-judge)
│   ├── calibration.py             # Brier Score, ECE
│   ├── classification_metrics.py  # Accuracy, F1, Precision, Recall (para AV)
│   └── retrieval_metrics.py       # Precision@k, MRR (para APS)
│
├── jobs/
│   ├── schema_extractor.py        # Dispatcher: elige mysql o postgres según DB_TYPE
│   ├── mysql/schema_extractor.py  # CDC MySQL: extrae DDL, índices, FKs → schema.json
│   └── postgres/schema_extractor.py # CDC PostgreSQL: misma función, dialectos propios
│
├── scripts/
│   ├── create_database.py         # Crea la BD demo_db en MySQL
│   ├── create_schema.py           # Crea tablas: students, courses, enrollments
│   ├── insert_data.py             # Inserta datos de prueba (10 filas/tabla)
│   ├── generate_random_university_data.py  # Genera datos masivos con Faker
│   ├── setup_bird_postgres.py     # Configura dataset BIRD en PostgreSQL
│   ├── eval_spider.py             # Evaluación end-to-end con Spider 1.0
│   └── eval_bird.py               # Evaluación end-to-end con BIRD
│
├── static/
│   └── index.html                 # UI web servida por api.py
│
├── experiments/
│   └── winners/
│       └── best_hyperparameters.json   # Ganadores Optuna — cargado por config.py al inicio
│
├── registry.py                    # Tool Registry: control de acceso por agente (13 skills, 7 agentes)
├── Dockerfile                     # Imagen base para todos los agentes (pesada/liviana según ARG REQS)
├── docker-compose.yml             # Orquestación de 8 servicios: MCP + 6 agentes + orquestador
├── config.py                      # Configuración global: modelos, paths, umbrales
├── api.py                         # Backend FastAPI (puerto 8000)
├── main.py                        # CLI conversacional
├── requirements.txt               # Dependencias completas (con torch, para desarrollo y experimentos)
└── requirements-light.txt         # Dependencias mínimas sin torch (para contenedores de agentes)
```

---

## Instalación

### Requisitos previos

- Python 3.11
- MySQL 8.0 y/o PostgreSQL 14+
- Claves API de los proveedores a usar (OpenAI, Anthropic, Google)

### Pasos

```bash
# 1. Clonar el repositorio
git clone <url>
cd agent_skills

# 2. Crear entorno virtual
python -m venv venv311
.\venv311\Scripts\activate       # Windows
# source venv311/bin/activate    # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Crear el archivo .env en la raíz del proyecto
#    OPENAI_API_KEY=sk-...
#    ANTHROPIC_API_KEY=sk-ant-...
#    GOOGLE_API_KEY=AIza...

# 5. (Opcional) Crear la BD demo y cargar datos de prueba
python scripts/create_database.py
python scripts/create_schema.py
python scripts/insert_data.py

# 6. Extraer el schema de la BD y construir el índice vectorial
python jobs/mysql/schema_extractor.py       # MySQL
python jobs/postgres/schema_extractor.py    # PostgreSQL
```

---

## Uso

### CLI conversacional

```bash
python main.py
```

| Comando | Acción |
|---------|--------|
| `nuevo tema` | Reinicia el contexto conversacional |
| `memoria` | Estado de la memoria de sesión y persistente |
| `limpiar` | Vacía la caché de sesión |
| `debug` | Activa/desactiva análisis detallado por agente |
| `reindex` | Re-indexa ChromaDB con el schema actual |
| `salir` | Cierra sesión y persiste STM → LTM |

### API REST + UI web

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Abrir `http://localhost:8000` en el navegador.

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/session` | Crear sesión (usuario + base de datos + dataset) |
| POST | `/api/query` | Enviar pregunta en lenguaje natural |
| DELETE | `/api/session/{id}` | Cerrar sesión y flush STM → LTM |
| GET | `/api/stats/{id}` | Estadísticas de la sesión |
| POST | `/api/reindex` | Re-indexar ChromaDB |
| DELETE | `/api/memory/{id}` | Borrar LTM del usuario |

### Docker — despliegue en microservicios

```bash
docker compose up --build    # primer arranque: construye las dos imágenes y descarga el modelo (~1 GB)
docker compose up            # reinicios posteriores (usa caché)
docker compose down          # detener todos los servicios
```

El sistema levanta **8 contenedores**. El orquestador espera a que todos los agentes estén `healthy` antes de arrancar:

| Contenedor | Puerto | Imagen |
|------------|--------|--------|
| `sql_agents_mcp` | 8010 | Pesada (~2.3 GB) |
| `sql_agents_aps` | 8002 | Pesada |
| `sql_agents_ar` | 8001 | Liviana (~570 MB) |
| `sql_agents_ag` | 8003 | Liviana |
| `sql_agents_av` | 8004 | Liviana |
| `sql_agents_ae` | 8005 | Liviana |
| `sql_agents_as` | 8006 | Liviana |
| `sql_agents_orchestrator` | 8000 | Liviana |

La diferencia de tamaño entre imágenes existe porque MCP y APS necesitan `sentence-transformers` con PyTorch para calcular embeddings vectoriales. Los demás agentes invocan modelos remotos vía API y no requieren torch.

---

## Datasets soportados

| Dataset | Identificador | Descripción |
|---------|--------------|-------------|
| Demo DB propia | `demo_db` | BD escuela (students, courses, enrollments) |
| Spider 1.0 | `spider:<db_id>` | 200 BDs, ej: `spider:concert_singer` |
| BIRD | `bird:<db_id>` | BDs reales de mayor complejidad |

Para cambiar el dataset activo, añadir en `.env`:

```
ACTIVE_DATASET=spider:concert_singer
```

---

## Ejecutar los experimentos

Cada agente tiene sus propios scripts por fase. Ejemplo para AG MySQL:

```bash
# Fase 1: búsqueda Optuna (10 trials, seed=66)
python agents/AG/mysql/phase_1_2/run_optuna.py

# Fase 2: robustez a paráfrasis
python agents/AG/mysql/phase_2_2/run_phase2.py

# Fase 3: evaluación del pipeline completo
python orchestrator/phase_3_2/mysql/run_optuna.py

# Fase 4: robustez del orquestador
python orchestrator/phase_4_2/mysql/run_phase4.py
```

Los resultados se guardan en `agents/<AGENTE>/phase_*/results/` y `orchestrator/phase_*/results/`, excluidos del repositorio por `.gitignore`.

---

## Alcance y limitaciones

- El sistema genera, valida y explica la consulta SQL pero **no la ejecuta** contra la base de datos.
- No cubre tipos de datos complejos: ARRAY, JSON anidado, datos geoespaciales.
- Evaluado sobre esquemas OLTP/ACID en estrella o copo de nieve.
- La ejecución del SQL y el manejo de resultados estructurados complejos quedan a cargo de la capa de aplicación que integre el sistema.

---

## Referencia

> Córdova García, F. D. (2026). *SQL-Agents: Sistema Multi-Agente para la Generación de Consultas SQL desde Lenguaje Natural*. Tesis de Maestría en Inteligencia Artificial, Pontificia Universidad Javeriana, Bogotá D.C.

---

