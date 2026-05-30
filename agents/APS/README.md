# Agente de Proximidad (APS)

## Rol

Recibe el `ar_result` del AR y selecciona qué tablas y columnas son
relevantes para la query. Es el "schema matcher" del pipeline.

## Modelo y configuración

APS no es un agente LLM puro — usa principalmente embeddings vectoriales.
Sus hiperparámetros optimizados (Phase 2 Optuna v3):

- similarity_metric: `ip` (inner product)
- embedding_model: `intfloat/multilingual-e5-base`
- top_n_tables: 5
- top_n_columns: 5
- LLM (zona gris): gpt-4o-mini

Se carga desde `experiments/winners/best_hyperparameters.json`.

## Recursos externos que usa

| Recurso | Para qué |
|---|---|
| **ChromaDB** (vector_db/) | Búsqueda semántica de tablas/columnas relevantes |
| **MCP Server** | 2 tools: `find_join_path` (camino FKs) y `find_column_for_concept` (mapeo concepto → columna) |
| **OpenAI** | SOLO en zona gris (similitud entre 0.40 y 0.80) |

**No accede a MySQL/PostgreSQL directamente.**

## Sub-carpetas mysql/ y postgres/

APS es **multi-backend**. Cada backend tiene su propia implementación:

- `agents/APS/mysql/schema_matcher_agent.py` → `MysqlSchemaMatcherAgent`
- `agents/APS/postgres/schema_matcher_agent.py` → `PostgresSchemaMatcherAgent`

Diferencias: prompts específicos, formato de identificadores (backticks vs
comillas dobles), conexión a vector_db distinto.

## ChromaDB — cómo se construye

Al primer arranque del APS, indexa al vector_db leyendo:

- `metadata/<dataset>/<bd>/light_schema.json` (estructura DDL)
- `metadata/<dataset>/<bd>/diccionario_datos.json` (descripciones humanas)

Genera embeddings con sentence-transformers y los persiste en
`vector_db/<backend>/<dataset>/<bd>/`. Después de la primera vez, no se
reindexa salvo que borres la carpeta.

## Flujo adaptativo

El APS NO sigue un orden fijo de pasos. Decide condicionalmente:

1. **Búsqueda inicial** en ChromaDB → obtiene top-K tablas y columnas.
2. **Si hay 2+ tablas relevantes** → llama `find_join_path` (MCP) para
   obtener camino de JOINs y posibles tablas puente.
3. **Si la similitud cae en zona gris** (0.40–0.80) → consulta LLM
   para resolver compatibilidad semántica.
4. **Si el concepto del usuario es ambiguo** → llama
   `find_column_for_concept` (MCP) para mapeo semántico.

Costo: solo paga el costo de tools adicionales cuando son necesarias.

## Estructura de carpetas

```
agents/APS/
├── __init__.py                ← factory que devuelve mysql/postgres por db_type
├── mysql/
│   ├── prompt.py
│   ├── schema_matcher_agent.py  ← MysqlSchemaMatcherAgent
│   ├── skills.py
│   └── phase_2/
│       ├── main.py            ← evaluador específico mysql
│       ├── dataset.json
│       ├── tools/optimize_optuna.py
│       └── results/optuna/
├── postgres/
│   ├── (mismo layout que mysql)
│   └── phase_2/
└── comparison/
    └── comparison.py          ← cross-backend (lee de ambos phase_2/results)
```

## Resultados experimentales (Phase 2)

| Backend | Best Faithfulness (F1) | Best Combined |
|---|---|---|
| MySQL | 0.533 | 0.255 |
| Postgres | 0.523 | 0.329 |

**Nota metodológica**: la métrica original era Precision-only y daba combined ~0.04
(falsamente bajo). Cambiamos a F1 (estándar para retrieval evaluation) y
detectamos un bug donde top_n_tables/top_n_columns no se propagaban al agente.
Documentado en `agents_experiments.md`.

**No tiene Phase 3** — DeepEval LLM-as-judge no aplica para retrieval
estructurado; se usan métricas IR (Precision/Recall/F1).
