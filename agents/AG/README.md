# Agente Generador (AG)

## Rol

Recibe el `aps_result` (tablas + columnas seleccionadas) + `ar_result` y
genera el SQL final. Es el corazón NL→SQL del pipeline.

Soporta corrección iterativa: si el AV detecta errores, el AG re-genera el
SQL incorporando el feedback.

## Modelo y configuración

Cargado desde `experiments/winners/best_hyperparameters.json`:

- model: gpt-4o
- temperature: 0.1
- combined_score Phase 2 (mysql): 1.428
- combined_score Phase 2 (postgres): 1.21

## Recursos externos que usa

| Recurso | Para qué |
|---|---|
| OpenAI API (gpt-4o) | Generación de SQL |
| Skills agentic propias | `validate_sql_safety`, `fix_reserved_words` |

**No accede a MCP. No accede a ChromaDB. No accede directamente a MySQL/PostgreSQL.**

## Sub-carpetas mysql/ y postgres/

AG es **multi-backend**. Cada backend tiene su propio prompt y skills
específicas:

- `agents/AG/mysql/sql_generator_agent.py`
- `agents/AG/postgres/sql_generator_agent.py`

Diferencias: sintaxis de identificadores, funciones específicas del motor,
manejo de palabras reservadas.

## Skills agentic

| Skill | Para qué |
|---|---|
| `validate_sql_safety` | Verifica que el SQL solo contenga SELECT (no DELETE, DROP, etc.) |
| `fix_reserved_words` | Agrega backticks/comillas a alias que coinciden con palabras reservadas |

El LLM las invoca durante la generación cuando lo necesita.

## Flujo de una llamada

1. Recibe `aps_result` con tablas/columnas/joins relevantes.
2. Construye prompt con contexto del schema.
3. Genera SQL via agentic loop (puede invocar skills).
4. Si hay feedback del AV de iteración previa, lo incorpora.
5. Retorna `ag_result` con `sql`, `strategy`, `confidence_score`, `reasoning`.

## Estructura de carpetas

```
agents/AG/
├── mysql/
│   ├── prompt.py
│   ├── sql_generator_agent.py
│   ├── agentic_skills.py
│   ├── functions.py
│   └── phase_2/
│       ├── main.py
│       ├── dataset.json
│       ├── tools/optimize_optuna.py
│       └── results/optuna/
├── postgres/
│   └── (mismo layout)
└── comparison/comparison.py    ← cross-backend
```

## Resultados experimentales (Phase 2)

| Backend | Faithfulness | ROUGE-L | Combined |
|---|---|---|---|
| MySQL | 0.902 | 0.89 | 1.428 |
| Postgres | 0.884 | 0.872 | 1.21 |

**No tiene Phase 3** — DeepEval LLM-as-judge no aplica para SQL
sintáctico/estructural. La métrica académica para NL2SQL es **Execution
Accuracy** (ejecutar contra MySQL/Postgres real y comparar resultados),
medida en el experimento E2E.

## Hallazgos publicables

- En Q11 del E2E, el AG generó SQL **mejor que el gold** (gold de Spider
  viola `ONLY_FULL_GROUP_BY` de MySQL strict mode; AG agrupa correctamente).
- gpt-4o gana sobre gpt-4o-mini para AG (tarea estructurada requiere mayor
  precisión sintáctica).
