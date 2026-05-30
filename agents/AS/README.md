# Agente Sustentador (AS)

## Rol

Agente "bajo demanda". Se activa cuando el usuario formula preguntas sobre
el proceso tras una consulta exitosa: "¿por qué elegiste esas tablas?",
"¿cómo decidiste el JOIN?", "¿qué tan seguro estás?".

A diferencia del AE (que explica QUÉ hace la query en lenguaje simple), el
AS justifica POR QUÉ se generó así, dirigido a un evaluador o académico.

## Modelo y configuración

Cargado desde `experiments/winners/best_hyperparameters.json`:

- model: gpt-4o-mini
- temperature: 0.3
- combined_score Phase 2 (con LTM skill): 1.654

## Recursos externos que usa

| Recurso | Para qué |
|---|---|
| OpenAI API (gpt-4o-mini) | Razonamiento principal |
| Skills agentic propias | `get_agent_reasoning`, `get_long_term_history` |
| **ChromaDB** (a través de `get_long_term_history`) | Solo cuando el usuario pregunta por sesiones pasadas |
| Memoria a corto plazo | Contexto del turno actual via `last_result_cache` |

**No accede a MCP estructural. No accede a MySQL/PostgreSQL.**

## Skills agentic

### `get_agent_reasoning(agent)`

Para preguntas sobre la **consulta actual**. Devuelve el reasoning
detallado de un agente específico (`AR`, `APS`, `AG`, `AV`) del turno actual.

Ejemplos disparadores:
- "¿qué tablas usaste?" → `get_agent_reasoning("APS")`
- "¿por qué ese JOIN?" → `get_agent_reasoning("AG")`
- "¿cómo se validó?" → `get_agent_reasoning("AV")`

### `get_long_term_history(query, limit, mode)`

Para preguntas sobre el **historial de sesiones pasadas**.

Modos:
- `"semantic"`: similitud coseno contra ChromaDB (LongTermMemory)
- `"recent"`: últimas N queries por timestamp

Ejemplos disparadores:
- "¿qué consultas similares respondiste antes?" → `mode="semantic"`
- "muéstrame las últimas 3" → `mode="recent"`
- "¿hace cuánto te pregunté algo de cantantes?" → `mode="semantic"`

## Diferencia con AE

| | AE (Explicador) | AS (Sustentador) |
|---|---|---|
| Audiencia | Usuario sin conocimiento técnico | Evaluador / académico |
| Cuándo se invoca | Siempre al final del pipeline | Solo cuando el usuario pregunta sobre el proceso |
| Lenguaje permitido | Sin jerga SQL ni términos técnicos | Puede mencionar tablas, columnas, JOINs |
| Skill agentic | `format_sql_readable` | `get_agent_reasoning` + `get_long_term_history` |
| Memoria | Corto plazo (chain_rejection) | Corto plazo + largo plazo (Chroma) |

## Flujo de una llamada

1. Recibe `last_result` (state del pipeline anterior) + `user_question`.
2. Cachea `last_result` en `_last_result_cache` para que las skills
   puedan acceder.
3. Construye prompt con resumen del pipeline.
4. Invoca agentic loop — el LLM decide qué skill usar según la pregunta:
   - Pregunta sobre turno actual → `get_agent_reasoning`
   - Pregunta sobre historial → `get_long_term_history`
5. Retorna texto plano con la justificación.

## Estructura de carpetas

```
agents/AS/
├── sustainer_agent.py
├── prompt.py
├── skills.py                   ← get_agent_reasoning + get_long_term_history
├── phase_2/
│   ├── main.py                 ← incluye MockLongTermMemory para evaluación
│   ├── dataset.json            ← 13 casos (Q1-11 actual + Q12-13 LTM)
│   ├── tools/optimize_optuna.py
│   ├── docs/
│   └── results/optuna/
└── phase_3/
    ├── paraphrases.json        ← 39 paráfrasis (13 × 3)
    ├── deepeval_runner.py
    └── results/
```

## Resultados experimentales

| Phase | Best combined | Métrica clave |
|---|---|---|
| Phase 2 v1 (1 skill) | 1.591 | Faith=0.955, Read=1.0 |
| Phase 2 v2 (con LTM skill) | **1.654** | Faith=0.962, Read=1.0 |
| Phase 3 (DeepEval) | — | FaithfulnessMetric=0.923, GEval=0.732, 8/13 robustos |

## Hallazgos publicables

- La adición de la skill `get_long_term_history` mejoró combined +0.063.
- En 8/8 trials, el LLM elige correctamente entre `get_agent_reasoning` y
  `get_long_term_history` según el tipo de pregunta — valida que la
  separación en el prompt funciona.
