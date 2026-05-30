# Agente Refinador (AR)

## Rol

Recibe la pregunta cruda del usuario (con typos, abreviaciones, ambigüedad) y
devuelve una versión refinada + intent estructurado + flag de validez.

Es el primer agente del pipeline. Si rechaza la query, los agentes downstream
no se invocan.

## Modelo y configuración

Cargado automáticamente desde `experiments/winners/best_hyperparameters.json`:

- model: gpt-4o
- temperature: 0.1
- combined_score Phase 2: 1.834
- Faithfulness Phase 3: 1.000

## Recursos externos que usa

| Recurso | Para qué |
|---|---|
| OpenAI API (gpt-4o) | Razonamiento principal |
| MCP Server | Una sola tool: `get_database_description` (lee `db_descriptions/*.md`) |

**No usa ChromaDB. No accede a MySQL/PostgreSQL.**

## Flujo de una llamada

1. Recibe `user_input` del orquestador.
2. Llama `get_database_description_client(ACTIVE_DATASET)` → lee el .md de descripción.
3. Construye system prompt = prompt original + descripción del dominio.
4. Llama OpenAI con prompt enriquecido.
5. Retorna dict con `refined_query`, `is_valid_query`, `confidence_score`,
   `intent`, `reasoning`.

## Skills agentic

Hoy: **ninguna** (lista vacía en `_get_skills()`). El AR es LLM puro con MCP
externo.

## Estructura de carpetas

```
agents/AR/
├── refiner_agent.py           ← lógica principal
├── prompt.py                  ← system prompt
├── skills.py                  ← (vacío hoy)
├── phase_1/                   ← grid search baseline
├── phase_2/                   ← Optuna 8 trials
│   ├── main.py                ← evaluador
│   ├── dataset.json           ← 12 preguntas
│   ├── tools/                 ← 5 herramientas (Optuna, DSPy, Langfuse, ...)
│   └── results/               ← results.csv + plots
└── phase_3/                   ← DeepEval LLM-as-judge
    ├── paraphrases.json       ← 36 paráfrasis
    ├── deepeval_runner.py     ← runner
    └── results/               ← scores_raw.csv + summary.csv
```

## Resultados experimentales

| Phase | Best combined | Métrica clave |
|---|---|---|
| Phase 2 (Optuna) | 1.834 | Faithfulness=1.0, ROUGE-L=1.0 |
| Phase 3 (DeepEval) | — | FaithfulnessMetric=1.0, GEval=0.704, 8/11 robustos |
