# Agente Explicador (AE)

## Rol

Recibe el `full_state` completo del pipeline (AR + APS + AG + AV +
final_sql) y genera una **narrativa unificada** en lenguaje simple,
orientada a usuarios sin conocimientos técnicos.

Sintetiza: qué se entendió de la pregunta, con qué datos se respondió, si
hubo correcciones del validador, y nivel de confianza global.

## Modelo y configuración

Cargado desde `experiments/winners/best_hyperparameters.json`:

- model: gpt-4o-mini
- temperature: 0.3
- combined_score Phase 2: 1.686 (post-bugfix de substring)

**Nota**: gpt-4o-mini gana sobre gpt-4o para AE — produce explicaciones más
directas que cubren mejor los tokens críticos.

## Recursos externos que usa

| Recurso | Para qué |
|---|---|
| OpenAI API (gpt-4o-mini) | Generación de la narrativa |
| Skill agentic propia | `format_sql_readable` para mostrar SQL formateado |

**No accede a MCP. No accede a ChromaDB. No accede a MySQL/PostgreSQL.**

## Skills agentic

| Skill | Para qué |
|---|---|
| `format_sql_readable` | Formatea el SQL final con saltos de línea ante cada keyword principal para que el usuario pueda leerlo |

## Caso especial: chain_rejection

Si el orquestador marca `chain_rejection=true` en el state (porque el usuario
encadenó una pregunta y la BD no tiene la columna requerida), el AE consulta
su `context_history` y construye un mensaje que reconoce la limitación e
indica el alcance máximo del sistema dado el esquema actual — sin frustrar
la conversación.

## Flujo de una llamada

1. Recibe `full_state` con AR/APS/AG/AV results + final_sql.
2. Construye user_message serializando el estado del pipeline.
3. Invoca `format_sql_readable` (vía agentic loop) para formatear el SQL.
4. Genera narrativa de 2-3 oraciones cubriendo: qué entendió + qué datos
   usó + si hubo correcciones + nivel de confianza.
5. Retorna `ae_result` con `final_reasoning`, `final_sql` formateado,
   `confidence_level`.

## Estructura de carpetas

```
agents/AE/
├── explainer_agent.py
├── prompt.py
├── skills.py                   ← format_sql_readable
├── phase_2/
│   ├── main.py
│   ├── dataset.json            ← 11 casos con expected_critical_tokens
│   ├── tools/optimize_optuna.py
│   ├── docs/
│   └── results/optuna/
└── phase_3/
    ├── paraphrases.json        ← 33 paráfrasis
    ├── deepeval_runner.py
    └── results/
```

## Métricas

| Métrica | Definición |
|---|---|
| Faithfulness | Cobertura de tokens críticos (con grupos de sinónimos) |
| Readability | Sin jerga SQL + longitud 50-500 chars (binario) |
| Brier + ECE | Calibración del confidence_score |

## Resultados experimentales

| Phase | Best combined | Métrica clave |
|---|---|---|
| Phase 2 (post-bugfix) | 1.686 | Faithfulness=0.947, Readability=1.0 |
| Phase 3 (DeepEval) | — | FaithfulnessMetric=0.960, GEval=0.749, 10/11 robustos |

## Hallazgos publicables

- **Bug del substring** detectado y corregido: `"INTO" in "DISTINTOS"` daba
  falso positivo en check de jerga SQL. Solucionado con regex de
  word-boundary. Combined +0.091.
- Q9 (NOT IN multi-tabla) es el caso problemático: AE simplifica las
  tablas intermedias al narrar.
