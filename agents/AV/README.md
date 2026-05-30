# Agente Validador (AV)

## Rol

Recibe el SQL del AG y verifica que sea válido, seguro y ejecutable contra
el backend correspondiente. Si detecta problemas, devuelve feedback que el
AG usa para corregir.

Es el "loop de calidad" entre AG y AV — pueden iterar varias veces hasta que
el AV apruebe el SQL o se agoten los reintentos.

## Modelo y configuración

Cargado desde `experiments/winners/best_hyperparameters.json`:

- model: gpt-4o
- temperature: 0.1
- combined_score Phase 2 (mysql): 1.625
- combined_score Phase 2 (postgres): 1.625

## Recursos externos que usa

| Recurso | Para qué |
|---|---|
| OpenAI API (gpt-4o) | Razonamiento sobre validez |
| Skills agentic propias | `check_syntax_rules`, `check_semantic_patterns`, `run_explain` |
| **MySQL/PostgreSQL real** (vía `EXPLAIN`) | Verificar que el SQL es ejecutable |

**Sí accede a la BD real** — pero solo con `EXPLAIN ANALYZE`, no ejecuta el SQL completo.

## Sub-carpetas mysql/ y postgres/

AV es **multi-backend**:

- `agents/AV/mysql/sql_validator_agent.py`
- `agents/AV/postgres/sql_validator_agent.py`

Diferencias: sintaxis de EXPLAIN, mensajes de error específicos del motor,
patrones a detectar.

## Skills agentic

| Skill | Para qué |
|---|---|
| `check_syntax_rules` | Verificación rápida programática (SELECT *, COUNT(*), DML, JOIN sin ON) |
| `check_semantic_patterns` | Detecta patrones problemáticos (window functions en WHERE, LIMIT global cuando debe ser PARTITION BY) |
| `run_explain` | Ejecuta EXPLAIN contra la BD real para validar plan de ejecución |

## Flujo de una llamada

1. Recibe `ag_result` con el SQL generado.
2. Ejecuta skills programáticas (syntax + semantic) — todo determinístico.
3. Ejecuta `EXPLAIN ANALYZE` contra MySQL/Postgres real.
4. Si encuentra errores, retorna `is_valid=False` + lista de errores +
   feedback que el AG usará para corregir.
5. Si está OK, retorna `is_valid=True` y el SQL pasa al AE.

## Iteraciones AG ↔ AV

El loop puede ejecutarse hasta `MAX_RETRIES=3` veces. Si después de 3
intentos el AV sigue rechazando, el orquestador desiste y reporta error al
usuario.

## Estructura de carpetas

```
agents/AV/
├── mysql/
│   ├── prompt.py
│   ├── sql_validator_agent.py
│   ├── agentic_skills.py
│   ├── functions.py
│   └── phase_2/
├── postgres/
│   └── (mismo layout)
└── comparison/comparison.py
```

## Resultados experimentales (Phase 2)

| Backend | Precision | Recall | F1 | Combined |
|---|---|---|---|---|
| MySQL | 1.0 | 1.0 | 1.0 | 1.625 |
| Postgres | 1.0 | 1.0 | 1.0 | 1.625 |

Ambos backends dieron el mismo combined exacto — el dataset de prueba (10
casos sintéticos con SQL inválidos) es manejado al 100% por gpt-4o.

**No tiene Phase 3** — DeepEval LLM-as-judge no aplica para detección
binaria de errores. Las métricas estándar son Precision/Recall sobre
detección de invalidez.
