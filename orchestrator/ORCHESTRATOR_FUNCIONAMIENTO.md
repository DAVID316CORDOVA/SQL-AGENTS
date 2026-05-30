# Orchestrator — How It Works

The Orchestrator is a **pure router**: one LLM call (max 250 tokens), no pipeline execution,
no skills, no agentic loop. It classifies the user's intent and decides which node runs next.

---

## What the Orchestrator Does

1. Receives the raw user message
2. Checks STM (RAM cache) for an exact match → if hit, returns cached result immediately
3. Calls the LLM once with `ORCHESTRATOR_PROMPT` → gets `decision`, `enriched_query`, `reasoning`
4. Checks STM/LTM (vector cache) based on the decision
5. Returns routing info to the graph → the graph activates the correct node

The Orchestrator does NOT:
- Generate SQL
- Execute queries
- Reveal table or column names
- Answer questions outside its routing role (conversacional responses are a separate LLM call)

---

## The 6 Intent Categories

| Category | When | enriched_query | Next node |
|---|---|---|---|
| `new_sql_query` | New data question, no history reference | `""` (empty) | AR → APS → AG → AV → AE |
| `continuation` | Explicit refinement of the current query (OR / AND / REPLACEMENT) | Enriched query built from CURRENT STATE | AR → APS → AG → AV → AE |
| `sustentation` | User asks WHY or HOW the system reached its answer | `""` | AS |
| `context_switch` | User explicitly changes topic | Copy of new query | AR → APS → AG → AV → AE (history cleared) |
| `conversacional` | Greeting, small talk, conceptual definition, system instructions | `""` | END (short LLM reply) |
| `clarification_needed` | Continuation but ambiguous (which filter / operation unclear) | Two candidate options separated by " or " | END (asks user to clarify) |

---

## Decision Tree

```
user_input arrives
      │
      ▼
STM exact match?  (key = input.lower().strip(), Python dict lookup, 0 ms, no LLM)
      │
     YES ──────────────────────────────────────────► return cached result
      │
      NO
      │
      ▼
Pending clarification + user said "yes/ok/sure"?   (no LLM)
      │
     YES ──► convert to sql_query with the pending question
      │
      NO
      │
      ▼
LLM classifies intent  (1 LLM call, max 250 tokens)
      │
      ├── sustentation    ──► AS node directly         (no memory search)
      │
      ├── conversacional  ──► END + short LLM reply    (no memory search)
      │
      ├── clarification   ──► END + show options       (no memory search)
      │
      ├── new_sql_query
      │       │
      │       ├─ STM similarity ≥ 0.93  ──HIT──► return cached result
      │       │
      │       └─ LTM similarity ≥ 0.97  ──HIT──► return cached result
      │                                   MISS──► full pipeline AR→APS→AG→AV→AE
      │
      ├── continuation
      │       │   (uses enriched_query, not raw user_input)
      │       │
      │       ├─ STM exact only           ──HIT──► return cached result
      │       │
      │       └─ LTM similarity ≥ 0.99   ──HIT──► return cached result
      │                                    MISS──► full pipeline with enriched_query
      │
      └── context_switch
              │   (clears conversation history first)
              │
              ├─ STM similarity ≥ 0.97   ──HIT──► return cached result
              │
              └─ LTM similarity ≥ 0.97   ──HIT──► return cached result
                                           MISS──► full pipeline with new query
```

---

## Memory System

### STM (Short-Term Memory) — RAM dict, session only

**How entries are saved** (after each successful pipeline run):
```python
cache_key = enriched_query.lower().strip()   # plain string key
embedding  = get_embedding(enriched_query)    # vector from intfloat/multilingual-e5-base
stm[cache_key] = {
    "query":     enriched_query,
    "embedding": embedding,          # stored alongside the result
    "result":    { final_sql, ar_result, aps_result, ... }
}
```

**How search works:**
1. `key = question.lower().strip()` → `if key in stm` → **exact match, O(1), no model**
2. If not found and `exact_only=False`:
   - Compute embedding for new question
   - Cosine similarity against all stored embeddings (linear scan, max ~20 entries)
   - Hit if similarity ≥ threshold

The embedding model is **multilingual**: "singers from France" and "cantantes de Francia"
produce vectors with ~0.97 similarity → same cache entry.

### LTM (Long-Term Memory) — ChromaDB on disk, persists across sessions

Filled by `close_session()` which flushes STM → LTM at the end of each session.
Search is delegated to ChromaDB which uses the same embedding model internally.

### Thresholds (all configurable in config.py)

| `config.py` variable | Value | Used for |
|---|---|---|
| `MEMORY_SIMILARITY_THRESHOLD` | 0.93 | STM similarity — new_sql_query |
| `MEMORY_CONTINUATION_THRESHOLD` | 0.97 | LTM similarity — new_sql_query |
| `MEMORY_CONTINUATION_LTM_THRESHOLD` | 0.99 | LTM similarity — continuation |
| `MEMORY_CONTEXT_SWITCH_THRESHOLD` | 0.97 | STM + LTM — context_switch |

Why different thresholds:
- **0.93 (STM new_sql)**: synonyms score ~0.97, same question with extra filters scores ~0.91. 0.93 sits in between: accepts paraphrases, rejects variants with added constraints.
- **0.97 (LTM new_sql)**: LTM searches thousands of past entries → more noise → higher bar.
- **0.99 (LTM continuation)**: The enriched query is already specific ("France or Italy"). A false positive at 0.97 could match "France or Spain". Must be nearly identical.
- **0.97 (context_switch)**: After a topic change the new query may be slightly rephrased; 0.97 allows minor reformulations without being too loose.

Why **continuation uses STM exact-only**:
"students from Lima who study art" and "students from Lima who study art with GPA > 3.0"
have cosine similarity ~0.92. If STM used 0.93 for continuation, the second query would
hit the first entry and return the wrong (incomplete) result. Exact-only prevents this.

---

## Enrichment Rules for `continuation`

The `enriched_query` always starts from **CURRENT STATE** (the last successful query),
not from history. Three rules in priority order:

```
1. REPLACEMENT (highest priority)
   Keywords: "only", "just", "instead of", "now I want only"
   → Replace the matching filter in CURRENT STATE with the new value
   → Keep ALL other filters

   Example:
   CURRENT STATE: "how many students from Lima who study art"
   User: "only the ones from Cusco"
   → "how many students from Cusco who study art"

2. OR
   Apply only if CURRENT STATE already has a filter of the same type
   → Insert new value immediately after the existing same-type value
   → NEVER append at the end

   Example:
   CURRENT STATE: "how many students from Lima who study art"
   User: "and also from Cusco"
   → "how many students from Lima or Cusco who study art"
   NOT: "how many students from Lima who study art or from Cusco"  ← WRONG

3. AND
   New filter type not present in CURRENT STATE → append it
   → Keep ALL existing filters

   Example:
   CURRENT STATE: "how many students from Lima"
   User: "who study art"
   → "how many students from Lima who study art"
```

---

## 7 Concrete Examples (concert_singer dataset, fresh session)

**P1: "How many singers are from France?"**
```
STM exact → MISS (empty)
LLM → new_sql_query
STM similarity → MISS (empty)
LTM similarity → MISS (empty)
→ PIPELINE: SELECT COUNT(*) FROM singer WHERE country = 'France'
→ Saved to STM: key="how many singers are from france?"
```

**P2: "How many singers are from France?" (same question again)**
```
STM exact: "how many singers are from france?" IN stm → HIT
→ CACHED RESULT. No LLM call. No pipeline. ~0 ms.
```

**P3: "Cuántos cantantes son de Francia?"**
```
STM exact → MISS
LLM → new_sql_query
STM similarity: cosine("Cuántos cantantes son de Francia?", "How many singers are from France?")
              = 0.971 ≥ 0.93 → HIT (multilingual model recognizes same meaning)
→ CACHED RESULT.
```

**P4: "How many male singers are from France?"**
```
STM exact → MISS
LLM → new_sql_query
STM similarity: 0.908 < 0.93 → MISS  (extra filter "male" makes it different enough)
LTM similarity: 0.908 < 0.97 → MISS
→ PIPELINE: SELECT COUNT(*) FROM singer WHERE country='France' AND is_male=1
```

**P5: "And also from Italy?"**
```
STM exact → MISS
LLM → continuation  (implicit reference to P4 result)
  enriched_query = "How many male singers from France or Italy?"
  (Rule OR: "France" exists in CURRENT STATE, "Italy" is same filter type)
STM exact for enriched_query → MISS
LTM ≥ 0.99 for enriched_query → MISS
→ PIPELINE with enriched_query:
  SELECT COUNT(*) FROM singer WHERE country IN ('France','Italy') AND is_male=1
```

**P6: "Why did you use the country filter?"**
```
STM exact → MISS
LLM → sustentation  (contains "why did you use")
→ GOES DIRECTLY TO AS. No STM. No LTM. No AR→AG pipeline.
  AS receives P5 result and justifies the reasoning.
```

**P7: "Hello!"**
```
STM exact → MISS
LLM → conversacional
→ END. Separate LLM call (max 150 tokens, English, 3 lines max).
  No memory search. No pipeline.
```

---

## What the Orchestrator Does NOT Do

- Does not write or execute SQL
- Does not answer questions about the database content ("how many records are there?")
- Does not reveal table names, column names, or schema details
- Does not solve programming questions or explain code
- Does not answer out-of-scope questions (those go to AR which rejects them)

The conversacional handler responds in English, maximum 3 lines, using the classifier model.
It never invents data and redirects database questions to the pipeline.
