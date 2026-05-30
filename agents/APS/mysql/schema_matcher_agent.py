"""
agents/APS/mysql/schema_matcher_agent.py

APS Schema Matcher para MySQL.

Toda la logica del APS (busqueda vectorial en ChromaDB, integracion con MCP,
deteccion de tablas puente, verificacion de columnas, agentic loop con
skills) vive en este unico archivo. Las particularidades MySQL (prompt de
enriquecimiento, formato de identificadores con backticks) se inyectan
desde `prompt.py` y `skills.py` de esta misma carpeta.
"""

import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from llm_client import get_model, get_client_for_model
import chromadb
from chromadb.config import Settings
from agents.base_skill_agent import SkillAgent


class MysqlSchemaMatcherAgent(SkillAgent):

    # Umbrales de la decision en dos pasos del _detect_missing_columns:
    #   - Por encima de HIGH: claramente compatible (sin LLM).
    #   - Por debajo de LOW: claramente incompatible (sin LLM).
    #   - Entre ambos: zona gris, se consulta al LLM con contexto MCP de la BD.
    _MISSING_HIGH_THRESHOLD = 0.80
    _MISSING_LOW_THRESHOLD = 0.40

    def __init__(self, schema_path="agents/MCP/metadata/mysql/demo_db/schema.json",
                 vector_db_path="agents/APS/mysql/chroma/demo_db", db_type: str = "mysql",
                 top_n_tables: int = None, top_n_columns: int = None):
        self.schema_path = schema_path
        self.vector_db_path = vector_db_path
        self.db_type = "mysql"
        self.model = get_model("APS", backend="mysql")
        self.client = get_client_for_model(self.model)
        try:
            from config import EMBEDDING_MODEL, APS_TOP_N_TABLES, APS_TOP_N_COLUMNS
            self.embedding_model = EMBEDDING_MODEL
            self.top_n_tables  = int(top_n_tables)  if top_n_tables  is not None else APS_TOP_N_TABLES
            self.top_n_columns = int(top_n_columns) if top_n_columns is not None else APS_TOP_N_COLUMNS
        except ImportError:
            self.embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
            self.model = "claude-haiku-4-5"
            self.top_n_tables  = int(top_n_tables)  if top_n_tables  is not None else 5
            self.top_n_columns = int(top_n_columns) if top_n_columns is not None else 10
        self._agent_label = "APS-SchemaMatcher-MYSQL"
        self._st_model = None

        self.schema = self._load_schema()
        self._init_vector_db()

    # ------------------------------------------------------------------
    # Particularidades MySQL (lo unico que cambia entre motores)
    # ------------------------------------------------------------------

    def _load_skills_module(self):
        """Devuelve helpers MySQL (backticks, formato de JOIN para MySQL)."""
        from agents.APS.mysql.skills import format_identifier, get_join_hint
        return format_identifier, get_join_hint

    # ------------------------------------------------------------------
    # Infraestructura: schema, embeddings, ChromaDB
    # ------------------------------------------------------------------

    def _load_schema(self):
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"No existe {self.schema_path}")
        with open(self.schema_path, encoding="utf-8") as f:
            return json.load(f)

    def _get_embedding(self, text):
        from utils.embeddings import get_embedding
        return get_embedding(text, self.embedding_model, mode="query")

    def _get_embeddings_batch(self, texts):
        if not texts:
            return []
        from utils.embeddings import get_embeddings_batch
        return get_embeddings_batch(texts, self.embedding_model, mode="passage")

    def _get_best_similarity(self) -> str:
        """
        Lee la similarity ganadora del experimento APS Fase 1.
        Si el archivo no existe (todavia no se hizo Fase 1), default cosine.
        Valores validos en ChromaDB HNSW: "cosine", "ip", "l2".
        """
        path = os.path.join(os.path.dirname(__file__), "best_similarity.json")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            sim = data.get("similarity", "cosine")
            if sim not in {"cosine", "ip", "l2"}:
                sim = "cosine"
            return sim
        except (FileNotFoundError, json.JSONDecodeError):
            return "cosine"

    def _init_vector_db(self):
        os.makedirs(self.vector_db_path, exist_ok=True)
        self.chroma = chromadb.PersistentClient(
            path=self.vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        # Lee similarity del archivo de la Fase 1 (default cosine)
        sim = self._get_best_similarity()
        self.tables_col = self.chroma.get_or_create_collection(
            name="tables", metadata={"hnsw:space": sim})
        self.columns_col = self.chroma.get_or_create_collection(
            name="columns", metadata={"hnsw:space": sim})
        if not self._check_index_exists():
            self._index_schema()

    def _index_schema(self):
        entities = self.schema.get("available_entities", {})
        try:
            from config import EXCLUDED_TABLES
        except ImportError:
            EXCLUDED_TABLES = ["audit_log"]

        business_entities = {k: v for k, v in entities.items() if k not in EXCLUDED_TABLES}
        if not business_entities:
            print("  APS: No hay tablas de negocio para indexar")
            return

        dict_descriptions = {}
        try:
            from config import DICCIONARIO_PATH
            if os.path.exists(DICCIONARIO_PATH):
                with open(DICCIONARIO_PATH, encoding="utf-8") as f:
                    data_dict = json.load(f)
                for tname, tinfo in data_dict.get("tablas", {}).items():
                    table_desc = tinfo.get("descripcion", "")
                    col_descs = {}
                    for cname, cinfo in tinfo.get("columnas", {}).items():
                        if isinstance(cinfo, dict):
                            desc = cinfo.get("descripcion", cname)
                            ejemplo = cinfo.get("ejemplo", "")
                            col_descs[cname] = f"{desc} Ejemplo: {ejemplo}." if ejemplo else desc
                    dict_descriptions[tname] = {"table": table_desc, "columns": col_descs}
        except Exception:
            pass

        table_docs, table_ids, table_metas = [], [], []
        col_docs, col_ids, col_metas = [], [], []
        for table_name, info in business_entities.items():
            all_cols = info.get("all_columns", [])
            numeric = info.get("numeric_columns", [])
            categorical = info.get("categorical_columns", [])

            dd = dict_descriptions.get(table_name, {})
            col_desc_dict = dd.get("columns", {})

            doc = f"{table_name}."
            dict_desc = dd.get("table", "")
            if dict_desc:
                doc += f" {dict_desc}."
            doc += f" Columns: {', '.join(all_cols)}."

            table_docs.append(doc)
            table_ids.append(f"table:{table_name}")
            table_metas.append({"table_name": table_name})

            for col in all_cols:
                col_type = "numeric" if col in numeric else ("categorical" if col in categorical else "other")
                col_desc = col_desc_dict.get(col, col)
                col_doc = f"Column {col} in table {table_name}. {col_desc}. Type: {col_type}."
                col_docs.append(col_doc)
                col_ids.append(f"col:{table_name}.{col}")
                col_metas.append({"table_name": table_name, "column_name": col, "column_type": col_type})

        if table_docs:
            embeddings = self._get_embeddings_batch(table_docs)
            self.tables_col.add(ids=table_ids, embeddings=embeddings,
                                documents=table_docs, metadatas=table_metas)
        if col_docs:
            embeddings = self._get_embeddings_batch(col_docs)
            self.columns_col.add(ids=col_ids, embeddings=embeddings,
                                 documents=col_docs, metadatas=col_metas)

        print(f"  APS: Indexado - {len(table_docs)} tablas, {len(col_docs)} columnas")

    def _check_index_exists(self):
        try:
            count = self.tables_col.count()
            if count == 0:
                return False
            test_embedding = self._get_embedding("test")
            self.tables_col.query(query_embeddings=[test_embedding], n_results=1)
            return True
        except Exception as e:
            print(f"  [WARN] Indice corrupto o vacio: {e}")
            return False

    # ------------------------------------------------------------------
    # Busqueda vectorial (via MCP server — ChromaDB en el servidor)
    # ------------------------------------------------------------------

    def _search_tables(self, query, n=5):
        try:
            from config import ACTIVE_DATASET
            db_name = ACTIVE_DATASET
        except ImportError:
            db_name = "demo_db"
        if os.environ.get("AGENT_MCP_URL"):
            from agents.MCP.client import search_tables_client
            return search_tables_client(query, db_name, n=n, backend=self.db_type)
        from agents.MCP.tools.aps_tools import search_tables_server
        return search_tables_server(query, db_name, n=n, backend=self.db_type)

    def _search_columns(self, query, tables=None, n=10):
        try:
            from config import ACTIVE_DATASET
            db_name = ACTIVE_DATASET
        except ImportError:
            db_name = "demo_db"
        if os.environ.get("AGENT_MCP_URL"):
            from agents.MCP.client import search_columns_client
            return search_columns_client(query, db_name, tables=tables, n=n, backend=self.db_type)
        from agents.MCP.tools.aps_tools import search_columns_server
        return search_columns_server(query, db_name, tables=tables, n=n, backend=self.db_type)

    def _find_joins(self, tables):
        # JOIN discovery is AG's responsibility via find_joins_among_tables
        # and find_join_path skills. APS only retrieves relevant tables/columns.
        return []

    # ------------------------------------------------------------------
    # Skills del agente (lo que el LLM del APS puede invocar)
    # ------------------------------------------------------------------

    def _get_skills(self) -> list:
        return [
            {"type": "function", "function": {
                "name": "search_tables",
                "description": "Busca las tablas mas relevantes para la consulta usando similitud semantica en ChromaDB. Invoca esto siempre primero.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "Terminos de busqueda derivados de la consulta del usuario"},
                    "n": {"type": "integer", "description": "Numero maximo de tablas (default 5)", "default": 5}
                }, "required": ["query"]}
            }},
            {"type": "function", "function": {
                "name": "search_columns",
                "description": "Busca columnas relevantes para la consulta, filtradas por las tablas encontradas. Invoca despues de search_tables.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "Terminos de busqueda para columnas"},
                    "tables": {"type": "array", "items": {"type": "string"}, "description": "Nombres de tablas para filtrar (opcional)"},
                    "n": {"type": "integer", "description": "Numero maximo de columnas (default 10)", "default": 10}
                }, "required": ["query"]}
            }},
        ]

    def _execute_skill(self, skill_name: str, args: dict) -> str:
        try:
            if skill_name == "search_tables":
                return json.dumps(self._search_tables(args["query"], n=args.get("n", 5)))
            elif skill_name == "search_columns":
                return json.dumps(self._search_columns(args["query"], tables=args.get("tables"), n=args.get("n", 10)))
            else:
                return json.dumps({"error": f"Skill '{skill_name}' no reconocida en APS"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    # Pipeline principal: process()
    # ------------------------------------------------------------------

    def process(self, ar_result):
        if not ar_result.get("is_valid_query"):
            return {"success": False, "agent": "APS-SchemaMatcher",
                    "confidence_score": 0.0, "reasoning": "AR rechazo la consulta"}

        query = ar_result.get("refined_query", "")

        # Leer umbrales desde config (con fallback si no existe config)
        try:
            from config import CONFIDENCE_THRESHOLDS
            _thr_tables     = CONFIDENCE_THRESHOLDS.get("APS_TABLES",        0.55)
            _thr_columns    = CONFIDENCE_THRESHOLDS.get("APS_COLUMNS",       0.50)
            _gray_zone_high = CONFIDENCE_THRESHOLDS.get("APS_GRAY_ZONE_HIGH", 0.80)
        except ImportError:
            _thr_tables     = 0.55
            _thr_columns    = 0.50
            _gray_zone_high = 0.80

        # ── Paso 1: buscar y validar tablas ──────────────────────────────
        tables_list = self._search_tables(query, n=self.top_n_tables)
        if not tables_list:
            return {
                "success": False, "below_threshold": True,
                "below_threshold_reason": "tables",
                "agent": "APS-SchemaMatcher", "confidence_score": 0.0,
                "no_info_reason": "No tables in the schema are related to this query.",
                "reasoning": "ChromaDB returned empty results for tables",
                "skills_used": ["search_tables"]
            }

        top1_table_sim = tables_list[0].get("similarity", 0.0)
        if top1_table_sim < _thr_tables:
            return {
                "success": False, "below_threshold": True,
                "below_threshold_reason": "tables",
                "agent": "APS-SchemaMatcher",
                "confidence_score": round(top1_table_sim, 3),
                "no_info_reason": (
                    f"No tables in the schema are related to this query "
                    f"(best table similarity: {top1_table_sim:.2f}, "
                    f"minimum required: {_thr_tables:.2f})."
                ),
                "reasoning": f"Top-1 table similarity {top1_table_sim:.3f} < threshold {_thr_tables}",
                "skills_used": ["search_tables"]
            }

        # ── Paso 2: buscar y validar columnas ────────────────────────────
        table_names = [t["table_name"] for t in tables_list]
        columns_list = self._search_columns(query, tables=table_names, n=self.top_n_columns)

        if not columns_list:
            return {
                "success": False, "below_threshold": True,
                "below_threshold_reason": "columns",
                "agent": "APS-SchemaMatcher",
                "confidence_score": round(top1_table_sim, 3),
                "no_info_reason": (
                    "The matched tables do not have columns relevant to this query."
                ),
                "reasoning": "ChromaDB returned empty results for columns",
                "skills_used": ["search_tables", "search_columns"]
            }

        top1_col_sim = columns_list[0].get("similarity", 0.0)
        if top1_col_sim < _thr_columns:
            return {
                "success": False, "below_threshold": True,
                "below_threshold_reason": "columns",
                "agent": "APS-SchemaMatcher",
                "confidence_score": round(top1_col_sim, 3),
                "no_info_reason": (
                    f"The tables found do not have columns relevant to this query "
                    f"(best column similarity: {top1_col_sim:.2f}, "
                    f"minimum required: {_thr_columns:.2f})."
                ),
                "reasoning": f"Top-1 column similarity {top1_col_sim:.3f} < threshold {_thr_columns}",
                "skills_used": ["search_tables", "search_columns"]
            }

        # ── Zona gris: LLM verifica si la query es respondible ───────────
        if top1_col_sim < _gray_zone_high:
            matched_tables_preview = {}
            for t in tables_list:
                tname = t.get("table_name", "")
                if tname:
                    raw_info = self.schema.get("available_entities", {}).get(tname, {})
                    matched_tables_preview[tname] = raw_info.get("all_columns", t.get("all_columns", []))
            answerable, missing = self._llm_verify_answerability(query, matched_tables_preview)
            if not answerable:
                return {
                    "success": False, "below_threshold": True,
                    "below_threshold_reason": "llm_schema_mismatch",
                    "agent": "APS-SchemaMatcher",
                    "confidence_score": round(top1_col_sim, 3),
                    "no_info_reason": (
                        f"The schema does not contain the required data to answer this query"
                        f"{f' (missing: {missing})' if missing else ''}."
                    ),
                    "reasoning": f"LLM gray zone: query not answerable (missing: {missing})",
                    "skills_used": ["search_tables", "search_columns"]
                }

        # ── Paso 3: ambos umbrales superados → calcular confianza ponderada
        joins = []  # responsabilidad delegada al AG

        avg_table_sim = (sum(t.get("similarity", 0) for t in tables_list) / max(len(tables_list), 1))
        avg_col_sim   = (sum(c.get("similarity", 0) for c in columns_list[:5]) / max(len(columns_list[:5]), 1))
        confidence = round(avg_table_sim * 0.6 + avg_col_sim * 0.4, 3)

        reasoning = self._build_reasoning(query, tables_list, columns_list[:5],
                                          joins, confidence)

        tables_dict = {}
        for t in tables_list:
            tname = t.get("table_name", "")
            if not tname:
                continue
            raw_info = self.schema.get("available_entities", {}).get(tname, {})
            tables_dict[tname] = {
                "table_name": tname,
                "similarity": t.get("similarity", 0),
                "all_columns": raw_info.get("all_columns", t.get("all_columns", [])),
                "numeric_columns": raw_info.get("numeric_columns", t.get("numeric_columns", [])),
                "categorical_columns": raw_info.get("categorical_columns", t.get("categorical_columns", [])),
                "categorical_semantic": raw_info.get("categorical_semantic", t.get("categorical_semantic", {})) or {},
                "indexes": raw_info.get("indexes", t.get("indexes", {})),
                "partitions": raw_info.get("partitions", t.get("partitions")),
                "row_count": raw_info.get("row_count", t.get("row_count", 0)),
            }

        try:
            from config import DICCIONARIO_PATH
            if os.path.exists(DICCIONARIO_PATH):
                with open(DICCIONARIO_PATH, encoding="utf-8") as _f:
                    _dict = json.load(_f)
                for tname, tinfo in tables_dict.items():
                    col_descs = _dict.get("tablas", {}).get(tname, {}).get("columnas", {})
                    if col_descs:
                        sem = tinfo.get("categorical_semantic") or {}
                        for cname, cinfo in col_descs.items():
                            if not sem.get(cname):
                                desc = cinfo.get("descripcion", "") if isinstance(cinfo, dict) else ""
                                if desc:
                                    sem[cname] = desc
                        tinfo["categorical_semantic"] = sem
        except Exception:
            pass

        self._filter_columns(tables_dict, columns_list, joins)

        return {
            "success": True, "agent": "APS-SchemaMatcher",
            "input_query": query, "tables": tables_dict,
            "tables_list": tables_list[:3], "columns": columns_list[:10],
            "joins": joins,
            "confidence_score": confidence, "reasoning": reasoning,
            "skills_used": ["search_tables", "search_columns"]
        }

    # ------------------------------------------------------------------
    # Zona gris — verificacion LLM de answerability
    # ------------------------------------------------------------------

    def _llm_verify_answerability(self, query: str, matched_tables: dict) -> tuple:
        """
        Usa gpt-4o-mini para decidir si la query es respondible con las columnas
        de las tablas ya identificadas por busqueda vectorial.
        Solo se activa cuando top1_col_sim esta en [APS_COLUMNS, APS_GRAY_ZONE_HIGH).
        Fail-open: si el LLM falla, retorna (True, "") para no bloquear el pipeline.

        Args:
            query: la consulta del usuario
            matched_tables: dict {table_name: [col1, col2, ...]} — solo tablas matcheadas
        """
        schema_lines = []
        for tname, cols in matched_tables.items():
            schema_lines.append(f"  Table {tname}: {', '.join(cols)}")
        schema_text = "\n".join(schema_lines) if schema_lines else "(no tables matched)"

        prompt = (
            f"You are verifying whether a user query can be answered "
            f"with the available database columns.\n\n"
            f"Query: \"{query}\"\n\n"
            f"Available tables and their columns:\n"
            f"{schema_text}\n\n"
            f"Does the schema above contain the data needed to answer this query with SQL?\n\n"
            f"Rules:\n"
            f"- answerable: true  → all required entities and attributes exist in the schema, "
            f"even if SQL operations (NOT IN, JOIN, GROUP BY, COUNT, subqueries) must be used.\n"
            f"- answerable: false → ONLY when a key metric or attribute is completely absent "
            f"from all listed columns (e.g., asking for a monetary value when no price/amount "
            f"column exists anywhere).\n"
            f"- The absence of a single derived concept does NOT make a query unanswerable "
            f"if the underlying data can support it via SQL logic.\n"
            f"- Do NOT invent columns. Work only with what is listed.\n\n"
            f"GEOGRAPHIC TYPE RULE (apply before anything else):\n"
            f"- Geographic levels: country > state/province > city.\n"
            f"  A city column CANNOT answer a country or nationality filter.\n"
            f"- This rule triggers ONLY when the geographic term is:\n"
            f"    (a) A COUNTRY NAME  — e.g. Colombia, Peru, France, Germany, Brazil.\n"
            f"    (b) A NATIONALITY / DEMONYM — e.g. Colombian, Peruvian, French, Brazilian.\n"
            f"  If the query uses one of these AND no country-level column exists\n"
            f"  (country, pais, nationality, nacionalidad, nation, country_code, etc.)\n"
            f"  → answerable: false, missing_concept: 'country/nationality column'.\n"
            f"- This rule does NOT trigger for CITY NAMES, even capital cities:\n"
            f"    Bogotá, Lima, Paris, Berlin, Buenos Aires → city column is sufficient.\n"
            f"    A capital city is still a city. 'from Lima' ≠ 'from Peru'.\n"
            f"- When in doubt: a country is a sovereign nation; a city is a settlement\n"
            f"  inside a country. Classify accordingly.\n\n"
            f"Respond ONLY with valid JSON (no markdown):\n"
            f"{{\"answerable\": true/false, \"missing_concept\": \"...\" or null}}"
        )
        try:
            from llm_client import call_llm, get_client_for_model
            import re as _re, json as _json
            _gz_model = "gpt-4o-mini"
            _gz_client = get_client_for_model(_gz_model)
            result = call_llm(
                _gz_client, _gz_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=120,
            )
            text = result.get("content", "")
            m = _re.search(r"\{.*?\}", text, _re.DOTALL)
            if m:
                data = _json.loads(m.group())
                answerable = data.get("answerable", True)
                if isinstance(answerable, str):
                    answerable = answerable.strip().lower() in ("true", "1", "yes")
                missing = data.get("missing_concept") or ""
                return bool(answerable), str(missing)
        except Exception:
            pass
        return True, ""

    # ------------------------------------------------------------------
    # Verificacion de columnas faltantes (embedding + LLM zona gris, MCP)
    # ------------------------------------------------------------------

    def _detect_missing_columns(self, filter_concepts: list, tables_dict: dict) -> list:
        """
        Decide que filter_concepts no tienen columna compatible en el schema.

        Estrategia agnostica del dominio (sin reglas hardcodeadas):
          1. Para cada concepto, llamar a la tool MCP find_column_for_concept_client.
          2. Si la mejor similitud es alta -> compatible (sin LLM).
          3. Si no hay matches sobre el umbral bajo -> incompatible (sin LLM).
          4. Si esta en zona gris, consultar al LLM con el contexto MCP de la BD
             (pero sin reglas hardcodeadas en el prompt).
        """
        if not filter_concepts or not tables_dict:
            return []

        try:
            from agents.MCP.client import find_column_for_concept_client
            from config import ACTIVE_DATASET
        except Exception:
            return []

        table_names = list(tables_dict.keys())
        missing: list[str] = []
        gray_zone: list[tuple[str, list]] = []

        for concept in filter_concepts:
            res = find_column_for_concept_client(
                concept, ACTIVE_DATASET,
                tables=table_names,
                threshold=self._MISSING_LOW_THRESHOLD,
                top_n=10,
            )

            if not res.get("is_compatible"):
                missing.append(concept)
                continue

            best = res.get("best_similarity", 0.0)
            if best >= self._MISSING_HIGH_THRESHOLD:
                continue  # claramente compatible

            gray_zone.append((concept, res.get("matches", [])))

        if gray_zone:
            missing.extend(self._verify_gray_zone(gray_zone, tables_dict))

        return missing

    def _verify_gray_zone(self, gray_cases: list, tables_dict: dict) -> list:
        """
        Para los conceptos en zona gris, pregunta al LLM uno por uno si tienen
        columna compatible en el schema. Una pregunta por concepto evita que
        el LLM contagie su decision entre conceptos cuando algunos son
        compatibles y otros no.
        """
        try:
            from agents.MCP.client import get_database_description_client
            from config import ACTIVE_DATASET
            db_context = get_database_description_client(ACTIVE_DATASET)
            if "sin descripcion disponible" in (db_context or "").lower():
                db_context = ""
        except Exception:
            db_context = ""

        ctx_block = (
            f"Contexto de la base de datos:\n{db_context}\n\n"
            if db_context else ""
        )

        missing = []
        for concept, matches in gray_cases:
            if self._is_concept_missing(concept, matches, tables_dict, ctx_block):
                missing.append(concept)
        return missing

    def _is_concept_missing(self, concept: str, matches: list,
                            tables_dict: dict, ctx_block: str) -> bool:
        """
        Decide si UN concepto carece de columna compatible. Aislado por concepto.
        """
        def _column_info(table: str, column: str) -> str:
            tinfo = tables_dict.get(table) or {}
            sem = (tinfo.get("categorical_semantic") or {}).get(column, "")
            return f"descripcion: {sem}" if sem else "(sin descripcion)"

        if not matches:
            candidates_text = "(sin candidatos sobre el umbral minimo)"
        else:
            lines = []
            for m in matches[:5]:
                info = _column_info(m["table"], m["column"])
                lines.append(
                    f"  - {m['table']}.{m['column']} "
                    f"(sim={m['similarity']}) -- {info}"
                )
            candidates_text = "\n".join(lines)

        prompt = (
            f"{ctx_block}"
            f"El usuario quiere filtrar por: '{concept}'\n\n"
            f"Columnas candidatas (encontradas por busqueda vectorial):\n"
            f"{candidates_text}\n\n"
            f"Tu tarea: decidir si '{concept}' tiene alguna columna donde "
            f"almacenarlo. Razona en estos pasos y responde con todos los "
            f"campos.\n\n"
            f"IMPORTANTE sobre 'Ejemplos de valores' en las descripciones:\n"
            f"Los ejemplos que aparecen en una descripcion son SIEMPRE "
            f"REPRESENTATIVOS, NUNCA exhaustivos. Una columna que dice "
            f"'Ejemplos: Lima, Bogota, Madrid' almacena cualquier ciudad, no "
            f"solo esas tres. Si el concepto pertenece a la MISMA CATEGORIA "
            f"que los ejemplos, considera la columna compatible aunque el "
            f"concepto no este literalmente listado.\n\n"
            f"Responde SOLO JSON valido con esta estructura:\n"
            f"{{\n"
            f"  \"concept_type\": \"tipo o categoria general de '{concept}' (ej: ciudad, pais, materia academica, nombre de persona, fecha, numero, etc.)\",\n"
            f"  \"best_candidate\": \"tabla.columna mas relevante de la lista\",\n"
            f"  \"candidate_category\": \"categoria general de lo que almacena esa columna, generalizando los ejemplos a su tipo (ej: ejemplos 'Lima, Bogota' -> categoria 'ciudades')\",\n"
            f"  \"candidate_excludes\": \"resumen de la clausula 'NO almacena' si existe\",\n"
            f"  \"categories_match\": \"true si concept_type pertenece a candidate_category Y no esta en candidate_excludes; false en caso contrario\",\n"
            f"  \"compatible\": \"copia el valor de categories_match\"\n"
            f"}}\n\n"
            f"Reglas estrictas:\n"
            f"- Compara CATEGORIAS, no instancias. 'arte' pertenece a la "
            f"categoria 'materia academica' aunque no aparezca en los "
            f"ejemplos listados.\n"
            f"- Si candidate_excludes menciona explicitamente la categoria "
            f"del concepto (ej: excludes='paises' y concept es un pais), "
            f"compatible es false.\n"
            f"- Compatible es true cuando concept_type encaja en "
            f"candidate_category Y no esta excluido."
        )

        try:
            from llm_client import call_llm
            import re
            result = call_llm(
                self.client, self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=400,
            )
            text = result.get("content", "")
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                import json as _json
                data = _json.loads(m.group())
                # El LLM puede devolver bool real o string "true"/"false"
                compat = data.get("compatible", True)
                if isinstance(compat, str):
                    compat = compat.strip().lower() in ("true", "1", "yes", "si", "sí")
                return not bool(compat)
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Filtrado de columnas y construccion del razonamiento
    # ------------------------------------------------------------------

    def _filter_columns(self, tables_dict, columns_list, joins):
        relevant_by_table = {}
        for col in columns_list:
            tname = col.get("table_name", "")
            cname = col.get("column_name", "")
            if tname and cname:
                relevant_by_table.setdefault(tname, set()).add(cname)

        fk_cols_by_table = {}
        for j in joins:
            parts = j.get("join_hint", "").replace(" ", "").split("=")
            for part in parts:
                if "." in part:
                    tname, cname = part.split(".", 1)
                    fk_cols_by_table.setdefault(tname, set()).add(cname)

        for rel in self.schema.get("relationships", []):
            for side in [("from_table", "from_column"), ("to_table", "to_column")]:
                t, c = rel.get(side[0], ""), rel.get(side[1], "")
                if t and c:
                    fk_cols_by_table.setdefault(t, set()).add(c)

        for table_name, table_info in tables_dict.items():
            all_cols = table_info.get("all_columns", [])
            if not all_cols:
                continue
            keep = set()
            pk_info = table_info.get("indexes", {}).get("PRIMARY", {})
            for pk_col in pk_info.get("columns", []):
                keep.add(pk_col)
            for fk_col in fk_cols_by_table.get(table_name, set()):
                keep.add(fk_col)
            for col in relevant_by_table.get(table_name, set()):
                keep.add(col)
            for col in table_info.get("categorical_columns", []):
                keep.add(col)
            for col in all_cols:
                if col == "name" or col.endswith("_name") or col.endswith("_title") or col.endswith("_label"):
                    keep.add(col)
            if keep:
                table_info["all_columns"] = [c for c in all_cols if c in keep]
                cat_sem = table_info.get("categorical_semantic", {})
                if cat_sem:
                    table_info["categorical_semantic"] = {k: v for k, v in cat_sem.items() if k in keep}

    def _build_reasoning(self, query, tables, columns, joins, confidence):
        lines = [f"Busqueda para: '{query}'", "", "Tablas encontradas:"]
        for t in tables:
            lines.append(f"  - {t['table_name']}: {t['similarity']:.3f}")
        lines.extend(["", "Columnas relevantes:"])
        for c in columns[:5]:
            sim = c.get("similarity", 0)
            lines.append(f"  - {c.get('table_name','')}.{c.get('column_name','')}: {sim:.3f}")
        if joins:
            lines.extend(["", "JOINs:"])
            for j in joins:
                lines.append(f"  - {j['join_hint']}")
        lines.extend(["", f"Score: {confidence:.3f}"])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Mantenimiento
    # ------------------------------------------------------------------

    def reindex(self):
        try:
            self.chroma.delete_collection("tables")
            self.chroma.delete_collection("columns")
        except Exception:
            pass
        self.tables_col = self.chroma.get_or_create_collection(
            name="tables", metadata={"hnsw:space": "cosine"})
        self.columns_col = self.chroma.get_or_create_collection(
            name="columns", metadata={"hnsw:space": "cosine"})
        self.schema = self._load_schema()
        self._index_schema()

    def get_info(self):
        return {"name": self._agent_label, "model": self.embedding_model}
