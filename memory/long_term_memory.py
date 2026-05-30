"""
long_term_memory.py

Memoria a largo plazo con ChromaDB.

Cada combinacion (usuario, backend, dataset) tiene su propio cliente
persistente con ruta independiente:

    memoria_chroma/<backend>/<usuario>/<dataset>/chroma.sqlite3

La coleccion dentro de cada cliente es siempre "chat".

Ubicacion: memory/long_term_memory.py
"""

import json
import os
import re
import time
import chromadb

try:
    from config import (
        MEMORY_LONG_TERM_ENABLED, EMBEDDING_MODEL,
        MEMORY_SIMILARITY_THRESHOLD, MEMORY_CHROMA_BASE
    )
except ImportError:
    MEMORY_LONG_TERM_ENABLED = True
    MEMORY_CHROMA_BASE = "memoria_chroma"
    MEMORY_SIMILARITY_THRESHOLD = 0.93
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _slug(text: str) -> str:
    """Normaliza un texto para usarlo como nombre de carpeta."""
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_") or "default"


class LongTermMemory:
    """
    Memoria a largo plazo con ChromaDB.
    Una coleccion por usuario: memory_{username}
    Almacena embeddings de preguntas con metadata (SQL, confianza, resultado).
    """

    def __init__(self, username, db_type="mysql", active_dataset="demo_db"):
        self.enabled = MEMORY_LONG_TERM_ENABLED
        self.username = username
        self.db_type = db_type
        self.active_dataset = active_dataset
        self.threshold = MEMORY_SIMILARITY_THRESHOLD
        # Path persistente por (backend, usuario, dataset).
        # Se crea on-demand al pedir self.collection.
        self.path = os.path.join(
            MEMORY_CHROMA_BASE,
            _slug(db_type),
            _slug(username),
            _slug(active_dataset),
        )
        self._client = None
        self._collection = None
        self._st_model = None

    @property
    def _embedding_model(self):
        if self._st_model is None:
            from utils.embeddings import get_model
            self._st_model = get_model(EMBEDDING_MODEL)
        return self._st_model

    @property
    def collection(self):
        if self._collection is None:
            os.makedirs(self.path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.path)
            self._collection = self._client.get_or_create_collection(
                name="chat",
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def _get_embedding(self, text):
        try:
            return self._embedding_model.encode(text).tolist()
        except Exception:
            return None

    def search(self, query, threshold: float = None):
        """
        Busca si hay una pregunta similar en ChromaDB.
        Retorna dict con resultado si similitud >= threshold, o None.
        threshold: umbral de similitud; si no se pasa usa self.threshold.
        """
        if not self.enabled:
            return None

        effective_threshold = threshold if threshold is not None else self.threshold

        embedding = self._get_embedding(query)
        if embedding is None:
            return None

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=3,
                include=["metadatas", "distances", "documents"]
            )
        except Exception:
            return None

        if not results["documents"] or not results["documents"][0]:
            return None

        # ChromaDB con cosine devuelve distancia (0 = identico, 2 = opuesto)
        # Similitud = 1 - distancia
        best_distance = results["distances"][0][0]
        best_similarity = 1 - best_distance
        best_doc = results["documents"][0][0]
        best_meta = results["metadatas"][0][0]

        top3 = []
        for i in range(min(3, len(results["documents"][0]))):
            sim = 1 - results["distances"][0][i]
            q = results["documents"][0][i]
            top3.append((round(sim, 4), q))

        if best_similarity >= effective_threshold:
            result_data = None
            if best_meta.get("result_json"):
                try:
                    result_data = json.loads(best_meta["result_json"])
                except (json.JSONDecodeError, TypeError):
                    pass

            return {
                "hit": True,
                "similarity": round(best_similarity, 4),
                "original_query": best_doc,
                "cached_result": result_data,
                "top3": top3,
            }

        return {
            "hit": False,
            "similarity": round(best_similarity, 4),
            "original_query": best_doc,
            "top3": top3,
        }

    def store(self, query, result, original_query: str = None):
        """
        Guarda pregunta y resultado en ChromaDB con metadata enriquecida.
        Si original_query != query, guarda ambas formas para maximizar recall.
        Solo guarda queries validas (que pasaron AR).
        """
        if not self.enabled:
            return

        ar = result.get("ar_result") or {}
        if not ar.get("is_valid_query"):
            return

        ae = result.get("ae_result") or {}
        explanation = (ae.get("explanation") or {}).get("final_reasoning", "")
        ag = result.get("ag_result") or {}
        ag_reasoning = ag.get("reasoning", "")
        aps = result.get("aps_result") or {}
        tables = json.dumps(aps.get("relevant_tables", []), ensure_ascii=False)
        columns = json.dumps(aps.get("relevant_columns", []), ensure_ascii=False)

        result_json = json.dumps(self._serialize_result(result), ensure_ascii=False)

        base_metadata = {
            "username": self.username,
            "timestamp": int(time.time()),
            "sql": result.get("final_sql", ""),
            "confidence": float(result.get("overall_confidence", 0)),
            "explanation": explanation[:500],
            "ag_reasoning": ag_reasoning[:300],
            "tables": tables[:500],
            "columns": columns[:500],
            "enriched_query": query[:300],
            "result_json": result_json[:40000],
        }

        ts = int(time.time() * 1000)

        # Siempre guarda la forma enriquecida
        emb = self._get_embedding(query)
        if emb:
            try:
                self.collection.add(
                    ids=[f"{self.username}_{ts}"],
                    embeddings=[emb],
                    documents=[query],
                    metadatas=[base_metadata]
                )
            except Exception as e:
                print(f"  [MEMORIA] Error guardando en ChromaDB: {e}")

        # Guardar forma original SOLO si es una pregunta completa (no encadenamiento).
        # Las queries de encadenamiento ("de esos...", "entonces...") son demasiado
        # ambiguas como claves LTM — matchean con queries de otros paises/contextos.
        chaining_starts = ("de esos", "de ellos", "de ellas", "de esas",
                           "cuantos de", "y de", "entonces", "y cuantos",
                           "tambien", "ademas", "de ahi", "de ese", "de esa")
        is_chaining = any(
            (original_query or "").lower().strip().startswith(p)
            for p in chaining_starts
        )
        if (original_query and
                original_query.strip().lower() != query.strip().lower() and
                not is_chaining):
            emb2 = self._get_embedding(original_query)
            if emb2:
                meta2 = {**base_metadata, "enriched_query": query[:300]}
                try:
                    self.collection.add(
                        ids=[f"{self.username}_{ts}_orig"],
                        embeddings=[emb2],
                        documents=[original_query],
                        metadatas=[meta2]
                    )
                except Exception as e:
                    print(f"  [MEMORIA] Error guardando original en ChromaDB: {e}")

    def _serialize_result(self, result):
        safe = {}
        for key in ["ar_result", "aps_result", "ag_result", "av_result",
                     "ae_result", "final_sql", "overall_confidence",
                     "iteration_count", "iteration_history"]:
            val = result.get(key)
            if val is not None:
                try:
                    json.dumps(val)
                    safe[key] = val
                except (TypeError, ValueError):
                    safe[key] = self._clean(val)
        return safe

    def _clean(self, obj):
        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean(i) for i in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def clear(self):
        try:
            ids = self.collection.get()["ids"]
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def clear_last_hour(self):
        """Elimina entradas guardadas en la última hora."""
        try:
            data = self.collection.get(include=["metadatas"])
            ids     = data.get("ids", [])
            metas   = data.get("metadatas", [])
            cutoff  = time.time() - 3600
            to_delete = [i for i, m in zip(ids, metas) if m.get("timestamp", 0) >= cutoff]
            if to_delete:
                self.collection.delete(ids=to_delete)
            return len(to_delete)
        except Exception:
            return 0

    def get_recent(self, n: int = 5) -> list:
        """Retorna las n preguntas mas recientes guardadas en LTM."""
        try:
            data = self.collection.get(include=["documents", "metadatas"])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])
            pairs = sorted(
                zip(metas, docs),
                key=lambda x: x[0].get("timestamp", 0),
                reverse=True
            )
            seen = set()
            result = []
            for meta, doc in pairs:
                q = (meta.get("enriched_query") or doc or "").strip()
                if q and q not in seen:
                    seen.add(q)
                    result.append(q)
                if len(result) >= n:
                    break
            return result
        except Exception:
            return []

    def get_stats(self):
        try:
            count = self.collection.count()
            return {"total": count, "username": self.username,
                    "path": self.path}
        except Exception:
            return {"total": 0, "username": self.username,
                    "path": self.path}