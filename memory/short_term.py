"""
short_term.py

Memoria a corto plazo del sistema NL->SQL.
Vive en RAM, dura lo que dura la sesion.
Busca preguntas similares por similitud coseno de embeddings.

Ubicacion: memory/short_term.py
"""

import os
import time
import numpy as np

try:
    from config import (
        EMBEDDING_MODEL, MEMORY_SIMILARITY_THRESHOLD,
        MEMORY_SHORT_TERM_ENABLED
    )
except ImportError:
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    MEMORY_SIMILARITY_THRESHOLD = 0.85
    MEMORY_SHORT_TERM_ENABLED = True


class ShortTermMemory:
    """
    Almacena en RAM las preguntas y respuestas de la sesion actual.
    Busca por similitud coseno entre embeddings.
    Cachea TODO: queries exitosas Y rechazos de AR/APS.
    """

    def __init__(self):
        self.enabled = MEMORY_SHORT_TERM_ENABLED
        self.threshold = MEMORY_SIMILARITY_THRESHOLD
        self.entries = []  # [{query, embedding, result, timestamp}]
        self._st_model = None

    @property
    def _embedding_model(self):
        if self._st_model is None:
            from utils.embeddings import get_model
            self._st_model = get_model(EMBEDDING_MODEL)
        return self._st_model

    def _get_embedding(self, text):
        try:
            return np.array(self._embedding_model.encode(text))
        except Exception:
            return None

    def _cosine_similarity(self, a, b):
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def search(self, query):
        """
        Busca si hay una pregunta similar en memoria.
        Retorna dict con resultado si similitud >= threshold, o None.
        Incluye top 3 similitudes para debug.
        """
        if not self.enabled or not self.entries:
            return None

        query_emb = self._get_embedding(query)
        if query_emb is None:
            return None

        # Calcular similitud con todas las entries
        scored = []
        for entry in self.entries:
            sim = self._cosine_similarity(query_emb, entry["embedding"])
            scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Top 3 para debug
        top3 = [(s, e["query"]) for s, e in scored[:3]]

        best_sim, best_entry = scored[0] if scored else (0.0, None)

        if best_sim >= self.threshold and best_entry is not None:
            return {
                "matched": True,
                "similarity": round(best_sim, 4),
                "original_query": best_entry["query"],
                "result": best_entry["result"],
                "source": "short_term",
                "top3": top3
            }

        # No match pero devolver info de debug
        return {
            "matched": False,
            "similarity": round(best_sim, 4) if scored else 0.0,
            "closest_query": scored[0][1]["query"] if scored else "",
            "top3": top3,
            "threshold": self.threshold
        }

    def store(self, query, result):
        """
        Guarda una pregunta y su resultado en memoria.
        Guarda TODO: rechazos de AR, rechazos de APS, y queries exitosas.
        """
        if not self.enabled:
            return

        embedding = self._get_embedding(query)
        if embedding is None:
            return

        self.entries.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "timestamp": time.time()
        })

    def size(self):
        return len(self.entries)

    def clear(self):
        self.entries.clear()

    def load_from_long_term(self, entries):
        """
        Carga entries del largo plazo al iniciar sesion.
        Esto permite que al reiniciar, las preguntas de la ultima hora
        ya esten disponibles en cache sin re-ejecutar el flujo.

        Cada entry debe tener: query, result.
        Se genera el embedding al cargar.
        """
        if not self.enabled:
            return 0

        loaded = 0
        for entry in entries:
            query = entry.get("query", "")
            result_data = entry.get("result")
            if not query or not result_data:
                continue

            embedding = self._get_embedding(query)
            if embedding is None:
                continue

            self.entries.append({
                "query": query,
                "embedding": embedding,
                "result": result_data,
                "timestamp": entry.get("timestamp", 0)
            })
            loaded += 1

        return loaded