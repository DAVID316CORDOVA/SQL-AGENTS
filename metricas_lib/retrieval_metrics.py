# -*- coding: utf-8 -*-
"""
metricas_lib/retrieval_metrics.py

Metricas de retrieval para APS (Agente de Proximidad Semantica).

APS hace busqueda semantica con embeddings y retorna top-K elementos
(tablas y columnas). Se evalua contra la respuesta esperada con metricas
clasicas de information retrieval (sin LLM-juez).

Funciones:
  - precision_at_k(retrieved, expected, k=5) : de los top-K retornados,
                                               cuantos son correctos
  - recall(retrieved, expected)              : de los esperados, cuantos
                                               recupero
  - f1_retrieval(retrieved, expected, k=None) : armonica de P y R
  - mrr(retrieved_lists, expected_lists)      : Mean Reciprocal Rank
                                                (que tan arriba aparecen
                                                 los correctos)

Convenciones:
  - retrieved y expected son SETS o LISTS de identificadores (strings o tuplas)
  - Para tablas: nombres "stadium", "singer"
  - Para columnas: tuplas ("stadium", "Capacity") o "stadium.Capacity"
"""

from typing import Sequence, Iterable


def precision_at_k(retrieved: Sequence, expected: Iterable, k: int | None = None) -> float:
    """
    Precision@k: de los primeros K items retornados, que fraccion son correctos.

    Args:
        retrieved: lista ordenada por relevancia (top primero).
        expected:  conjunto de items correctos.
        k:         truncar a top-k. Si None, usa todos los retrieved.

    Returns:
        float en [0, 1]. 0.0 si no hay retrieved.
    """
    if not retrieved:
        return 0.0
    expected_set = set(expected)
    top_k = list(retrieved)[:k] if k else list(retrieved)
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in expected_set)
    return hits / len(top_k)


def recall(retrieved: Iterable, expected: Iterable) -> float:
    """
    Recall: de los items esperados, cuantos aparecen en retrieved.

    Returns:
        float en [0, 1]. 1.0 si expected esta vacio (caso degenerado).
    """
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    retrieved_set = set(retrieved)
    hits = len(retrieved_set & expected_set)
    return hits / len(expected_set)


def f1_retrieval(retrieved: Sequence, expected: Iterable, k: int | None = None) -> float:
    """
    F1 score = 2 * P * R / (P + R).

    Args:
        retrieved: lista ordenada (top primero).
        expected:  conjunto de items correctos.
        k:         truncar precision a top-k. Si None, usa todos.

    Returns:
        float en [0, 1].
    """
    p = precision_at_k(retrieved, expected, k=k)
    r = recall(list(retrieved)[:k] if k else retrieved, expected)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(retrieved_lists: Sequence[Sequence], expected_lists: Sequence[Iterable]) -> float:
    """
    Mean Reciprocal Rank sobre N queries.

    Para cada query: 1/rank del primer item correcto en retrieved.
    Si ninguno esta, contribuye 0.

    Args:
        retrieved_lists: list of N listas (una por query).
        expected_lists:  list of N conjuntos correctos (una por query).

    Returns:
        float en [0, 1]. 0 si ninguna query encontro item correcto.
    """
    if not retrieved_lists:
        return 0.0
    total_rr = 0.0
    n = len(retrieved_lists)
    for retrieved, expected in zip(retrieved_lists, expected_lists):
        expected_set = set(expected)
        for rank, item in enumerate(retrieved, start=1):
            if item in expected_set:
                total_rr += 1.0 / rank
                break
    return total_rr / n
