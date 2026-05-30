"""
metricas_lib/calibration.py

Brier Score y Expected Calibration Error (ECE).

Estas dos metricas son ESTADISTICAS sobre N muestras: necesitan listas
de confianzas predichas y outcomes binarios. NO se pueden calcular con
un LLM-juez ni con DeepEval; se calculan con sklearn y NumPy.

Se invocan al final de un estudio Optuna (o de cualquier evaluacion),
una vez recolectadas las N evaluaciones de cada trial.
"""

import numpy as np
from sklearn.metrics import brier_score_loss


def brier_score(confidences: list[float], outcomes: list[int]) -> float:
    """
    Brier Score = mean((confidence_i - outcome_i)^2)

    Cuantifica el error cuadratico medio entre la confianza declarada y
    el outcome real (1 si el agente acerto, 0 si no). Menor es mejor;
    0.0 = calibracion perfecta.

    Args:
        confidences: lista de floats en [0,1], confianza reportada por
                     el agente para cada uno de los N casos.
        outcomes:    lista de enteros 0 o 1, outcome real para cada caso.

    Returns:
        Brier Score como float en [0,1].
    """
    if not confidences:
        return 0.0
    if len(confidences) != len(outcomes):
        raise ValueError(
            f"confidences y outcomes deben tener el mismo largo "
            f"({len(confidences)} vs {len(outcomes)})"
        )
    return float(brier_score_loss(outcomes, confidences))


def ece(confidences: list[float], outcomes: list[int],
        n_bins: int = 5) -> float:
    """
    Expected Calibration Error (ECE).

    Agrupa las predicciones en n_bins por nivel de confianza y promedia
    la diferencia absoluta entre confianza media y accuracy real en cada
    bin, ponderada por el tamano del bin. Menor es mejor; 0.0 = perfecta
    calibracion por intervalos.

    Args:
        confidences: lista de floats en [0,1].
        outcomes:    lista de 0/1.
        n_bins:      cuantos intervalos usar (5 por defecto).

    Returns:
        ECE como float en [0,1].
    """
    if not confidences:
        return 0.0
    if len(confidences) != len(outcomes):
        raise ValueError(
            f"confidences y outcomes deben tener el mismo largo "
            f"({len(confidences)} vs {len(outcomes)})"
        )

    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for c, o in zip(confidences, outcomes):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, o))

    total = len(confidences)
    return float(sum(
        len(b) / total * abs(
            np.mean([c for c, _ in b]) - np.mean([o for _, o in b])
        )
        for b in bins if b
    ))
