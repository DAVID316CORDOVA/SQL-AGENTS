# -*- coding: utf-8 -*-
"""
metricas_lib/classification_metrics.py

Metricas de clasificacion binaria para AV (Agente Validador).

Wrappers sobre sklearn.metrics que mantienen la firma simple del modulo
(listas plain de int/bool) y devuelven floats. Los usa AV Fase 1 y 3
porque el output del AV es un booleano (is_valid), no texto NL.

Funciones:
  - accuracy(true, pred)  : aciertos / total
  - precision(true, pred) : TP / (TP + FP)
  - recall(true, pred)    : TP / (TP + FN)
  - f1(true, pred)        : 2 * P * R / (P + R)
  - confusion(true, pred) : dict con TP, TN, FP, FN
"""

from typing import Sequence
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def _to_int_list(xs: Sequence) -> list[int]:
    """Convierte True/False/0/1 a 0/1 ints."""
    return [int(bool(x)) for x in xs]


def accuracy(true_labels: Sequence, predicted: Sequence) -> float:
    """Proporcion de aciertos. 1.0 = perfecto."""
    if not true_labels:
        return 0.0
    return float(accuracy_score(_to_int_list(true_labels), _to_int_list(predicted)))


def precision(true_labels: Sequence, predicted: Sequence,
              positive_class: int = 0) -> float:
    """
    Precision para la clase positiva.

    Por defecto positive_class=0 (inválido) porque el AV se mide por su
    capacidad de detectar SQL invalido sin marcar valido como invalido.

    TP / (TP + FP) — de los que el AV dijo "invalido", cuantos lo eran.
    """
    if not true_labels:
        return 0.0
    return float(precision_score(
        _to_int_list(true_labels), _to_int_list(predicted),
        pos_label=positive_class, zero_division=0,
    ))


def recall(true_labels: Sequence, predicted: Sequence,
           positive_class: int = 0) -> float:
    """
    Recall para la clase positiva (default = invalido).

    TP / (TP + FN) — de los realmente invalidos, cuantos detecto.
    """
    if not true_labels:
        return 0.0
    return float(recall_score(
        _to_int_list(true_labels), _to_int_list(predicted),
        pos_label=positive_class, zero_division=0,
    ))


def f1(true_labels: Sequence, predicted: Sequence,
       positive_class: int = 0) -> float:
    """
    F1 score para la clase positiva (default = invalido).

    Media armonica de precision y recall: 2 * P * R / (P + R).
    Es la metrica primaria para el combined_score del AV.
    """
    if not true_labels:
        return 0.0
    return float(f1_score(
        _to_int_list(true_labels), _to_int_list(predicted),
        pos_label=positive_class, zero_division=0,
    ))


def confusion(true_labels: Sequence, predicted: Sequence) -> dict:
    """
    Matriz de confusion. Devuelve dict con TP, TN, FP, FN considerando
    'invalido' (0) como la clase positiva (lo que se quiere detectar).

      TP: AV dijo invalido y era invalido      (acierto critico)
      TN: AV dijo valido y era valido          (acierto)
      FP: AV dijo invalido pero era valido     (falso positivo, molesto)
      FN: AV dijo valido pero era invalido     (falso negativo, peligroso)
    """
    if not true_labels:
        return {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    y_true = _to_int_list(true_labels)
    y_pred = _to_int_list(predicted)
    # sklearn devuelve [[TN, FP], [FN, TP]] con labels=[0, 1] por defecto.
    # Tratamos 0 (invalido) como positivo, asi que invertimos labels=[1, 0]:
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    # cm = [[TN_for_invalid, FP_for_invalid],
    #       [FN_for_invalid, TP_for_invalid]]
    return {
        "TN": int(cm[0][0]),
        "FP": int(cm[0][1]),
        "FN": int(cm[1][0]),
        "TP": int(cm[1][1]),
    }
