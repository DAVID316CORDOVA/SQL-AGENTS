"""
metricas_lib

Modulo transversal de metricas de evaluacion compartido entre agentes.

Submodulos:
  - deepeval_metrics : Faithfulness y Groundedness via GEval (LLM-juez),
                       ROUGE-L via Scorer nativo (deterministico).
                       Lo usan AR, AE, AS (salidas en lenguaje natural)
                       y AG (con normalizacion SQL).
  - calibration      : Brier Score y ECE (estadisticos puros, sin LLM).
                       Lo usan TODOS los agentes que reportan confidence.
  - sql_metrics      : Normalizacion SQL + ROUGE-L sobre SQL.
                       Lo usa AG.
"""

from metricas_lib.deepeval_metrics import (
    faithfulness_geval,
    groundedness_geval,
    av_faithfulness_geval,
    av_groundedness_geval,
    as_faithfulness_geval,
    as_groundedness_geval,
    e2e_faithfulness_geval,
    e2e_groundedness_geval,
    tool_correctness_metric,
    rouge_l_score,
    JUDGE_MODEL,
)
from metricas_lib.calibration import brier_score, ece
from metricas_lib.sql_metrics import (
    normalize_sql,
    normalize_sql_v1,
    rouge_l_sql,
    rouge_l_sql_v1,
    rouge_l_sql_v2,
)
from metricas_lib.classification_metrics import (
    accuracy,
    precision,
    recall,
    f1,
    confusion,
)
from metricas_lib.retrieval_metrics import (
    precision_at_k,
    recall as recall_retrieval,
    f1_retrieval,
    mrr,
)

__all__ = [
    # LLM-juez (DeepEval GEval)
    "faithfulness_geval",
    "groundedness_geval",
    "av_faithfulness_geval",
    "av_groundedness_geval",
    "as_faithfulness_geval",
    "as_groundedness_geval",
    # Determinista lexical (DeepEval Scorer)
    "rouge_l_score",
    "rouge_l_sql",
    "rouge_l_sql_v1",
    "rouge_l_sql_v2",
    "normalize_sql",
    "normalize_sql_v1",
    # Calibracion (sklearn / numpy)
    "brier_score",
    "ece",
    # Clasificacion (sklearn) — para AV
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion",
    # Retrieval — para APS
    "precision_at_k",
    "recall_retrieval",
    "f1_retrieval",
    "mrr",
    # Constantes
    "JUDGE_MODEL",
]
