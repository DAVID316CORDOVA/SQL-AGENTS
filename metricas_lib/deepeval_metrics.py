"""
metricas_lib/deepeval_metrics.py

Metricas de evaluacion de salidas de agentes mediante DeepEval.

  - faithfulness_geval()  : GEval custom (LLM-juez) que mide Faithfulness
                            comparando la salida del agente (actual_output)
                            contra la respuesta esperada (expected_output).
                            Sigue la definicion de Amazon Web Services
                            (AWS Bedrock RAG Evaluation, 2024): "Faithfulness
                            means avoiding hallucination with respect to the
                            retrieved text chunks". En este proyecto el rol
                            de "retrieved text chunks" lo cumple la respuesta
                            esperada del dataset.

  - groundedness_geval()  : GEval custom (LLM-juez) que mide Groundedness
                            comparando la salida del agente (actual_output)
                            contra la entrada original del usuario (input).
                            Sigue la definicion de Microsoft (Azure AI Content
                            Safety, 2024): "the extent to which the model's
                            outputs are based on provided information,
                            avoiding speculation or fabrication". En este
                            proyecto el rol de "provided information" lo
                            cumple la entrada original (input).

  - rouge_l_score()       : ROUGE-L F1 usando el Scorer nativo de DeepEval
                            (deterministico, sin LLM-juez).

DeepEval no expone una metrica nativa llamada Groundedness; trata
"groundedness" como sinonimo de FaithfulnessMetric. Para tenerlas como
dos metricas separadas (con la distincion AWS vs Microsoft) usamos GEval
con criterio personalizado para cada una.

El juez LLM por defecto es gpt-4o. Se puede cambiar con la env var
EVAL_JUDGE_MODEL antes de importar este modulo.
"""

import os

from deepeval.metrics import GEval
from deepeval.test_case.llm_test_case import LLMTestCaseParams as _LLMParams
from deepeval.scorer import Scorer


# Modelo juez por defecto. gpt-4o tiene la mejor correlacion con humanos
# segun el paper original de G-Eval (Liu et al., 2023). Se sobrescribe
# por env var EVAL_JUDGE_MODEL si se quiere usar otro (Claude, Bedrock).
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o")


# ----------------------------------------------------------------------
# Faithfulness (definicion AWS Bedrock 2024)
# ----------------------------------------------------------------------

def faithfulness_geval() -> GEval:
    """
    Faithfulness = la salida del agente NO alucina respecto a la respuesta
    esperada. Compara `actual_output` vs `expected_output`. Score 0..1.

    Definicion AWS:
      "Faithfulness means avoiding hallucination with respect to the
      retrieved text chunks."
      (https://docs.aws.amazon.com/bedrock/latest/userguide/
       knowledge-base-eval-llm-results.html)
    """
    return GEval(
        name="Faithfulness",
        criteria=(
            "Determine if the actual_output is faithful to the expected_output, "
            "meaning it avoids hallucinations with respect to the expected_output. "
            "The actual_output is faithful when (a) it does not contradict any fact "
            "stated in the expected_output, and (b) it does not introduce additional "
            "claims not supported by the expected_output. The actual_output may "
            "rephrase, correct typos, or normalize formatting; that is acceptable. "
            "Penalize contradictions and unsupported additions; do not penalize "
            "stylistic differences."
        ),
        evaluation_params=[
            _LLMParams.ACTUAL_OUTPUT,
            _LLMParams.EXPECTED_OUTPUT,
        ],
        model=JUDGE_MODEL,
    )


# ----------------------------------------------------------------------
# Groundedness (definicion Microsoft Azure 2024)
# ----------------------------------------------------------------------

def groundedness_geval() -> GEval:
    """
    Groundedness = la salida del agente esta anclada en el input original;
    no especula ni fabrica info ausente. Compara `actual_output` vs `input`.
    Score 0..1.

    Definicion Microsoft (Azure AI Content Safety):
      "the extent to which the model's outputs are based on provided
      information, avoiding speculation or fabrication."
      (https://learn.microsoft.com/en-us/azure/ai-services/
       content-safety/concepts/groundedness)
    """
    return GEval(
        name="Groundedness",
        criteria=(
            "Determine if the actual_output is grounded in the input (the original "
            "user question or pipeline state). Grounded means the actual_output "
            "is based on the information present in the input, avoiding speculation "
            "or fabrication. Acceptable changes: fixing typos, normalizing casing, "
            "adding accents, or rephrasing while preserving meaning. NOT acceptable: "
            "introducing new entities, filters, constraints, or domain assumptions "
            "that were not present in the input. Penalize any hallucinated addition."
        ),
        evaluation_params=[
            _LLMParams.INPUT,
            _LLMParams.ACTUAL_OUTPUT,
        ],
        model=JUDGE_MODEL,
    )


# ----------------------------------------------------------------------
# GEval personalizado para AV (Validador SQL)
# ----------------------------------------------------------------------

def as_faithfulness_geval() -> GEval:
    """
    Faithfulness especifica para el AS (Sustentador).
    El AS justifica POR QUE el pipeline tomo ciertas decisiones.
    Criterio: la justificacion identifica el agente correcto y el
    razonamiento principal, independientemente de la redaccion exacta.
    """
    return GEval(
        name="AS_Faithfulness",
        criteria=(
            "You are evaluating a SQL pipeline sustainer agent that explains WHY "
            "the pipeline made specific decisions. "
            "The actual_output is the agent's justification and the expected_output "
            "is the reference justification. "
            "Score as faithful if: "
            "(a) both identify the SAME agent responsible (AR/APS/AG/AV/none), AND "
            "(b) both convey the SAME key reasoning (e.g. both say 'subquery for average', "
            "or both say 'similarity threshold filtered the table'). "
            "Do NOT penalize different wording, different level of detail, or different language. "
            "Only penalize if the actual_output attributes the decision to the WRONG agent "
            "or gives a completely different reason."
        ),
        evaluation_params=[
            _LLMParams.ACTUAL_OUTPUT,
            _LLMParams.EXPECTED_OUTPUT,
        ],
        model=JUDGE_MODEL,
    )


def as_groundedness_geval() -> GEval:
    """
    Groundedness especifica para el AS. Mide si la justificacion
    se basa en el SQL, tablas y query provistos, sin inventar
    informacion sobre el pipeline.
    """
    return GEval(
        name="AS_Groundedness",
        criteria=(
            "You are evaluating a SQL pipeline sustainer agent. "
            "The input contains the pipeline context: refined query, SQL generated, "
            "and tables/columns selected. "
            "The actual_output is the agent's justification of WHY the pipeline "
            "made those decisions. "
            "Score as grounded if: "
            "(a) any SQL constructs mentioned (JOIN, GROUP BY, subquery, BETWEEN, etc.) "
            "are actually present in the SQL from the input, AND "
            "(b) any table or column names mentioned exist in the provided schema. "
            "The agent MAY reference pipeline concepts (similarity threshold, APS, AR, AG, AV) "
            "as valid domain knowledge. "
            "Only penalize if the agent invents SQL constructs, tables, or columns "
            "that do not exist in the provided input."
        ),
        evaluation_params=[
            _LLMParams.INPUT,
            _LLMParams.ACTUAL_OUTPUT,
        ],
        model=JUDGE_MODEL,
    )


def av_faithfulness_geval() -> GEval:
    """
    Faithfulness especifica para el AV. El AV es un clasificador binario
    que emite un veredicto (valido/invalido) + razonamiento estructurado.

    Criterio: la salida es fiel si alcanza la MISMA CONCLUSION que el
    expected_output (valido vs invalido) y menciona el motivo correcto
    sin contradecirlo. No se penaliza la diferencia de formato ni estilo.
    """
    return GEval(
        name="AV_Faithfulness",
        criteria=(
            "You are evaluating a SQL validation agent's explanation. "
            "The actual_output is the agent's reasoning and the expected_output "
            "is the expected reference reasoning. "
            "Score the actual_output as faithful if: "
            "(a) it reaches the SAME conclusion (valid/approved vs invalid/error) "
            "as the expected_output, AND "
            "(b) if the SQL is invalid, it identifies the SAME TYPE of issue "
            "(wrong column, wrong table, forbidden pattern like SELECT*, COUNT(*), DML). "
            "Do NOT penalize differences in formatting, structure, language, or verbosity. "
            "Only penalize if the conclusion is OPPOSITE or the error type is completely different."
        ),
        evaluation_params=[
            _LLMParams.ACTUAL_OUTPUT,
            _LLMParams.EXPECTED_OUTPUT,
        ],
        model=JUDGE_MODEL,
    )


def av_groundedness_geval() -> GEval:
    """
    Groundedness especifica para el AV. Mide si el veredicto del AV
    esta basado en el SQL y schema recibidos, sin inventar columnas,
    tablas o errores que no existan en la consulta analizada.
    """
    return GEval(
        name="AV_Groundedness",
        criteria=(
            "You are evaluating a SQL validation agent. "
            "The input contains a SQL query and its schema (tables and columns). "
            "The actual_output is the agent's validation reasoning. "
            "Score the actual_output as grounded if: "
            "(a) any columns or tables mentioned in the reasoning actually appear "
            "in the SQL query or schema provided in the input, AND "
            "(b) any errors identified (missing column, wrong table, forbidden pattern) "
            "are verifiable from the SQL text itself. "
            "The agent MAY reference domain rules (e.g. SELECT* forbidden, COUNT(*) forbidden, "
            "DML not allowed) — these are valid domain constraints, not hallucinations. "
            "Only penalize if the agent invents column names, table names, or errors "
            "that do not exist in the provided SQL."
        ),
        evaluation_params=[
            _LLMParams.INPUT,
            _LLMParams.ACTUAL_OUTPUT,
        ],
        model=JUDGE_MODEL,
    )


# ----------------------------------------------------------------------
# ToolCorrectnessMetric wrapper — para evaluacion end-to-end
# ----------------------------------------------------------------------

def tool_correctness_metric(threshold: float = 0.5):
    """
    ToolCorrectnessMetric de DeepEval para evaluar si el pipeline
    activo los agentes correctos y cada agente invoco las skills correctas.

    Uso:
        from deepeval.test_case import LLMTestCase, ToolCall
        metric = tool_correctness_metric()
        tc = LLMTestCase(
            input=question,
            actual_output=explanation,
            tools_called=[ToolCall(name="AR"), ToolCall(name="APS"), ...],
            expected_tools=[ToolCall(name="AR"), ToolCall(name="APS"), ...],
        )
        metric.measure(tc)
        score = metric.score   # 0.0 - 1.0
        reason = metric.reason # explicacion de que fallo

    Convencion de nombres:
        - Agentes: "AR", "APS", "AG", "AV", "AE"
        - Skills:  "APS.search_tables", "AG.validate_sql_safety", etc.
    """
    from deepeval.metrics import ToolCorrectnessMetric as _TCM
    return _TCM(threshold=threshold)


def e2e_faithfulness_geval() -> GEval:
    """
    Faithfulness para evaluacion end-to-end.
    Compara la explicacion generada por AE contra la explicacion esperada.
    Misma definicion que faithfulness_geval() pero con criterio adaptado
    al contexto de evaluacion global del pipeline.
    """
    return GEval(
        name="E2E_Faithfulness",
        criteria=(
            "Determine if the actual_output (the pipeline's explanation to the user) "
            "is faithful to the expected_output (the reference explanation). "
            "Faithful means: (a) it conveys the same key information about the SQL query, "
            "(b) it does not contradict the expected explanation, and "
            "(c) it does not introduce information absent from the expected explanation. "
            "Do not penalize differences in style or verbosity."
        ),
        evaluation_params=[
            _LLMParams.ACTUAL_OUTPUT,
            _LLMParams.EXPECTED_OUTPUT,
        ],
        model=JUDGE_MODEL,
    )


def e2e_groundedness_geval() -> GEval:
    """
    Groundedness para evaluacion end-to-end.
    Mide si la explicacion del AE se ancla en el SQL generado y la query
    del usuario, sin fabricar informacion ausente en el pipeline.
    """
    return GEval(
        name="E2E_Groundedness",
        criteria=(
            "You are evaluating an end-to-end SQL pipeline. "
            "The input contains the user query, the SQL generated, and the tables used. "
            "The actual_output is the natural language explanation produced by the pipeline. "
            "Score as grounded if: "
            "(a) any SQL constructs mentioned (JOIN, GROUP BY, MAX, etc.) are present in the SQL, "
            "(b) any table or column names mentioned exist in the provided schema, "
            "(c) the explanation does not add facts absent from the SQL or input context. "
            "Only penalize if the explanation fabricates SQL operations or schema elements."
        ),
        evaluation_params=[
            _LLMParams.INPUT,
            _LLMParams.ACTUAL_OUTPUT,
        ],
        model=JUDGE_MODEL,
    )


# ----------------------------------------------------------------------
# ROUGE-L (Scorer nativo, deterministico)
# ----------------------------------------------------------------------

def rouge_l_score(actual_output: str, expected_output: str) -> float:
    """
    ROUGE-L F1 entre `actual_output` y `expected_output` usando el Scorer
    nativo de DeepEval (basado en rouge-score). Es deterministico, NO usa
    LLM-juez. Util como baseline lexico junto a las metricas LLM-juez.

    Args:
        actual_output: salida producida por el agente.
        expected_output: gold de referencia.

    Returns:
        F1 score en [0, 1]. 0 si alguno de los dos strings esta vacio.
    """
    if not actual_output or not expected_output:
        return 0.0
    # Atencion: la API de DeepEval recibe (target, prediction) en ese orden.
    return float(Scorer.rouge_score(
        target=expected_output,
        prediction=actual_output,
        score_type="rougeL",
    ))
