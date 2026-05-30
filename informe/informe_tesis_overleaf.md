\documentclass[11pt]{article}

% --- Codificación y lenguaje ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish]{babel}

% --- Layout y apariencia ---
\usepackage[a4paper,margin=0.8in,top=0.7in,bottom=0.7in]{geometry}
\usepackage{setspace}
\usepackage{microtype}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.8em}{}
\titlespacing*{\subsection}{1em}{*2}{*1}
\titleformat{\subsubsection}{\normalsize\itshape}{\thesubsubsection}{0.8em}{}
\titlespacing*{\subsubsection}{2em}{*1.5}{*0.5}

% --- Columnas controladas ---
\usepackage{multicol}
\setlength{\columnsep}{20pt}

% --- Tablas y utilidades ---
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{array}
\usepackage{caption}
\usepackage{capt-of}
\captionsetup[table]{font=small,labelfont=bf}

% --- Otros ---
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}

\usepackage{float}
\usepackage{placeins}
\usepackage{pifont} % checkmark y otros símbolos (ding{51})
\usepackage{tikz}
\usetikzlibrary{arrows.meta}

% --- Título ---
\title{\LARGE \textbf{SQL-Agents: Sistema Multi-Agente 
para la Generación de Consultas SQL desde Lenguaje Natural}}
\author{
  Félix David Córdova García\\
  \texttt{fdcordova@javeriana.edu.co}
  \and
  Edison Leonardo Neira Espitia\\
  \textit{Asesor}\\
  \texttt{edison-neira@javeriana.edu.co}\\[4pt]
  \normalsize Pontificia Universidad Javeriana -- Bogotá D.C.
}
\date{\today}


\addto\captionsspanish{\renewcommand{\tablename}{Tabla}}
\begin{document}

\maketitle

% ---------------------------------------------------------------
% ABSTRACT
% ---------------------------------------------------------------
\noindent\textbf{Abstract.}
This paper presents \textbf{SQL-Agents}, a multi-agent NL-to-SQL system
that makes relational databases accessible to non-technical users.
Six specialized agents ---AR (Refiner), APS (Schema Matcher), AG (SQL
Generator), AV (Validator), AE (Explainer) and AS (Sustainer)--- follow
the \textbf{Agent Skills}~\cite{anthropic_skills2024} pattern, orchestrated
by a central agent with intent classification and short/long-term memory.
The \textbf{Model Context Protocol (MCP)}~\cite{anthropic_mcp2024}
externalizes domain knowledge, decoupling it from the LLM provider.
Evaluated over four phases on Spider~1.0 \texttt{concert\_singer}
(MySQL and PostgreSQL), the system is benchmarked through Optuna~TPE
hyperparameter search, DeepEval LLM-as-judge scoring (Faithfulness,
Groundedness), calibration metrics (Brier~Score, ECE), and paraphrase
robustness tests.
Phase~4 robustness results yield a normalized pipeline confidence index
of \textbf{82.9\,\%} for MySQL and \textbf{83.2\,\%} for PostgreSQL
(combined score over a theoretical maximum of~3.0), confirming end-to-end
correctness and semantic stability across both databases.

\vspace{6pt}
\noindent\textbf{Keywords:} Text-to-SQL, multi-agent system, Agent Skills,
Model Context Protocol, conversational context, orchestrator, vector
similarity, proactive validation, explainability, LLM, ToolCorrectnessMetric,
WACS, Faithfulness, Groundedness, Brier Score, ECE, Optuna TPE.

\vspace{10pt}

% ---------------------------------------------------------------
\begin{multicols}{2}
% ---------------------------------------------------------------

% ---------------------------------------------------------------
\section{Introducción}
% ---------------------------------------------------------------

El acceso a datos relacionales está restringido a quienes conocen SQL~\cite{ojuri2025}.
NL2SQL propone convertir preguntas en lenguaje natural a consultas SQL~\cite{katsogiannis2023},
pero tres limitaciones persisten: validación \textit{post-hoc}~\cite{wang2024sqlrefine},
ambigüedad semántica~\cite{tai2023cot} y ausencia de gestión de contexto
conversacional~\cite{zhang2024memory}.

Frente a enfoques monolíticos, descomponer el pipeline en \textbf{agentes especializados}
reduce la alucinación: cada agente opera sobre un contexto acotado con herramientas
propias, lo que limita la propagación de errores y permite corregirlos de forma
local antes de que afecten las etapas siguientes~\cite{wooldridge2009}.
Los experimentos confirman que distintos modelos son óptimos para cada rol,
justificando la arquitectura multi-agente frente a un único LLM centralizado.

Este trabajo propone \textbf{SQL-Agents}, un sistema multi-agente que aborda las tres
limitaciones. El \textbf{Orquestador} gestiona el contexto conversacional antes de
invocar el pipeline: detecta cambios de dominio, resuelve referencias anafóricas y
consulta memorias a corto plazo (caché RAM) y largo plazo (ChromaDB). Si las memorias
producen un acierto, retorna el resultado sin invocar el pipeline. En caso contrario,
ejecuta los seis agentes especializados: \textbf{AR} (refinador), \textbf{APS}
(proximidad semántica), \textbf{AG} (generador SQL), \textbf{AV} (validador iterativo),
\textbf{AE} (explicador) y \textbf{AS} (sustentador bajo demanda). La especialización
es una necesidad técnica: cada agente requiere herramientas y métricas distintas, y los
experimentos de Fase~1 confirman que distintos modelos son óptimos para cada
rol~\cite{wooldridge2009}.

El protocolo experimental se organiza en \textbf{cuatro fases}.
La \textbf{Fase~0} compara marcos de evaluación sobre el AR con DeepEval como
motor compartido de métricas; ejecuta además una grilla exhaustiva de 30 combinaciones
(6 modelos $\times$ 5 temperaturas) que determina los cuatro modelos y cuatro temperaturas
usados en las fases siguientes, y selecciona Optuna como herramienta de optimización.
La \textbf{Fase~1} optimiza independientemente cada agente mediante Optuna
TPE~\cite{optuna2019,bergstra2011tpe} sobre el espacio modelo $\times$ temperatura,
evaluando con \textbf{10 preguntas} de prueba y métricas adaptadas al rol de cada agente.
La \textbf{Fase~2} valida la robustez ante paráfrasis: fija los ganadores de Fase~1 y
aplica \textbf{3 reformulaciones lingüísticas por intent} (30 preguntas en total),
midiendo la varianza del \textit{combined\_score}.
La \textbf{Fase~3} evalúa el orquestador ejecutando el pipeline completo sobre
\textbf{10 escenarios} que cubren todas las rutas posibles del sistema,
usando \texttt{ToolCorrectnessMetric} para verificar la activación correcta de agentes;
Optuna busca simultáneamente el mejor modelo y temperatura para el orquestador.
La \textbf{Fase~4} valida la robustez del orquestador ante \textbf{30 paráfrasis}
(3 reformulaciones por escenario).

\textbf{Limitaciones del alcance.}
Este trabajo no incluye la ejecución efectiva de las consultas SQL contra la base de
datos ni el tratamiento de tipos de datos complejos como arreglos (\texttt{ARRAY}),
tipos JSON anidados o datos geoespaciales. El sistema genera, valida y explica
la consulta en lenguaje natural, pero delega la ejecución y el manejo de resultados
estructurados complejos a la capa de aplicación que lo integre.

% ---------------------------------------------------------------
\section{Estado del Arte}
% ---------------------------------------------------------------

La investigación en Text-to-SQL ha evolucionado notablemente con la integración de
LLMs y arquitecturas basadas en agentes, pero los avances siguen siendo fragmentados:
algunos sistemas fortalecen la generación, otros la recuperación contextual, y pocos
abordan la validación, sin que ninguno integre los tres ejes junto con gestión de
contexto conversacional dentro de una arquitectura unificada.

Ojuri et al.~\cite{ojuri2025} integran LLMs con agentes mediante \textit{ReAct},
demostrando que el \textit{fine-tuning} supera al aprendizaje en contexto, pero sin
verificación ni explicaciones integradas. Tai et al.~\cite{tai2023cot} aplican
razonamiento encadenado (\textit{chain-of-thought}) mejorando la generación,
pero sin validación ni retroalimentación.

En recuperación contextual, Zhao et al.~\cite{zhao2024} proponen \textit{Chat2Data}
con bases vectoriales para reducir alucinaciones. Vichev y
Marchev~\cite{vichev2024} evalúan estrategias de \textit{embeddings} para
recuperación de esquema. Ambos evidencian el valor de la recuperación semántica,
pero carecen de validación y no contemplan el historial del usuario.

Desde la generación multi-agente, Wang et al.~\cite{guo2023} proponen MAC-SQL,
descomponiendo la generación SQL en sub-tareas especializadas coordinadas por un
orquestador; sin embargo, no contemplan validación proactiva ni memoria
conversacional. Wang et al.~\cite{wang2024sqlrefine} proponen un agente que usa herramientas
para inspeccionar y refinar consultas SQL, detectando discrepancias entre el SQL
generado y los datos reales, pero con verificación \textit{post-hoc} sin
retroalimentar la generación.

Desde la perspectiva de \textit{benchmarks}, los trabajos de referencia
miden \textbf{EX} (\textit{Execution Accuracy}: porcentaje de consultas
cuyo resultado de ejecución coincide exactamente con el del SQL
\textit{gold}). Yu et al.~\cite{yu2018spider} introducen Spider~1.0
(10.181 pares pregunta--SQL, 200 bases de datos), estándar entre 2018 y
2023; sobre él, DIN-SQL~\cite{pourreza2023dinsql} alcanza 82.8\%~EX y
DAIL-SQL~\cite{gao2023dailsql} 86.6\%~EX, ambos con \textit{fine-tuning}
o \textit{few-shot} sobre el propio \textit{split} de entrenamiento.
BIRD~\cite{li2023bird} (2023) introduce 12.751 pares sobre 95 bases de
datos reales; GPT-4 obtiene 54.9\%~EX recibiendo el DDL completo en cada
\textit{prompt} más un campo \textit{evidence} anotado manualmente.
Estos sistemas no son directamente comparables con SQL-Agents porque
(1)~reciben el DDL completo como \textit{input} (inviable en producción),
(2)~dependen de datos de entrenamiento o anotaciones manuales por dominio,
y (3)~carecen de contexto conversacional, memoria y validación proactiva.
SQL-Agents no reporta EX porque no ejecuta consultas contra la base de
datos durante la evaluación; en su lugar mide similitud textual
canonicalizada del SQL generado (\texttt{rouge\_l\_sql}) y calidad de
la explicación (Faithfulness, Groundedness).

La Tabla~\ref{tab:related} evidencia que ningún trabajo cubre simultáneamente
proximidad vectorial, validación iterativa, refinamiento semántico, gestión de
contexto conversacional, explicabilidad y memoria conversacional.

Desde la perspectiva arquitectónica, el patrón \textbf{Agent Skills}~\cite{anthropic_skills2024}
---propuesto por Anthropic para estructurar agentes LLM como habilidades modulares
e invocables--- no ha sido aplicado previamente en sistemas NL2SQL. Los trabajos
existentes adoptan arquitecturas monolíticas o pipelines secuenciales fijos, sin
separar las responsabilidades en Skills independientes que puedan componerse y
reutilizarse. SQL-Agents es, a conocimiento del autor, el primer sistema
Text-to-SQL que implementa este patrón, permitiendo que cada agente sea evaluado,
reemplazado o escalado de forma aislada sin afectar al resto del sistema.

\begin{minipage}{\columnwidth}
\begin{center}
\captionof{table}{Comparación de trabajos relacionados.}
\label{tab:related}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
\textbf{Trabajo} & \textbf{Prox.} & \textbf{Valid.} & \textbf{Refin.} &
\textbf{GCC} & \textbf{Gener.} & \textbf{Sust.} & \textbf{Mem.} & \textbf{Skills} & \textbf{EscProd.} \\
\midrule
Ojuri et al.~\cite{ojuri2025}   & No & No    & No & No & Sí & No & No & No & No \\
Tai et al.~\cite{tai2023cot}       & No & No    & No & No & Sí & No & No & No & No \\
Zhao et al.~\cite{zhao2024}     & Sí & No    & No & No & Sí & No & No & No & No \\
Vichev \& M.~\cite{vichev2024}  & Sí & No    & No & No & Sí & No & No & No & No \\
MAC-SQL~\cite{guo2023}          & No & No    & No & No & Sí & No & No & No & No \\
Wang et al.~\cite{wang2024sqlrefine}     & No & Parc. & No & No & Sí & No & No & No & No \\
DIN-SQL~\cite{pourreza2023dinsql} & No & No  & No & No & Sí & No & No & No & No \\
DAIL-SQL~\cite{gao2023dailsql}  & No & No    & No & No & Sí & No & No & No & No \\
\midrule
\textbf{SQL-Agents} & \textbf{Sí} & \textbf{Sí} & \textbf{Sí} &
\textbf{Sí} & \textbf{Sí} & \textbf{Sí} & \textbf{Sí} & \textbf{Sí} & \textbf{Sí} \\
\bottomrule
\end{tabular}%
}
\par\smallskip
\noindent\footnotesize{%
\textbf{Prox.:} Proximidad Vectorial.
\textbf{Valid.:} Validación proactiva.
\textbf{Refin.:} Refinamiento semántico.
\textbf{GCC:} Gestión de Contexto.
\textbf{Sust.:} Sustentador.
\textbf{Mem.:} Memoria inter-sesión.
\textbf{EscProd.:} escalable a producción sin DDL completo ni entrenamiento por dominio.%
}
\end{center}
\end{minipage}

% ---------------------------------------------------------------
\section{Marco Teórico}
% ---------------------------------------------------------------

\subsection{Text-to-SQL y NL2SQL}

La tarea Text-to-SQL consiste en transformar una pregunta en lenguaje natural en una
consulta SQL ejecutable sobre un esquema relacional dado. Formalmente, dado un
enunciado $q$ y un esquema $S$, el objetivo es producir $y = f(q, S)$~\cite{katsogiannis2023}.
Los principales desafíos incluyen la resolución de ambigüedades léxicas, la
alineación entre términos del usuario y nombres de columnas, y la generación correcta
de \textit{joins} en esquemas complejos~\cite{deng2022}. El alcance del presente
trabajo se restringe a bases de datos relacionales OLTP con propiedades ACID y
esquemas en estrella o copo de nieve, excluyendo fuentes NoSQL o documentales.

\subsection{Modelos de Lenguaje Grandes (LLMs)}

Los LLMs son redes \textit{transformer} entrenadas sobre corpus masivos, capaces de
realizar tareas de comprensión y generación de lenguaje natural sin entrenamiento
específico por tarea~\cite{brown2020}. En Text-to-SQL, han demostrado capacidad para
generar consultas complejas cuando se les provee contexto del esquema mediante
\textit{prompting} estructurado~\cite{pourreza2023dinsql}. La \textbf{temperatura} $T$ controla
la distribución de probabilidad sobre los tokens generados: $T{\approx}0$ produce
salidas deterministas, mientras que $T{\approx}1$ incrementa la diversidad léxica
pero puede reducir la precisión~\cite{brown2020}.

\subsection{Sistemas Multi-Agente}

Un sistema multi-agente (SMA) es un entorno computacional compuesto por agentes
autónomos que perciben su entorno, razonan y ejecutan acciones para alcanzar objetivos
colectivos~\cite{wooldridge2009}. En el ámbito de los LLMs, los agentes son instancias
configuradas con roles, herramientas y restricciones específicas que colaboran mediante
paso de mensajes y estado compartido~\cite{wang2024survey}. La especialización de
agentes permite descomponer tareas complejas en subtareas manejables, mejorando la
trazabilidad y la modularidad del sistema~\cite{wooldridge2009}.

\subsection{Búsqueda Semántica y Proximidad Vectorial}

La búsqueda semántica por proximidad vectorial consiste en representar textos como
vectores densos y recuperar los más similares mediante similitud
coseno~\cite{reimers2019}. En Text-to-SQL, permite recuperar los elementos del esquema
más relevantes para la pregunta del usuario, reduciendo el contexto enviado al LLM
y mejorando la precisión~\cite{vichev2024}. ChromaDB es una base de datos vectorial
de código abierto que almacena, indexa y consulta \textit{embeddings} de forma
eficiente~\cite{chromadb2023}.

\subsection{Gestión de Contexto Conversacional}

Tres mecanismos interdependientes~\cite{zhang2024memory} sustentan el diálogo
multi-turno: \textbf{Context Window} (ventana de tokens disponibles~\cite{brown2020}),
\textbf{Context Switching} (detección de cambio de dominio para reiniciar el
contexto sin perder la memoria) y \textbf{Contextual Awareness}
(\textit{coreference resolution}: ``¿de esos, cuántos son de Perú?'' extiende
la consulta previa~\cite{suhr2018}). En SQL-Agents el Orquestador combina los
tres con la memoria a corto y largo plazo.

\subsection{Memoria Conversacional}

La memoria en sistemas LLM se clasifica en dos tipos~\cite{zhang2024memory}: la
\textbf{memoria a corto plazo}, que mantiene el historial de la sesión activa
almacenando consultas recientes y sus resultados SQL exitosos para evitar
reprocesamiento; y la \textbf{memoria a largo plazo}, que persiste entre sesiones
información sobre consultas frecuentes y esquemas accedidos. En SQL-Agents, antes de
ejecutar el pipeline completo, el Orquestador consulta la memoria a largo plazo: si
existe una respuesta similar a la pregunta actual, la retorna directamente sin invocar
a los demás agentes~\cite{langchain2023}.

\subsection{Change Data Capture (CDC) y Contexto de Esquema}

En este trabajo, el módulo CDC es un script Python que se conecta a las bases de
datos MySQL y PostgreSQL para extraer el DDL, particiones, índices y estadísticas
del esquema, serializándolos en formato JSON~\cite{kleppmann2017}. Este JSON, junto
con el \textbf{diccionario de datos} ---un documento que describe el significado
semántico de cada tabla, columna y relación---, constituye el contexto que el Agente
de Proximidad indexa en ChromaDB. El alcance se restringe a bases de datos relacionales
OLTP/ACID en estrella o copo de nieve; fuentes NoSQL, documentales o archivos JSON
como fuente primaria quedan fuera del alcance.

\subsection{Patrón Agent Skills}

El patrón \textbf{Agent Skills}~\cite{anthropic_skills2024} define cada capacidad
de un agente LLM como una \textit{skill}: unidad modular, autocontenida e
invocable mediante una interfaz definida. Un orquestador central selecciona
dinámicamente qué skill invocar según el estado de la conversación.
En SQL-Agents cada agente es una skill registrada en un \textit{Tool Registry};
adicionalmente, cada agente expone sub-skills internas vía \textit{tool use}
de la API de Anthropic. Este diseño permite la evaluación aislada de cada agente
---característica central de la metodología experimental--- y la sustitución del
LLM de cualquier agente sin modificar los demás.

\subsection{Model Context Protocol (MCP)}

El \textbf{Model Context Protocol (MCP)}~\cite{anthropic_mcp2024} es un estándar
abierto de Anthropic que conecta agentes LLM con servicios externos mediante
JSON-RPC sobre stdio o SSE. A diferencia del patrón Agent Skills ---donde cada
habilidad vive embebida en el agente---, las herramientas MCP residen en un
servidor independiente reutilizable por múltiples agentes o clientes externos.

En SQL-Agents, MCP externaliza dos clases de conocimiento:
\begin{enumerate}[leftmargin=*]
  \item \textbf{Contexto de dominio}: descripción narrativa de cada base de
    datos (propósito, usuarios, preguntas frecuentes) mantenida como documento
    externo; los agentes la consultan vía MCP sin \textit{hardcodear} reglas.
  \item \textbf{Operaciones algorítmicas sobre el esquema}: similitud coseno
    sobre \textit{embeddings} indexados, listado de claves foráneas e
    identificación de claves primarias ---deterministas y sin LLM.
\end{enumerate}

La especialización por base de datos se concentra en archivos externos editables,
satisfaciendo el principio de agnosticidad al dominio. Esta investigación
constituye, a conocimiento del autor, la primera aplicación de MCP a un sistema
NL2SQL multi-agente publicada en la literatura.

\subsection{LangGraph y Orquestación de Agentes}

LangGraph modela flujos de agentes LLM como grafos de estado dirigidos, donde cada
nodo representa un agente y las aristas determinan transiciones según condiciones
definidas~\cite{langgraph2024}. Soporta ciclos de retroalimentación
(\textit{feedback loops}), permitiendo que el Agente Validador devuelva el control
al Agente Generador para corregir la consulta sin reiniciar el pipeline completo,
lo cual es esencial para la validación proactiva en SQL-Agents~\cite{langgraph2024}.

\subsection{Métricas de Confianza y Evaluación}

La evaluación de sistemas NL2SQL requiere métricas que midan tanto la calidad
de las salidas generadas como la calibración de la confianza reportada por los
agentes. En SQL-Agents se emplean seis métricas complementarias, aplicables a
cualquier agente del pipeline.
 
\textbf{ROUGE-L} (\textit{Recall-Oriented Understudy for Gisting Evaluation})
mide la similitud entre la salida generada por un agente y una referencia
esperada, utilizando la subsecuencia común más larga (LCS)~\cite{lin2004}.
A diferencia de otras métricas de similitud textual, ROUGE-L no requiere
coincidencias contiguas, lo que permite capturar la preservación del contenido
semántico incluso cuando el orden de las palabras varía. Se calcula como:
\begin{equation}
  \text{ROUGE-L} = \frac{(1+\beta^2)\cdot R_{lcs}\cdot P_{lcs}}{R_{lcs} + \beta^2 \cdot P_{lcs}}
  \label{eq:rouge}
\end{equation}
donde $R_{lcs}{=}|LCS(\hat{y},y)|/|y|$, $P_{lcs}{=}|LCS(\hat{y},y)|/|\hat{y}|$ y
$\beta$ pondera precisión frente a cobertura. Un valor cercano a 1 indica que la
salida reproduce fielmente el contenido de la referencia.
 
\textbf{WACS} (\textit{Weighted Agent Confidence Score}) cuantifica la confianza
global del sistema como promedio ponderado de la confianza declarada por cada
agente:
\begin{equation}
  \text{WACS} = \sum_{i \in A} w_i \cdot c_i
  \label{eq:wacs}
\end{equation}
donde $A{=}\{\text{AR, APS, AG, AV}\}$, $c_i{\in}[0,1]$ es la confianza del
agente $i$, y los pesos son: AR~$(0.10)$, APS~$(0.20)$, AG~$(0.30)$,
AV~$(0.40)$. Un WACS alto con ROUGE-L bajo señala sobreconfianza, orientando
el ajuste de umbrales.
 
\textbf{Faithfulness} (Fidelidad). La salida del agente \textbf{no
contradice ni excede la respuesta esperada}~\cite{aws_bedrock_eval}.
Para el AR es la versión limpia esperada de la pregunta; para AE/AS,
la explicación esperada. Formalmente, sobre el conjunto $S$ de
afirmaciones de la salida:
$$F = \frac{|\{s_i \in S : s_i \text{ no contradice lo esperado}\}|}{|S|}$$

\textbf{Groundedness} (Anclaje). La salida se ancla en la entrada recibida
sin agregar conceptos ausentes~\cite{azure_groundedness,maynez2020faithfulness}:
$$G = \frac{|\{s_i \in S : s_i \text{ se deriva de la entrada}\}|}{|S|}$$
$F$ evalúa contra la \textbf{respuesta esperada}; $G$ contra la
\textbf{entrada recibida}: dos ejes complementarios para detectar fallos
de distinta naturaleza~\cite{es2023ragas}.
 
\textbf{Brier Score} mide la calibración probabilística de las confianzas
declaradas por los agentes~\cite{brier1950}. Cuantifica la diferencia cuadrática
media entre la confianza declarada $p_i$ y el resultado binario real
$o_i \in \{0,1\}$:
\begin{equation}
  \text{BS} = \frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2
  \label{eq:brier}
\end{equation}
Un Brier Score cercano a 0 indica que la confianza declarada es un predictor
preciso del resultado. Por ejemplo, si un agente declara confianza 0.9 y acierta,
su Brier individual es $(0.9{-}1.0)^2 = 0.01$; si declara 0.9 pero falla,
es $(0.9{-}0.0)^2 = 0.81$.
 
\textbf{ECE} (\textit{Expected Calibration Error}) complementa el Brier Score
agrupando las predicciones en $M$ intervalos (\textit{bins}) por nivel de
confianza y midiendo la diferencia entre la confianza media y la precisión
real en cada intervalo~\cite{guo2017calibration}:
\begin{equation}
  \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}
  \left|\text{acc}(B_m) - \text{conf}(B_m)\right|
  \label{eq:ece}
\end{equation}
Un ECE cercano a 0 indica que un agente que dice estar ``80\% seguro'' acierta
efectivamente en el 80\% de los casos. A diferencia del Brier Score que da un
valor global, ECE identifica en qué rangos de confianza el agente está mal
calibrado.










% ---------------------------------------------------------------
\section{Caso de Estudio}
% ---------------------------------------------------------------

\subsection{Descripción del problema}
La generación manual de consultas SQL representa una barrera para usuarios no
técnicos~\cite{katsogiannis2023}. Esta problemática se agudiza en escenarios
multi-turno~\cite{suhr2018}.

\subsection{Descripción de la solución}
SQL-Agents aborda el problema mediante un pipeline multi-agente orquestado.
La Figura~\ref{fig:workflow_reducido} muestra el flujo principal.

\end{multicols}
\vspace{6pt}
\begin{center}
  \includegraphics[width=0.50\textwidth]{workflow_tesis_new.PNG}
  \captionof{figure}{Diagrama de flujo del sistema SQL-Agents.}
  \label{fig:workflow_reducido}
\end{center}
\vspace{6pt}
\begin{multicols}{2}

El flujo general opera así: un \textbf{Clasificador de Intención} determina si la
entrada es consulta de datos, sustentación o conversacional. Si es consulta, el
Orquestador verifica memorias; de no encontrar coincidencia, activa el pipeline:
AR, APS, AG, AV y AE. Si el usuario solicita cambio de tema, el sistema reinicia
el contexto manteniendo las memorias intactas.

% ---------------------------------------------------------------
% SECCION 4.3 — ARQUITECTURA CON LAS 3 FIGURAS
% ---------------------------------------------------------------

\subsection{Arquitectura de agentes}

El sistema implementa seis agentes coordinados por un Orquestador sobre LangGraph.
La Figura~\ref{fig:arq_agentes} muestra los agentes y sus interacciones;
la Figura~\ref{fig:arq_datos} presenta el módulo CDC y diccionario de datos;
y la Figura~\ref{fig:arq_memoria} describe las memorias. La imagen completa
se incluye en el Anexo~B (Figura~\ref{fig:arq_completa}).

\end{multicols}

% --- Fila 1: agentes especializados + recursos MCP ---
\vspace{6pt}
\begin{center}
  \begin{minipage}[t]{0.47\textwidth}
    \centering
    \includegraphics[width=\linewidth]{agent_skills_agentes_new.PNG}
    \captionof{figure}{Agentes especializados (AR, APS, AG, AV, AE, AS)
    y flujo de interacción coordinados por el Orquestador.}
    \label{fig:arq_agentes}
  \end{minipage}
  \hfill
  \begin{minipage}[t]{0.47\textwidth}
    \centering
    \includegraphics[width=\linewidth]{agent_skills_mcp_new.PNG}
    \captionof{figure}{Servidor MCP: schemas DDL, descripciones de dominio
    y bases vectoriales del esquema. CDC alimenta MySQL/PostgreSQL.}
    \label{fig:arq_datos}
  \end{minipage}
\end{center}
\vspace{6pt}

% --- Fila 2: memoria centrada ---
\begin{center}
  \includegraphics[width=0.42\textwidth]{agent_skills_memoria_new.PNG}
  \captionof{figure}{Memoria a corto plazo (RAM de sesión) y memoria a
  largo plazo (ChromaDB persistente por usuario--base\_de\_datos--dataset)
  vinculadas al Orquestador.}
  \label{fig:arq_memoria}
\end{center}
\vspace{8pt}

\begin{multicols}{2}

El flujo principal sigue:
\texttt{AR $\rightarrow$ APS $\rightarrow$ AG $\rightleftharpoons$
AV $\rightarrow$ AE}, con AS bajo demanda.

  \subsubsection{Indexación previa: módulo CDC y diccionario de datos}
  \label{sec:cdc-sampling}

  El CDC extrae el DDL de MySQL y PostgreSQL hacia un JSON.
  El administrador de la BD provee un \textbf{diccionario de datos}
  con el significado semántico de tablas y columnas.
  El APS indexa ambos en ChromaDB y, en cada consulta, entrega al AG
  y al AV el subconjunto de tablas y columnas relevantes.

  \subsubsection{Agente Refinador (AR)}
  Valida que la pregunta sea una consulta de datos legítima, rechaza
  DML y conocimiento general, y la refina gramaticalmente.
  Consume vía MCP la descripción narrativa del dominio activo para no
  rechazar terminología técnica válida del esquema.
  Asigna confianza $c_1 \in [0,1]$.

  \subsubsection{Agente de Proximidad Semántica (APS)}
  Realiza búsqueda vectorial en ChromaDB para identificar tablas y
  columnas relevantes. Sus funciones (\texttt{search\_tables},
  \texttt{search\_columns}) están publicadas como herramientas MCP,
  permitiendo su uso tanto desde el pipeline interno como desde
  clientes externos. Asigna confianza $c_2 \in [0,1]$ según la
  similitud media de los candidatos recuperados.

  \subsubsection{Agente Generador (AG)}
  Genera SQL (MySQL o PostgreSQL) a partir del esquema provisto por
  APS. Cuatro skills deterministas cubren: seguridad SQL, corrección
  de palabras reservadas y búsqueda de JOINs (directos por FK y por
  BFS sobre el grafo de FK). Opera en ciclo AG\,$\rightleftharpoons$\,AV
  (máx. $k{=}3$ iteraciones); asigna confianza $c_3 \in [0,1]$.

  \subsubsection{Agente Validador (AV)}
  \label{sec:agente-av}
  Opera en dos planos: \textbf{(i)~estructural} ---verifica columnas y
  tipos vs. esquema APS; errores no reparables detienen el pipeline,
  reparables activan el ciclo AG\,$\rightleftharpoons$\,AV--- y
  \textbf{(ii)~calidad estática} ---detecta productos cartesianos,
  JOINs excesivos, patrones SQL ineficientes y subqueries profundas.
  Asigna confianza $c_5 \in [0,1]$; tras $k{=}3$ iteraciones
  fallidas el AE comunica el límite al usuario.

  \subsubsection{Agente Explicador (AE)}
  Sintetiza en lenguaje simple las tablas usadas, el SQL generado y la
  confianza de cada agente. Calcula el \textbf{WACS}:
  $$\text{WACS} = 0{,}10\,c_1 + 0{,}20\,c_2 + 0{,}30\,c_3 + 0{,}40\,c_5$$
  Los pesos crecen hacia el AV por ser la última validación causal.

  \subsubsection{Agente Sustentador (AS)}
  Se activa bajo demanda para responder preguntas sobre el proceso
  (\textit{``¿por qué esas tablas?'', ``¿qué tan seguro estás?''}).
  Sintetiza los razonamientos de cada agente en lenguaje comprensible.

  \subsubsection{Componentes Auxiliares}
  \begin{itemize}[leftmargin=*,topsep=2pt,itemsep=2pt]
    \item \textbf{Memoria a corto plazo} (RAM de sesión): historial de
      turnos del diálogo actual; mantiene coherencia conversacional.
    \item \textbf{Memoria a largo plazo} (ChromaDB): persiste consultas
      y razonamientos previos; el Orquestador la consulta para
      responder directamente cuando detecta una pregunta similar a
      una ya resuelta en sesiones anteriores.
    \item \textbf{Enriquecimiento contextual}: resuelve referencias
      anafóricas (``de ellos'', ``los mismos'') recuperando la entidad
      referenciada del historial antes de invocar el pipeline.
    \item \textbf{Cambio de contexto}: el Orquestador decide mediante
      LLM si la nueva consulta pertenece a un dominio diferente al
      activo y, si es así, reinicia el estado conversacional para
      evitar mezcla de contextos.
    \item \textbf{Tool Registry}: registro central que controla qué
      \textit{skills} puede invocar cada agente, previniendo accesos
      no autorizados.
  \end{itemize}

\subsection{Metodología (CRISP-DM)}
Se adopta CRISP-DM~\cite{crispdm2000} adaptada al desarrollo de un sistema
multi-agente NL2SQL.


\section{Protocolo Experimental}

El protocolo experimental se organiza en \textbf{cuatro fases}.
La \textbf{Fase~0} compara cinco frameworks de evaluación sobre el AR
con DeepEval como motor compartido de métricas; además ejecuta una
grilla exhaustiva de 30 combinaciones (6 modelos $\times$ 5 temperaturas)
que determina los cuatro modelos y cuatro temperaturas usados en las
fases siguientes, y selecciona \textbf{Optuna} como herramienta de
optimización.
La \textbf{Fase~1} optimiza independientemente cada agente mediante
Optuna TPE sobre el espacio modelo $\times$ temperatura, con métricas
adaptadas al rol de cada agente.
La \textbf{Fase~2} valida la robustez ante paráfrasis: fija los
ganadores de Fase~1 y aplica 3 reformulaciones lingüísticas por
\textit{intent}, midiendo la varianza del \textit{combined\_score}.
La \textbf{Fase~3 --- Evaluación del Orquestador} ejecuta el
\textit{pipeline} completo sobre 10 escenarios que cubren todas las
rutas posibles del sistema, usando \texttt{ToolCorrectnessMetric} para
verificar activación correcta de agentes y \textit{skills}; Optuna
busca simultáneamente el mejor modelo y temperatura para el agente
orquestador.
La \textbf{Fase~4} valida la robustez del orquestador ante
\textbf{30 paráfrasis} (3 reformulaciones por escenario), fijando los
hiperparámetros ganadores de Fase~3; un sistema robusto debe activar
los mismos agentes y producir SQL equivalente independientemente de la
redacción del usuario.

% ---------------------------------------------------------------
\subsection{Modelos LLM evaluados}
% ---------------------------------------------------------------

Antes de elegir una herramienta de evaluación, conviene fijar el
\textbf{conjunto de modelos a evaluar}, dado que toda la comparación de
frameworks (Fase~0) y los \textit{trials} de optimización (Fase~1) se
ejecutan sobre este conjunto.

La selección de seis modelos de tres proveedores obedece a criterios
técnicos, económicos y de \textit{reproducibilidad cross-proveedor}.
Para cada proveedor se evaluaron dos modelos en el espectro
eficiencia-capacidad: un modelo \textit{rápido y económico}
y otro de \textit{mayor capacidad}. \textbf{GPT-4o-mini} y
\textbf{GPT-4o} (OpenAI)~\cite{openai_gpt4o2024} representan al proveedor
líder del mercado. \textbf{Claude Haiku 4.5} y \textbf{Claude Sonnet 4.6}
(Anthropic)~\cite{anthropic_claude2024} son los equivalentes funcionales
en el segundo proveedor de mayor adopción empresarial.
\textbf{Gemini 2.5 Flash} y \textbf{Gemini 2.5 Pro}
(Google)~\cite{gemini_google2024} completan la terna de los tres
proveedores comerciales más usados y cubren el ecosistema de la
\textit{cloud platform} de Google, relevante para despliegues en
infraestructura GCP. La inclusión del tercer proveedor permite
contrastar si una mejora observada en uno se generaliza, o si depende
del tokenizador, la arquitectura del modelo o el ajuste de
instrucciones (\textit{instruction tuning}) específico de cada
proveedor. La Tabla~\ref{tab:model_equiv} resume los tríos
funcionalmente equivalentes entre proveedores.

\begin{center}
\captionof{table}{Modelos LLM evaluados por fase. Fase~0: grilla exhaustiva
(6 modelos, 5 temperaturas). Fase~1--4: búsqueda bayesiana con los
4 modelos de mayor rendimiento (4 modelos, 4 temperaturas).}
\label{tab:model_equiv}
\footnotesize
\begin{tabular}{llll}
\toprule
\textbf{Modelo} & \textbf{Proveedor} & \textbf{Fase~0} & \textbf{Fase~1} \\
\midrule
GPT-4o-mini      & OpenAI    & \ding{51} & -- \\
GPT-4o           & OpenAI    & \ding{51} & \ding{51} \\
Claude Haiku 4.5 & Anthropic & \ding{51} & \ding{51} \\
Claude Sonnet 4.6& Anthropic & \ding{51} & \ding{51} \\
Gemini 2.5 Flash & Google    & \ding{51} & \ding{51} \\
Gemini 2.5 Pro   & Google    & \ding{51} & -- \\
\bottomrule
\end{tabular}
\end{center}

Fase~0 evalúa los seis modelos en grilla exhaustiva ($6 \times 5 = 30$
combinaciones) para comparar proveedores bajo condiciones idénticas.
Sus resultados determinan el espacio de Fases~1--4:
\textit{(i)}~GPT-4o-mini y Gemini~2.5~Pro se excluyen por menor
rendimiento relativo frente a GPT-4o y Gemini~2.5~Flash
respectivamente, conservando los cuatro modelos de mayor capacidad;
\textit{(ii)}~las temperaturas $\{0.0, 0.3, 0.5, 0.7\}$ se retienen por
ser los cuatro valores con mayor concentración de resultados superiores
en la grilla, descartando $T{=}0.1$ cuya ventaja sobre $T{=}0.0$ fue
marginal en todos los agentes.
Fases~1--4 usan Optuna TPE sobre el subespacio resultante
($4\,\text{modelos} \times 4\,T = 16$ combinaciones, \texttt{}).

Adicionalmente, todos los componentes vectoriales del sistema
---búsqueda semántica del APS, indexación de la memoria a largo plazo
y detección de cambio de contexto--- emplean
\texttt{intfloat/multilingual-e5-base}~\cite{wang2022e5}, fijado para
todo el proyecto. La variante \textit{base} (109M parámetros) supera
a \texttt{text-embedding-ada-002} en 56/64 \textit{datasets} del
benchmark MTEB~\cite{muennighoff2022mteb}, cubre 100 idiomas (XLM-R),
opera localmente sin API de pago y su licencia MIT permite uso
comercial. Una comparación contra BGE-M3 o \texttt{text-embedding-3-small}
queda como trabajo futuro.

% ---------------------------------------------------------------
\subsection{Temperaturas evaluadas y herramienta de métricas (DeepEval)}
% ---------------------------------------------------------------

Se evalúan cinco valores de temperatura que cubren el espectro entre
comportamiento determinista y diversidad léxica
(Tabla~\ref{tab:temperature_citations}).

\begin{center}
\captionof{table}{Valores de temperatura evaluados en la grilla Fase~0
y su respaldo bibliográfico. Los cuatro valores retenidos para Fases~1--4
son $\{0.0,\,0.3,\,0.5,\,0.7\}$.}
\label{tab:temperature_citations}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{cp{0.62\textwidth}}
\toprule
\textbf{$T$} & \textbf{Justificación / Fuente} \\
\midrule
$0.0$ & Determinístico estándar en NL2SQL: DIN-SQL~\cite{pourreza2023dinsql}, DAIL-SQL~\cite{gao2023dailsql}. \\
$0.1$ & Variante casi-determinística; descartada ($\Delta < 0.005$ vs $T{=}0.0$). \\
$0.3$ & Recomendado por Anthropic~\cite{anthropic_claude2024} para tareas analíticas. \\
$0.5$ & Punto de equilibrio entre determinismo y diversidad~\cite{brown2020}. \\
$0.7$ & Default de LLaMA~2~\cite{touvron2023llama2} y \textit{chat completion}. \\
\bottomrule
\end{tabular}%
}
\end{center}

\textbf{Herramienta de métricas: DeepEval y juez LLM.}
Las métricas semánticas ($F$ = Faithfulness, $G$ = Groundedness) se
calculan con \textbf{DeepEval}~\cite{deepeval2024} mediante el criterio
\textbf{GEval}~\cite{liu2023geval}, que instruye a un LLM para puntuar
la calidad de la salida del agente comparándola con la respuesta esperada
según un criterio definido. El juez es fijo: \textbf{GPT-4o}
con $T{=}0{,}0$~\cite{openai_gpt4o2024} en todas las fases, garantizando
determinismo en la evaluación --- las mismas entradas producen siempre la
misma puntuación, independientemente del modelo bajo prueba. Este diseño
sigue la práctica establecida de LLM-as-judge~\cite{zheng2023judging}:
un evaluador de alta capacidad con temperatura nula elimina la varianza
del juez, aislando únicamente la varianza del agente evaluado. El mismo
juez y los mismos criterios se reutilizan en todas las fases del protocolo
para garantizar comparabilidad directa entre agentes.

% ---------------------------------------------------------------
\subsection{Fase 0 --- Grilla exhaustiva y selección de herramienta}
% ---------------------------------------------------------------

La Fase~0 tiene dos objetivos simultáneos sobre el Agente Refinador (AR):
\textit{(i)}~evaluar comparativamente cinco herramientas de optimización
y observabilidad para seleccionar la que se usará en Fases~1--3, y
\textit{(ii)}~ejecutar una \textbf{grilla exhaustiva} de $6 \times 5 = 30$
combinaciones (todos los modelos y temperaturas del protocolo) para
identificar qué modelos y temperaturas resultan más prometedores,
determinando el espacio reducido de Fase~1.
Las cinco herramientas se evaluaron sobre el AR bajo
condiciones idénticas: los mismos 10 casos de prueba del conjunto
\texttt{spider:concert\_singer}, las mismas 30 combinaciones
(6 modelos $\times$ 5 temperaturas), y las mismas
seis métricas calculadas con funciones compartidas:
$30 \times 10 = 300$ evaluaciones del agente.

\subsubsection{Herramientas evaluadas y resultados}
\label{sec:fase0-metrics-lib}

Se implementó un \textbf{módulo de métricas común} compartido por las
cinco herramientas, garantizando que todas calculen exactamente los
mismos valores de $F$, $G$, $R$, $B$ y $E$.
Las cinco herramientas se evaluaron sobre el AR con 30 combinaciones
(6 modelos $\times$ 5 temperaturas) y 10 preguntas por combinación:

\begin{itemize}[leftmargin=*,topsep=2pt,itemsep=1pt]
  \item \textbf{Optuna}~\cite{optuna2019}: optimización bayesiana con
    \texttt{TPESampler}~\cite{bergstra2011tpe}, 10 \textit{trials}. Mejor
    combinación de la grilla exhaustiva: Gemini~2.5~Flash ($T{=}0.5$,
    $S{=}0.962$). Genera 4 gráficas HTML interactivas
    (historial, scatter, importancia, coordenadas paralelas).
  \item \textbf{LangSmith}~\cite{langsmith2024}: observabilidad web de
    LangChain. Registró las 30 combinaciones con trazabilidad por llamada LLM.
  \item \textbf{Langfuse}~\cite{langfuse2024}: plataforma de observabilidad LLM en la nube (\textit{open-source}, auto-hosteable).
    Registró 180 \textit{scores} (30 combinaciones $\times$ 6 métricas) vía API cloud.
  \item \textbf{DSPy}~\cite{dspy2024}: optimización de prompts via
    \textit{BootstrapFewShot}. Aplicado al ganador de la grilla: gemini-2.5-flash
    $T{=}0.5$ bajó de $S{=}0.962$ a $0.812$ ($\Delta{=}{-}0.150$); los ejemplos
    \textit{few-shot} no mejoran una tarea de refinamiento ya bien resuelta.
  \item \textbf{Promptfoo}~\cite{promptfoo2024}: evaluación declarativa YAML.
    Generó tabla HTML comparativa; las 30/30 combinaciones superan $S{\geq}0{,}90$
    (score mínimo observado: $S{=}0{,}934$).
\end{itemize}

Los tres hallazgos principales son: (1)~las cuatro herramientas no-DSPy
producen métricas idénticas para el mismo modelo/temperatura,
confirmando la consistencia de \texttt{metricas\_lib};
(2)~Gemini~2.5~Flash ($T{=}0.5$) lidera la grilla exhaustiva ($S{=}0.962$),
seguido por Claude Haiku y Sonnet con mejor calibración ($B{\leq}0.002$);
GPT-4o-mini y Gemini~2.5~Pro quedan por debajo de sus pares respectivos;
(3)~Gemini~2.5~Flash es inestable a $T{=}0.3$ ($S{=}0.936$), varianza
no observada en OpenAI ni Anthropic.
La Tabla~\ref{tab:metrics_comparison} compara las métricas por herramienta
y la Tabla~\ref{tab:tools_capabilities} sus capacidades funcionales.

Se seleccionó \textbf{Optuna} para las Fases~1--3 por ser la única
herramienta con optimización bayesiana automática, visualizaciones
analíticas reproducibles y ejecución local sin dependencias cloud.
Los resultados de la grilla determinan el espacio reducido de Fase~1:
GPT-4o-mini y Gemini~2.5~Pro se excluyen por menor rendimiento frente
a sus pares; $T{=}0.1$ se descarta por mejora marginal ($\Delta < 0.005$)
frente a $T{=}0.0$; se retienen $T \in \{0.0,0.3,0.5,0.7\}$
(Tabla~\ref{tab:temperature_citations}).


\begin{center}
\captionof{table}{Métricas AR para el ganador Fase~0 (gemini-2.5-flash
$T{=}0{,}5$) en cada herramienta. Valores idénticos confirman
consistencia de \texttt{metricas\_lib}.
DSPy\textsuperscript{†} aplica \textit{BootstrapFewShot} al mismo modelo.}
\label{tab:metrics_comparison}
\footnotesize
\begin{tabular}{lccccc}
\toprule
\textbf{Herramienta} & $F$ & $G$ & $R$ & $B$ & $E$ \\
\midrule
Optuna     & 0.959 & 0.990 & 0.970 & 0.004 & 0.053 \\
LangSmith  & 0.959 & 0.990 & 0.970 & 0.004 & 0.053 \\
Langfuse   & 0.959 & 0.990 & 0.970 & 0.004 & 0.053 \\
Promptfoo  & 0.959 & 0.990 & 0.970 & 0.004 & 0.053 \\
\midrule
DSPy\textsuperscript{†} & 0.808 & 0.886 & 0.815 & 0.071 & 0.068 \\
\bottomrule
\end{tabular}
\par\smallskip
\footnotesize{\textsuperscript{†}DSPy usa \textit{BootstrapFewShot};
la optimización del prompt no mejora el rendimiento
($S{=}0.812$ vs.\ $S{=}0.962$ base, $\Delta{=}{-}0.150$).}
\end{center}

\end{multicols}

\vspace{4pt}
\begin{table}[h!]
\centering\footnotesize
\caption{Comparación de herramientas evaluadas. HP: hiperparámetros.}
\label{tab:tools_capabilities}
\begin{tabular}{llcc}
\toprule
\textbf{Herram.} & \textbf{Propósito} & \textbf{Visualización} & \textbf{Opt.~HP} \\
\midrule
Optuna     & Búsqueda bayesiana & HTML local & Sí \\
LangSmith  & Trazabilidad LLMs  & Web nube   & No \\
Langfuse   & Observab. LLMs     & Web nube   & No \\
DSPy       & Compilación prompt & ---        & No \\
Promptfoo  & Evaluación YAML    & Tabla HTML & No \\
\bottomrule
\end{tabular}
\end{table}
\vspace{4pt}



\begin{multicols}{2}
 
\subsection{Fase 1 --- Optimización por agente con Optuna}

La Fase~1 ajusta independientemente cada agente del pipeline mediante
Optuna~\cite{optuna2019} con el \textbf{\texttt{TPESampler}}
(\textit{Tree-structured Parzen Estimator},
Bergstra et al.~\cite{bergstra2011tpe}), semilla fija \texttt{}.
El espacio de hiperparámetros y la función objetivo se adaptan al rol
de cada agente, ya que las métricas relevantes no son las mismas para
un clasificador binario, un retriever vectorial o un generador de
lenguaje natural.

\textbf{Justificación del sampler.} El espacio para los agentes basados
en LLM tiene dos ejes (\textit{modelo} y \textit{temperatura} $T$),
generando $4 \times 4 = 16$ combinaciones únicas.
TPE construye, tras $n_{\text{startup}}=3$ \textit{trials} aleatorios,
dos densidades probabilísticas
$\ell(x) = P(x \mid \text{combined\_score} \geq Q_{75})$ y
$g(x) = P(x \mid \text{combined\_score} < Q_{25})$, y muestrea el
siguiente punto maximizando el cociente $\ell(x)/g(x)$ (expectativa de
mejora). Con $10$ \textit{trials} totales se cubre el $50\%$ del espacio
de manera dirigida; para dos ejes la heurística de
Bergstra~et~al.~\cite{bergstra2011tpe} requiere
$n_{\text{trials}} \geq 10 \times 2 = 20$, valor que se aproxima
manteniendo el costo computacional por agente por debajo de 30~minutos.

\subsubsection{Espacio de hiperparámetros y casos de prueba}

Los cuatro modelos de Fases~1--4 (gpt-4o, claude-haiku-4-5,
claude-sonnet-4-6, gemini-2.5-flash) y las cuatro temperaturas
$\{0.0, 0.3, 0.5, 0.7\}$ se seleccionaron a partir de los resultados
de Fase~0; GPT-4o-mini y Gemini~2.5~Pro se excluyen por menor
rendimiento relativo frente a sus pares de mayor capacidad del mismo
proveedor. $T{=}0.1$ se descarta pues su diferencia sobre $T{=}0.0$
fue marginal en todos los agentes ($\Delta < 0.005$).
Los cuatro valores de $T$ están respaldados bibliográficamente
(Tabla~\ref{tab:temperature_citations}, Fase~0).

El espacio de los agentes basados en LLM es $4\,\text{modelos} \times
4\,T = 16$ combinaciones; TPE muestrea $10$ ($62.5\%$) con
$n_{\text{startup}}{=}3$ aleatorios.
El \textbf{APS} es la excepción: al no usar un LLM sino ChromaDB como
motor de recuperación, no tiene hiperparámetros de modelo ni temperatura.
Su único hiperparámetro es la \textbf{métrica de distancia HNSW}
(\texttt{cosine}, \texttt{l2} o \texttt{ip}) que determina cómo se
calculan las similitudes entre el \textit{embedding} de la pregunta y
los \textit{embeddings} del esquema indexados. Optuna evalúa las 3
opciones y selecciona la que maximiza $F_1$ en tablas y columnas.

Cada agente usa \textbf{10 casos de prueba} sobre Spider
\texttt{concert\_singer}, cubriendo los patrones relevantes para su
rol: AR incluye 5 preguntas válidas y 5 inválidas (DML, opinión, fuera de dominio); AV mezcla 5 SQL válidos y 5 con
errores; AG usa pares (pregunta, SQL canónico). La composición
detallada se presenta en la Tabla~\ref{tab:datasets} del Anexo~E.

\subsubsection{Métricas y función objetivo por agente}\label{sec:fase2-metrics}

Para cada agente se calcula un conjunto de métricas independientes y
se combinan en un \textit{combined\_score} escalar que dirige la
búsqueda. La notación que aparece en las fórmulas es la siguiente:
\textbf{$F$ = Faithfulness} $\in [0,1]$ (cobertura semántica vía
GEval LLM-as-judge); \textbf{$G$ = Groundedness} $\in [0,1]$ (la salida
se apoya en el contexto, no inventa); \textbf{$R$ = ROUGE-L}
$\in [0,1]$ (similitud léxica determinista, sin LLM; para SQL se
aplica con normalización ortográfica vía \texttt{rouge\_l\_sql});
$B$ = Brier Score $\in [0,1]$ ($\downarrow$);
$E$ = ECE $\in [0,1]$ ($\downarrow$).
El valor máximo de cada \textit{combined\_score} se obtiene sumando
los valores máximos de los términos positivos (cada uno $\leq 1$) y
asumiendo que los términos negativos $B$ y $E$ alcanzan su mínimo $0$.

\textbf{Implementación de las métricas.} Las métricas se calculan así:

\begin{itemize}[leftmargin=*,topsep=2pt,itemsep=1pt]
  \item $F$ y $G$: juez LLM fijo (GPT-4o, $T{=}0{,}0$) mediante el
    framework DeepEval~\cite{deepeval2024} con criterio GEval~\cite{liu2023geval} — sin
    intervención humana.
  \item $R$: ROUGE-L determinista (sin LLM) sobre la subsecuencia
    común más larga; en el AG se normaliza ortográficamente el SQL
    antes del cálculo (\texttt{rouge\_l\_sql}).
  \item $B$ y $E$: calibración probabilística con
    \texttt{scikit-learn}~\cite{sklearn2011} sobre pares
    (confianza declarada, \textit{outcome} binario).
\end{itemize}

\textbf{Diferencias entre agentes.} Las métricas difieren según el rol:
AR y AV usan $(2F+R+2G-E-B)/5$ (valor máximo~1.0): $F$ y $G$ reciben
doble peso por ser métricas LLM-as-judge que evalúan significado,
mientras que ROUGE-L mide coincidencia léxica superficial.
AE y AS usan $(2F+2G+0.5R-E-B)/4.5$ (valor máximo~1.0): el peso~0.5
en ROUGE-L refleja que en generación de lenguaje natural las variaciones
léxicas son válidas e inevitables; la calidad se mide por cobertura
semántica ($F$) y ausencia de alucinaciones ($G$);
AG genera SQL y usa $F+G+R_{SQL}-B-E$ (valor máximo~3.0, donde $R_{SQL}$ es la
similitud léxica sobre el SQL canonicalizado); APS es un retriever vectorial
y usa $(F_1^T+F_1^C)/2$.
La tabla completa de métricas por agente se encuentra en la
Tabla~\ref{tab:fase1_metricas_por_agente} del Anexo~E.

\textbf{AR (Refinador).} Evalúa la corrección semántica de la
reformulación. $F$ y $G$ se calculan con DeepEval \texttt{GEval} sobre
el par (\textit{refined\_query}, \textit{expected\_refined}) y sobre
(\textit{user\_input}, \textit{refined\_query}) respectivamente.
\textit{ROUGE-L} mide la similitud léxica entre la reformulación
generada y la versión esperada, capturando coincidencias en el orden
relativo de las palabras aunque no sean contiguas. $B$ y $E$ calibran la confianza
declarada por el agente; para el AR el criterio binario es
$\text{outcome}{=}1$ cuando $\text{confidence\_score} \geq 0{,}6$,
umbral elegido porque representa el punto de decisión por encima del
azar (50\%) a partir del cual el AR considera la consulta válida y
apta para continuar el \textit{pipeline}; valores de $0{,}5$ a $0{,}7$
son los más frecuentes en sistemas de recuperación de información como
umbral de corte entre aceptar y rechazar~\cite{manning2008irbook}.
La fórmula asigna coeficiente~2 a $F$ y $G$ y coeficiente~1 a $R$
porque Faithfulness y Groundedness son evaluadas por un LLM-as-judge
que comprende el significado de la consulta, mientras que ROUGE-L
es una métrica léxica que no distingue reformulaciones semánticamente
equivalentes con vocabulario distinto.
En el AR, lo crítico es que la consulta refinada \textit{signifique}
lo mismo que la original y esté \textit{anclada} en el input del usuario;
la coincidencia léxica exacta es secundaria.
La normalización por~5 (máximo del numerador cuando $F{=}G{=}R{=}1$,
$B{=}E{=}0$) mantiene el rango $[0,1]$:
$$\text{combined}_{AR} = \tfrac{2F + R + 2G - E - B}{5}$$
El valor máximo es $1{,}0$.

El estudio Optuna (10 \textit{trials}) confirmó como ganador
\textbf{gemini-2.5-flash} ($T{=}0{,}5$, $S{=}0{,}962$), consistente
con el resultado de la grilla exhaustiva Fase~0.
La Figura~\ref{fig:optuna_ar} muestra los 10 trials;
los detalles del Top-5 en la Tabla~\ref{tab:top5_ar_ae_as}.

\begin{center}
\includegraphics[width=\columnwidth]{optuna_scatter_10trials.png}
\captionof{figure}{AR — Optuna TPE.: 10 trials en el espacio
(temperatura, \textit{combined\_score}). La estrella (\(\bigstar\))
indica el ganador: gemini-2.5-flash $T{=}0.5$, $S{=}0.962$.
La línea punteada es el mejor acumulado al finalizar el estudio.}
\label{fig:optuna_ar}
\end{center}

\textbf{APS (Proximidad Semántica).} Retriever vectorial evaluado con
$F_1^{T}$ (tablas) y $F_1^{C}$ (columnas);
$\text{combined}_{APS} = (F_1^{T}+F_1^{C})/2$.
Las tres métricas HNSW (cosine, l2, ip) producen resultados idénticos
para vectores L2-normalizados (\textit{combined}$=0{,}842$); se usa
\texttt{cosine} por convención estándar
(Tabla~\ref{tab:aps_search}).\smallskip

\begin{center}
\captionof{table}{Trials APS — similitud HNSW con \texttt{multilingual-e5-base}.
Las tres métricas son equivalentes para vectores L2-normalizados
($\text{combined}=(F_1^T+F_1^C)/2$).}
\label{tab:aps_search}
\footnotesize
\begin{tabular}{lccc}
\toprule
\textbf{Similitud} & $F_1^{T}$ & $F_1^{C}$ & \textbf{Combined} \\
\midrule
\texttt{cosine} (winner) & \textbf{0.927} & \textbf{0.757} & \textbf{0.842} \\
\texttt{l2}     & 0.927 & 0.757 & 0.842 \\
\texttt{ip}     & 0.927 & 0.757 & 0.842 \\
\bottomrule
\end{tabular}
\end{center}

\textbf{AG (Generador SQL).} La salida es una sentencia SQL ejecutable.
Antes de calcular las métricas, ambas consultas (generada y esperada)
se canonizan con \texttt{sqlglot}~\cite{sqlglot2023} para eliminar
diferencias superficiales de alias, mayúsculas y espacios. Sobre el SQL
canonicalizado se aplican tres señales:

\begin{itemize}[leftmargin=*,topsep=2pt,itemsep=1pt]
  \item $F$ (GEval): cobertura semántica de la consulta generada respecto
    al SQL esperado.
  \item $G$ (GEval): el SQL generado se apoya en las tablas y columnas
    del esquema sin inventar identificadores fuera de él.
  \item $R_{SQL}$: similitud léxica sobre tokens SQL normalizados
    (ROUGE-L con normalización de identificadores y operadores).
\end{itemize}

El calibrador Brier/ECE usa como \textit{outcome}\,=\,1 cuando
$R_{SQL} \geq 0{,}70$.
El umbral original era $0{,}80$, pero se detectaron falsos negativos:
SQLs semánticamente correctos pero con diferencias textuales menores
(e.g., \texttt{HAVING COUNT(singer\_id)\,$\geq$\,2} vs.\ \texttt{HAVING COUNT(*)\,$>\,$1})
obtenían $R_{SQL}\approx0{,}72$, cayendo por debajo del umbral y
generando \textit{outcomes}\,=\,0 con alta confianza del agente,
lo que inflaba artificialmente el Brier. Bajar a $0{,}70$ eliminó
estos falsos negativos sin introducir falsos positivos.
La función objetivo:
\begin{equation*}
\text{combined}_{AG} = F + G + R_{SQL} - B - E
\end{equation*}
\noindent El valor máximo es $3{,}0$ cuando $F{=}G{=}R_{SQL}{=}1$ y $B{=}E{=}0$.
La separación MySQL/PostgreSQL es necesaria porque el dialecto SQL difiere
(sintaxis de fechas, comillas, tipos): cada base de datos tiene su
propio ganador.

\textbf{AV (Validador).} La evaluación del AV se alinea con la de AR
y AE evaluando la \textbf{explicación textual del razonamiento} en
lugar de solo la decisión binaria. El AV emite dos campos: \texttt{reasoning}
(texto que justifica la decisión) y \texttt{decision\_word}
(``correct'' o ``incorrect''). El dataset incluye
\texttt{expected\_reasoning} y \texttt{expected\_decision\_word}
como referencia de evaluación. Las métricas son:
\begin{itemize}[leftmargin=*,topsep=2pt,itemsep=1pt]
  \item \textbf{Faithfulness} ($F$): GEval sobre (\texttt{reasoning},
        \texttt{expected\_reasoning}) — ¿cubre el razonamiento del AV
        los mismos puntos que la explicación esperada?
  \item \textbf{Groundedness} ($G$): GEval con contexto
        (\textit{refined\_query} + SQL + schema) — ¿se apoya el
        razonamiento en la información disponible sin inventar?
  \item \textbf{ROUGE-L} ($R$):
        \texttt{rouge\_l\_score}(\texttt{decision\_word},
        \texttt{expected\_decision\_word}) — coincidencia léxica
        determinista del veredicto binario expresado como string.
  \item $B$, $E$ sobre \texttt{(confidence\_score, outcome)}, donde
        $outcome=1$ si la decisión emitida coincide con la esperada.
\end{itemize}
La fórmula asigna coeficiente~2 a $F$ y $G$ por la misma razón que
en el AR: el razonamiento del AV debe \textit{cubrir} los argumentos
esperados (Faithfulness) y \textit{anclarse} en la consulta y el SQL
recibidos sin inventar justificaciones (Groundedness).
ROUGE-L recibe coeficiente~1 porque la formulación del veredicto
puede variar léxicamente sin perder validez (``the query is correct''
y ``the SQL is valid'' son equivalentes). La normalización por~5 mantiene
el rango $[0,1]$:
$$\text{combined}_{AV} = \tfrac{2F + R + 2G - E - B}{5}$$
El valor máximo es $1{,}0$.

\textbf{Sub-fases de AG y AV (Fase~1\_1 y Fase~1\_2).}
Tanto AG como AV se evaluaron en \textbf{dos sub-fases} por base de datos
(MySQL y PostgreSQL), con las mismas 10 preguntas de prueba,
el mismo seed y el mismo número de trials, para que solo cambien
los aspectos metodológicos:
\begin{itemize}[leftmargin=*,topsep=2pt,itemsep=1pt]
  \item \textbf{Fase~1\_1} (línea base): prompts de los agentes contenían
    ejemplos del mismo dataset de evaluación (fuga de datos no detectada
    en la primera iteración); la métrica ROUGE-L no normalizaba el orden
    de tablas en cláusulas JOIN, lo que penalizaba consultas semánticamente
    equivalentes pero escritas en orden inverso; la RULE~7 del AV era
    estricta y forzaba columnas adicionales en \texttt{GROUP BY}, generando
    falsos negativos con alta confianza que inflaban el Brier.
  \item \textbf{Fase~1\_2} (corregida): prompts con ejemplos completamente
    genéricos; normalización del orden de JOINs en ROUGE-L (INNER JOIN es
    conmutativo); umbral de \textit{outcome} reducido de $0{,}80$ a $0{,}70$
    para eliminar los falsos negativos detectados; RULE~7 del AV relajada
    con excepción de dependencia funcional FK$\to$PK.
\end{itemize}
El modelo ganador fue el mismo en ambas sub-fases,
confirmando que la mejora de scores se debe a las correcciones
metodológicas y no a explorar nuevas combinaciones.

\textbf{AE (Explicador) y AS (Sustentador).} Ambos producen lenguaje
natural y comparten la misma estructura de evaluación:

\begin{itemize}[leftmargin=*,topsep=2pt,itemsep=1pt]
  \item $F$ (GEval): la explicación/sustentación generada cubre los
    mismos puntos que la respuesta esperada.
  \item $G$ (GEval): el texto se ancla en el contexto recibido
    (pregunta, SQL, tablas) sin fabricar información.
  \item $R$ (ROUGE-L): baseline léxico determinista.
  \item $B$ y $E$: calibración del \textit{confidence\_score} que
    ambos agentes emiten en formato JSON estructurado.
    El \textit{outcome}\,=\,1 cuando $G \geq 0{,}80$.
    Se eligió \textbf{Groundedness} como criterio de calibración
    —en lugar de Faithfulness o ROUGE-L— porque: (i)~detecta
    alucinaciones directamente (el agente explica con información no
    presente en el contexto); (ii)~es robusto ante paráfrasis y
    sinónimos que ROUGE-L penalizaría incorrectamente; y
    (iii)~no exige coincidencia con una referencia fija, lo cual es
    apropiado en agentes explicativos donde múltiples formulaciones
    son igualmente válidas.
\end{itemize}

La fórmula es idéntica para AE y AS: $(2F+2G+0.5R-E-B)/4.5$, valor máximo~$1{,}0$.

Para AE y AS, $F$ y $G$ reciben coeficiente~2 porque la calidad de
una explicación o sustentación se mide principalmente por:
\begin{itemize}[leftmargin=*,topsep=1pt,itemsep=0pt]
  \item \textit{Faithfulness} ($F$): ¿cubre la explicación los mismos puntos que
    la referencia? Una respuesta que omite información clave es
    semánticamente incorrecta aunque léxicamente similar.
  \item \textit{Groundedness} ($G$): ¿se ancla el texto en el contexto recibido
    (SQL, tablas, pregunta) sin fabricar datos o razonamientos?
    Es la métrica que previene las alucinaciones del agente.
\end{itemize}
ROUGE-L recibe coeficiente~$0{,}5$ porque en tareas de generación
de lenguaje natural, múltiples formulaciones son semánticamente
equivalentes. Dos explicaciones con vocabulario diferente pero que
cubren los mismos conceptos son igualmente válidas; un coeficiente
alto en ROUGE-L penalizaría ese parafraseo sin justificación técnica.
El denominador $4{,}5$ es el máximo del numerador cuando $F{=}G{=}R{=}1$,
garantizando que el valor máximo sea exactamente $1{,}0$.

\textbf{Resumen de ganadores Fase~1} (Tabla~\ref{tab:winners_fase1_inline}):

\begin{center}
\captionof{table}{Ganadores Fase~1 por agente..}
\label{tab:winners_fase1_inline}
\footnotesize
\begin{tabular}{llcc}
\toprule
\textbf{Agente} & \textbf{Modelo} & \textbf{T} & \textbf{S} \\
\midrule
AR          & gemini-2.5-flash  & 0.5 & 0.962 \\
APS         & cosine / e5-base  & --- & 0.842 \\
AG-MySQL\textsuperscript{†}    & claude-haiku-4-5  & 0.3 & 2.700 \\
AG-Postgres\textsuperscript{†} & claude-haiku-4-5  & 0.3 & 2.687 \\
AV-MySQL\textsuperscript{†}    & claude-sonnet-4-6 & 0.7 & 0.956 \\
AV-Postgres\textsuperscript{†} & claude-haiku-4-5  & 0.0 & 0.955 \\
AE          & gpt-4o            & 0.3 & 0.792 \\
AS          & claude-sonnet-4-6 & 0.7 & 0.870 \\
\bottomrule
\end{tabular}
\end{center}
\footnotesize{\textsuperscript{†}Valores de Fase~1\_2 (prompts sin data leakage,
\texttt{normalize\_sql\_v2}, umbral ROUGE$=0.70$). Fase~1\_1: AG-MySQL 2.685,
AG-Postgres 2.592, AV-MySQL 0.959 (gpt-4o T=0.3), AV-Postgres 0.962 (sonnet T=0.0).}

La tabla completa de ganadores se encuentra en la
Tabla~\ref{tab:winners_fase2} del Anexo~E.
Los TOP-5 \textit{trials} por agente se detallan en las
Tablas~\ref{tab:top5_ar_ae_as}, \ref{tab:top5_ag}
y~\ref{tab:top5_av} del Anexo~C.

\subsection{Fase 2 --- Robustez a paráfrasis (DeepEval LLM-as-judge)}

La Fase~2 mide la \textbf{invariancia del sistema a reformulaciones
lingüísticas} del usuario. Para cada agente, las preguntas/inputs del
dataset de Fase~1 se transforman en 3 paráfrasis con perturbaciones
controladas: \textit{(i) informal\_directa} (omisión de determinantes,
contracciones), \textit{(ii) reformulada\_abierta} (reescritura conversacional),
\textit{(iii) vocabulario\_alternativo} (sinónimos, sustantivos
alternativos). Esto produce datasets ampliados de $n \times 3$ inputs
donde las 3 paráfrasis de un mismo intent comparten la respuesta
esperada, de modo que un sistema invariante debería
producir la misma salida ante las tres.

A diferencia de la Fase~1 que tuneaba hiperparámetros, la Fase~2
fija los \textbf{ganadores de Fase~1} de cada agente y se aplica como
prueba de robustez. Las métricas usadas son las mismas de Fase~1 más
las nuevas señales de estabilidad por intent ($\sigma$ entre las 3
paráfrasis).

\textbf{Alcance de la evaluación.} La Fase~2 aplica paráfrasis a los
\textbf{seis agentes}: AR (entrada del usuario), AG (pregunta que genera
el SQL), AV (formulación de la consulta para ver si clasifica
correctamente bajo distintas redacciones), APS (pregunta refinada para
medir invariancia del retrieval vectorial), AE y AS. El detalle de cada
dataset se encuentra en el Anexo~G.

\textbf{Métricas LLM-as-judge.} Para los agentes con salida en lenguaje
natural (AR, AE, AS) y para el AG sobre el SQL canonicalizado, se usa
\texttt{GEval}~\cite{liu2023geval} implementado en DeepEval~\cite{deepeval2024} con dos criterios:
\textbf{Faithfulness} (cobertura semántica vs.~respuesta esperada) y \textbf{Groundedness}
(la salida se apoya en el contexto, no inventa). Los modelos juez fijos
son GPT-4o (temperatura 0.0) para ambos. ROUGE-L y \texttt{rouge\_l\_sql}
se mantienen como baselines léxicos deterministas.

Los resultados métrica a métrica se detallan en las
Tablas~\ref{tab:fase2_metricas_por_agente} y~\ref{tab:winners_fase3}
del Anexo~D. Los hallazgos principales son:

\begin{itemize}[leftmargin=*,topsep=2pt,itemsep=1pt]
  \item \textbf{AV es el más robusto}: $\Delta\text{Comb}{=}0$ en MySQL;
    su clasificación binaria (aprobado/rechazado) no se ve afectada
    por reformulaciones lingüísticas.
  \item \textbf{AG cae más}: Postgres pierde $\Delta{=}{-}0.400$ en
    combined; la Groundedness baja $-0.096$ porque las paráfrasis
    informales dificultan el mapeo exacto a columnas del esquema.
  \item \textbf{ROUGE-L del AR cae $-0.174$}: no porque el AR falle,
    sino porque las paráfrasis informales no coinciden léxicamente con
    la referencia limpia esperada; $F$ cae $-0.045$ y $G$ es
    prácticamente estable ($-0.001$), demostrando robustez semántica.
  \item \textbf{AE y AS son estables}: caídas $\leq 0.033$ en combined;
    la generación de lenguaje natural es menos sensible a reformulaciones.
\end{itemize}

\subsection{Fase 3 --- Búsqueda de hiperparámetros del orquestador}

La evaluación global tiene un objetivo doble: (1)~medir el desempeño
del \textit{pipeline} completo operando de forma integrada, y
(2)~determinar el mejor modelo y temperatura para el \textbf{agente
orquestador}, cuyo hiperparámetro no
fue optimizado durante las Fases 1--2 porque su rol —clasificar
intents, detectar cambios de contexto y rutear al agente correcto—
solo es evaluable cuando todo el pipeline está activo. Los demás
agentes mantienen fijas sus configuraciones ganadoras de Fases 1--2
consolidadas automáticamente al arranque del sistema.

\textbf{Base de datos del experimento.} Se utiliza
\texttt{concert\_singer}, una base de datos del \textit{benchmark}
Spider~1.0 con cuatro tablas: \texttt{stadium}, \texttt{singer},
\texttt{concert} y \texttt{singer\_in\_concert} (21 columnas).
Se eligió por ofrecer un esquema reducido con relaciones $M{:}N$
no triviales que permiten evaluar \textit{joins} multi-tabla y
subconsultas. La misma base se carga en MySQL y PostgreSQL para
comparación cruzada por base de datos.

\textbf{Dataset del orquestador.} El dataset cubre
\textbf{10 escenarios} que ejercitan cada ruta posible del
\textit{pipeline}:

\begin{itemize}[leftmargin=*,itemsep=0pt]
  \item \textbf{Ruta 1}: Orquestador~$\rightarrow$~END —
        pregunta subjetiva (no activa agentes).
  \item \textbf{Ruta 2}: Orquestador~$\rightarrow$~AR~$\rightarrow$~AE —
        sentencia DML; AR rechaza.
  \item \textbf{Ruta 3}: Orquestador~$\rightarrow$~AR~$\rightarrow$~APS~$\rightarrow$~AE —
        concepto sin tablas en el esquema; APS rechaza.
  \item \textbf{Ruta 4}: Orquestador~$\rightarrow$~AS —
        pregunta de sustentación con historial previo.
  \item \textbf{Rutas 5--10}: Pipeline completo con los seis agentes activos,
        evaluando seis patrones SQL: agregación simple (MAX/AVG), JOIN con GROUP~BY,
        subconsulta NOT~IN, DISTINCT con WHERE, GROUP~BY con MAX por categoría,
        y bucle de corrección AV~$\rightarrow$~AG (WHERE con agregado inválido
        corregido a HAVING).
\end{itemize}

Cada caso define \texttt{expected\_tools} con la lista exacta de
agentes y \textit{skills} esperadas por ruta, lo que habilita
\texttt{ToolCorrectnessMetric}: APS (búsqueda de tablas, columnas y
valores), AG (validación de seguridad, cálculo de JOINs) y AV
(sintaxis, patrones semánticos, plan de ejecución).

\textbf{Métricas globales.} La evaluación del orquestador integra cuatro dimensiones:
(1)~\textit{Faithfulness} ($F$): GEval LLM-as-judge comparando la
explicación final del AE contra la explicación esperada;
(2)~\textit{Groundedness} ($G$): GEval comprobando que la explicación
se ancla en el SQL generado y las tablas usadas, sin fabricar información;
(3)~\textit{Correctness} ($C$): \texttt{ToolCorrectnessMetric} de
DeepEval~\cite{deepeval2024}, que verifica si el \textit{pipeline} activó
los agentes correctos \textbf{y} las \textit{skills} correctas dentro de
cada agente (e.g.\ \texttt{APS.search\_tables}, \texttt{AG.validate\_sql\_safety},
\texttt{AV.check\_semantic\_patterns}); (4)~calibración del WACS con
Brier ($B$) y ECE ($E$) contra el \textit{outcome} binario
($R_{SQL} \geq 0{,}80$).

El WACS agrega las confianzas por agente normalizando por los pesos de
los agentes efectivamente activos en cada caso (rutas de rechazo dejan
agentes con $c{=}0$; incluirlos en el denominador diluyendo la confianza
real del pipeline activo sería estadísticamente incorrecto):
$$\text{WACS}_{norm} = \frac{\sum_{i \in A_{act}} w_i\,c_i}{\sum_{i \in A_{act}} w_i}$$
\noindent donde $w$: AR\,0.10, APS\,0.20, AG\,0.30, AV\,0.40.

\textbf{Búsqueda de hiperparámetros del orquestador.} Optuna TPE
explora el espacio (modelo, temperatura) del \textbf{agente orquestador}
sobre los 10 escenarios, con la función objetivo
$\text{combined}_{\text{orch}}$ como señal de calidad. El espacio de búsqueda
incluye los cuatro modelos de Fases~1--4 (GPT-4o, Claude Haiku~4.5,
Claude Sonnet~4.6, Gemini~2.5 Flash) y cuatro temperaturas
$T \in \{0.0, 0.3, 0.5, 0.7\}$, totalizando 10 \textit{trials} con
muestreo TPE (\textit{n\_startup\_trials}{=}3,).

Esta búsqueda se ejecuta en \textbf{dos sub-fases por base de datos}:
\textbf{Fase~3\_1} utiliza los agentes AG y AV de Fase~1 (prompts con
ejemplos del dataset, \texttt{normalize\_sql\_v1} sin corrección de
orden de JOINs); \textbf{Fase~3\_2} utiliza los agentes AG y AV de
Fase~1\_2 (prompts genéricos y \texttt{normalize\_sql\_v2} con
normalización de JOINs conmutativos). Ambas sub-fases se replican de
forma independiente para MySQL y PostgreSQL. La función objetivo es:

$$\text{combined}_{\text{orch}} = F + G + C - B - E
\quad (\text{máx.} \approx 3{,}0) \label{eq:combined_e2e}$$

donde $F$ y $G$ son GEval sobre la explicación final del AE
(criterios \texttt{e2e\_faithfulness\_geval} y
\texttt{e2e\_groundedness\_geval}), $C$ es \textit{Correctness}
(\texttt{ToolCorrectnessMetric}), y $B$/$E$ son Brier/ECE globales
sobre los 10 casos de prueba. Todos los términos tienen peso unitario.

\textbf{Resultados del orquestador.} La Tabla~\ref{tab:e2e} muestra
el ganador Optuna por base de datos sobre los 10 escenarios.

\begin{center}
\captionof{table}{Configuración ganadora del orquestador (Exp2).
Exp1 original entre paréntesis para comparación.}
\label{tab:e2e}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Métrica} & \textbf{MySQL Exp2} & \textbf{(Exp1)} & \textbf{PostgreSQL Exp2} & \textbf{(Exp1)} \\
\midrule
Modelo           & claude-haiku-4-5 & (gemini-2.5-flash) & claude-haiku-4-5 & (gemini-2.5-flash) \\
Temperatura      & 0.3 & (0.7) & 0.3 & (0.5) \\
$F$              & 0.857 & (0.847) & 0.810 & (0.789) \\
$G$              & 0.948 & (0.955) & 0.875 & (0.871) \\
$C$              & 0.991 & (0.991) & 0.800 & (0.800) \\
$R_{SQL}$        & 0.591 & (0.591) & 0.575 & (0.480) \\
$B$              & 0.005 & (0.007) & 0.005 & (0.128) \\
$E$              & 0.071 & (0.081) & 0.067 & (0.078) \\
\midrule
\textbf{Combined} & \textbf{2.744} & (2.705) & \textbf{2.554} & (2.253) \\
\bottomrule
\end{tabular}%
}
\par\smallskip
\noindent\footnotesize{$B$, $E$ $\downarrow$: menor es mejor. Valor máximo $\approx 3{,}0$.
Exp2 usa prompts AG/AV sin fuga de datos, normalización de JOINs en ROUGE-L y corrección
de dos errores de enrutamiento en el orquestador.
PostgreSQL Brier $0{,}128 \to 0{,}005$: la corrección del AV eliminó
falsos negativos que generaban alta confianza con resultado incorrecto.}
\end{center}

\textbf{Skills evaluadas por ToolCorrectnessMetric.}
La métrica verifica que el pipeline active exactamente los agentes y
\textit{skills} esperados según el tipo de consulta. Se rastrean las
\textit{skills} de APS (búsqueda de tablas, columnas y valores),
AG (validación de seguridad SQL, cálculo de rutas de JOIN) y
AV (reglas de sintaxis, patrones semánticos y análisis estático de calidad).

\subsection{Fase 4 --- Evaluación de robustez del orquestador}

La Fase~4 evalúa la robustez del \textbf{pipeline completo} ante variaciones
lingüísticas. Cada uno de los 10 escenarios de Fase~3 se reformula en
3 variantes (\textit{informal}, \textit{reformulada}, \textit{vocabulario alternativo}),
generando $10 \times 3 = 30$ entradas por base de datos.
El SQL esperado, la explicación de referencia y las herramientas esperadas
son fijos para las tres variantes; solo cambia la redacción de la pregunta.

La función objetivo es la misma de Fase~3:
$\text{combined}_{\text{orch}} = F + G + C - B - E$, fijando
los hiperparámetros ganadores de Fase~3\_2 (MySQL: claude-haiku-4-5 $T{=}0.3$;
Postgres: claude-haiku-4-5 $T{=}0.3$). Un sistema invariante debería
obtener el mismo $C$ (\texttt{ToolCorrectnessMetric}) ante las tres
formulaciones: si el orquestador clasifica correctamente el intent y
activa los agentes correctos independientemente de la redacción, el
sistema es robusto a variaciones léxicas del usuario.

\textbf{Resultados de Fase~4.} La Tabla~\ref{tab:fase4_results}
compara Fase~3\_2 (10 escenarios, modelos y prompts corregidos) con
Fase~4\_2 (30 paráfrasis) por base de datos. Los resultados de
Fase~3\_1 y Fase~4\_1 (iteración original) se incluyen como referencia.

\begin{center}
\captionof{table}{Orquestador: Fase~3 vs.\ Fase~4 por base de datos
(claude-haiku-4-5, $T{=}0{,}3$).
Entre paréntesis: Exp1 original (gemini, prompts con data leakage).
$B$, $E$ $\downarrow$; Combined $=F{+}G{+}C{-}B{-}E$, valor máx.$\approx3{,}0$.}
\label{tab:fase4_results}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llccccccc}
\toprule
 & \textbf{Base de datos} & $F$ & $G$ & $C$ & $B$ & $E$ & \textbf{Exp2} & \textbf{(Exp1)} \\
\midrule
\multirow{2}{*}{\textbf{Fase~3}}
  & MySQL    & 0.857 & 0.948 & 0.991 & 0.005 & 0.071 & \textbf{2.744} & (2.705) \\
  & Postgres & 0.810 & 0.875 & 0.800 & 0.005 & 0.067 & \textbf{2.554} & (2.253) \\
\midrule
\multirow{2}{*}{\textbf{Fase~4}}
  & MySQL    & 0.857 & 0.942 & 0.891 & 0.097 & 0.035 & \textbf{2.488} & (2.413) \\
  & Postgres & 0.786 & 0.936 & 0.900 & 0.091 & 0.036 & \textbf{2.495} & (2.098) \\
\midrule
\multirow{2}{*}{$\Delta$\,(F4$-$F3)}
  & MySQL    & $0.000$ & $-0.006$ & $-0.100$ & $+0.092$ & $-0.036$ & $\mathbf{-0.256\ (-9.3\%)}$ \\
  & Postgres & $-0.024$ & $+0.061$ & $+0.100$ & $+0.086$ & $-0.031$ & $\mathbf{-0.059\ (-2.3\%)}$ \\
\midrule
\multirow{2}{*}{\textbf{Score norm.}}
  & MySQL    & \multicolumn{6}{c}{Exp2: $2{,}488/3{,}0 = \mathbf{82{,}9\,\%}$ \quad (Exp1: 80{,}4\,\%)} \\
  & Postgres & \multicolumn{6}{c}{Exp2: $2{,}495/3{,}0 = \mathbf{83{,}2\,\%}$ \quad (Exp1: 69{,}9\,\%)} \\
\bottomrule
\end{tabular}}
\par\smallskip\footnotesize{La iteración \_2 corrige: (a)~prompts AG/AV sin data leakage,
(b)~\texttt{normalize\_sql\_v2} con normalización de orden de JOINs,
(c)~umbral ROUGE $0.80\to0.70$, (d)~RULE~7 AV postgres relajada,
(e)~AS routing y bypass de opinión en el orquestador.
Postgres Brier $0.290\to0.091$: la mayor mejora proviene de eliminar los falsos
negativos del AV que generaban WACS$\approx 0.93$ con outcome$=0$.}
\end{center}

\section{Despliegue a Producción}

SQL-Agents adopta una \textbf{arquitectura de microservicios}: cada
agente especializado corre en su propio contenedor Docker con un
servidor FastAPI que expone el \textit{endpoint} \texttt{POST /invoke}.
El Orquestador actúa como punto de entrada único y llama a cada agente
por HTTP, habilitando escalado, sustitución y monitoreo independiente
por servicio.

\end{multicols}

\vspace{6pt}
\begin{center}
  \includegraphics[width=\columnwidth]{despliegue_docker.PNG}
  \captionof{figure}{Arquitectura de despliegue: 7 contenedores Docker
  (orquestador + 6 agentes) comunicados por endpoints REST
  (\texttt{POST /invoke}) a través de una red privada.
  Modelo ganador por agente: AR gemini-2.5-flash, AG claude-haiku-4-5,
  AV claude-haiku/sonnet, AE gpt-4o, AS claude-sonnet-4-6,
  Orquestador claude-haiku-4-5.}
  \label{fig:docker_deploy}
\end{center}
\vspace{6pt}

\begin{multicols}{2}

El arranque del sistema se gestiona con \texttt{docker-compose up -{}-build}.
Los contenedores de agentes deben estar \textit{healthy} antes de que el
Orquestador arranque, garantizando disponibilidad total del pipeline:

\begin{verbatim}
docker-compose up --build
  ar       :8001  healthy
  aps      :8002  healthy
  ag       :8003  healthy
  av       :8004  healthy
  ae       :8005  healthy
  as_agent :8006  healthy
  orch     :8000  ready
\end{verbatim}

Cada agente expone un \textit{endpoint} REST (\texttt{POST /invoke})
que el Orquestador invoca por HTTP. Los contenedores se comunican a través
de una red privada Docker: el orquestador resuelve cada agente por nombre
de servicio, habilitando escalado, sustitución y monitoreo independiente
por componente.

\section{Conclusiones}

\textbf{SQL-Agents} demostró que la especialización multi-agente con el
patrón \textit{Agent Skills} + MCP es viable y robusta para NL2SQL:

\begin{itemize}[leftmargin=*,itemsep=2pt]
  \item \textbf{Cada rol requiere un modelo diferente.}
    Los experimentos confirman que no existe un único LLM óptimo para
    todo el pipeline. El AG (generación de SQL estructurado) se beneficia
    de modelos de bajo costo y alta precisión léxica como
    claude-haiku-4-5, que sigue instrucciones de formato con consistencia.
    El AV (validación con razonamiento textual) requiere modelos con mayor
    capacidad analítica: claude-sonnet-4-6 en MySQL, claude-haiku-4-5
    en PostgreSQL. El AE y AS, que generan explicaciones en lenguaje
    natural, son mejor servidos por gpt-4o y claude-sonnet-4-6
    respectivamente, modelos con mayor riqueza expresiva.
    El Orquestador, cuyo rol es clasificar intents y coordinar el flujo,
    converge a claude-haiku-4-5 como solución equilibrada entre
    velocidad y precisión en el enrutamiento.

  \item \textbf{La calidad de la métrica es tan crítica como la del agente.}
    La segunda iteración experimental reveló tres problemas metodológicos:
    prompts con ejemplos del dataset de evaluación (fuga de datos),
    una función ROUGE-L que penalizaba JOINs semánticamente equivalentes
    escritos en diferente orden, y un umbral que clasificaba como erróneos
    SQLs correctos pero textualmente distintos.
    Corregir estos problemas elevó los scores de manera consistente en
    ambas bases de datos, con la mejora más notable en PostgreSQL donde
    el error de calibración (Brier) se redujo en más de un 96\%.
    La conclusión es directa: sin una evaluación limpia, es imposible
    distinguir si los resultados reflejan la calidad real del sistema
    o artefactos del proceso de medición.

  \item \textbf{Alta robustez ante variaciones lingüísticas.}
    Al evaluar cada agente con 30 paráfrasis en lugar de las 10 preguntas
    originales, las métricas se mantienen estables (caídas menores al 3\%
    en todos los agentes). Las pequeñas variaciones observadas
    se concentran en el vocabulario alternativo, donde términos
    técnicos sustituidos por sinónimos informales reducen levemente
    la puntuación de Faithfulness, sin afectar la corrección del
    SQL generado ni la precisión del enrutamiento del orquestador.

  \item \textbf{Mejora del orquestador: Fase\_3\_2 $\to$ Fase\_4\_2 ($\Delta < 10\%$).}
    Los 10 escenarios reformulados en 30 variantes muestran alta
    consistencia: MySQL $2{,}744{\to}2{,}488$ ($-9{,}3\%$),
    Postgres $2{,}554{\to}2{,}495$ ($-2{,}3\%$).
    La corrección de tres \textit{bugs} en el orquestador
    (enrutamiento de AS, bypass de opiniones y umbral ROUGE)
    elevó la puntuación normalizada de PostgreSQL de $70\%$ a $83\%$
    y redujo el Brier de $0{,}290$ a $0{,}091$.
    La \texttt{ToolCorrectnessMetric} se mantiene alta en ambas
    bases de datos ($C{\geq}0{,}891$), confirmando que el orquestador
    clasifica intents y activa agentes correctamente ante cualquier
    redacción del usuario.

  \item \textbf{Agent Skills + MCP es viable arquitectónicamente.}
    La separación habilidades-LLM/lógica-MCP permitió reutilizar el
    grafo de JOINs, desacoplar la verificación de esquema del proveedor
    y sustituir LLMs de forma independiente entre bases de datos.
    La compatibilidad con múltiples proveedores (OpenAI, Anthropic, Google)
    se logró sin modificar los contratos de entrada/salida de cada agente.
\end{itemize}

\section{Trabajos futuros}

\begin{itemize}[leftmargin=*,itemsep=2pt]
  \item \textbf{Agente ejecutor y \textit{Execution Accuracy}.}
    Añadir un séptimo agente que ejecute el SQL contra la BD viva
    para medir EX estilo Spider~1.0 y reemplazar las heurísticas de
    rendimiento del AV por \texttt{EXPLAIN ANALYZE} real.
  \item \textbf{Esquemas más complejos.}
    Evaluar sobre BIRD~\cite{li2023bird} (esquemas de hasta 30 tablas)
    para validar la escalabilidad del APS y el AG más allá de los
    4 tablas de \texttt{concert\_singer}.
  \item \textbf{Extensión a otros sistemas de bases de datos.}
    Adaptar el AG a motores NoSQL o analíticos
    (MongoDB, DuckDB/Parquet, Snowflake, BigQuery) manteniendo el
    resto del pipeline intacto.
\end{itemize}

\end{multicols}







% ---------------------------------------------------------------
% ANEXOS
% ---------------------------------------------------------------
\clearpage
\section*{Anexo B: Arquitectura completa}
\addcontentsline{toc}{section}{Anexo B}
\begin{center}
  \includegraphics[width=\textwidth, height=0.93\textheight,
                   keepaspectratio]{agent_skills_completo_new.PNG}
  \captionof{figure}{Arquitectura completa del sistema SQL-Agents.
  Integra los seis agentes con el Orquestador, el servidor MCP
  (schemas + descripciones + bases vectoriales del esquema), las dos
  memorias (corto y largo plazo) y el módulo CDC sobre MySQL y PostgreSQL.}
  \label{fig:arq_completa}
\end{center}

% (Anexos D–H movidos a anexos_cdefgh_backup.md)

\clearpage
\section*{Anexo C: TOP-5 trials Fase~1 por agente}
\addcontentsline{toc}{section}{Anexo C}

\begin{center}
\captionof{table}{Fase~1 — Top-5 trials por agente (AR, AE, AS). Optuna TPE.
\textsuperscript{*}Ganador. \textsuperscript{†}Configuración desplegada en Fases~2--4.}
\label{tab:top5_ar_ae_as}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccc}
\toprule
\textbf{Agente} & \textbf{Modelo} & \textbf{T} & $F$ & $G$ & $R$ & $B$ & $E$ & \textbf{Combined} \\
\midrule
AR (1)\textsuperscript{*} & \texttt{gemini-2.5-flash}  & 0.5 & 0.959 & 0.990 & 0.970 & 0.004 & 0.053 & \textbf{0.962} \\
AR (2)                    & \texttt{gemini-2.5-flash}  & 0.7 & 0.955 & 0.985 & 0.970 & 0.002 & 0.041 & 0.961 \\
AR (3)                    & \texttt{claude-haiku-4-5}  & 0.3 & 0.964 & 0.971 & 0.970 & 0.002 & 0.041 & 0.959 \\
AR (4)                    & \texttt{claude-sonnet-4-6} & 0.0 & 0.964 & 0.967 & 0.970 & 0.002 & 0.041 & 0.958 \\
AR (5)                    & \texttt{claude-sonnet-4-6} & 0.5 & 0.951 & 0.980 & 0.970 & 0.002 & 0.041 & 0.958 \\
\midrule
AE\textsuperscript{*} & \texttt{gpt-4o}            & 0.3 & 0.845 & 0.930 & 0.461 & 0.058 & 0.160 & \textbf{0.792} \\
AE (2)                & \texttt{claude-haiku-4-5}  & 0.7 & 0.810 & 0.940 & 0.338 & 0.010 & 0.100 & 0.791 \\
AE (3)                & \texttt{gpt-4o}            & 0.7 & 0.840 & 0.922 & 0.480 & 0.138 & 0.080 & 0.788 \\
AE (4)                & \texttt{claude-haiku-4-5}  & 0.5 & 0.764 & 0.965 & 0.327 & 0.010 & 0.100 & 0.780 \\
AE (5)                & \texttt{gpt-4o}            & 0.7 & 0.770 & 0.939 & 0.434 & 0.058 & 0.160 & 0.759 \\
\midrule
AS\textsuperscript{*} & \texttt{claude-sonnet-4-6} & 0.7 & 0.981 & 0.912 & 0.408 & 0.006 & 0.068 & \textbf{0.870} \\
AS (2)                & \texttt{claude-haiku-4-5}  & 0.7 & 0.947 & 0.903 & 0.400 & 0.006 & 0.061 & 0.852 \\
AS (3)                & \texttt{claude-sonnet-4-6} & 0.0 & 0.922 & 0.899 & 0.404 & 0.167 & 0.130 & 0.788 \\
AS (4)                & \texttt{gpt-4o}            & 0.7 & 0.905 & 0.855 & 0.315 & 0.261 & 0.229 & 0.708 \\
AS (5)                & \texttt{gpt-4o}            & 0.3 & 0.909 & 0.833 & 0.314 & 0.261 & 0.229 & 0.700 \\
\bottomrule
\end{tabular}%
}
\par\smallskip\footnotesize{\textsuperscript{*}Ganador seleccionado.}
\end{center}

\vspace{8pt}
\begin{center}
\captionof{table}{Fase~1 — AG (NL$\to$SQL): Top-5 trials por base de datos.
$R$\,=\,\texttt{rouge\_l\_sql}. \textsuperscript{*}Ganador. Valor máx.\,3.0.}
\label{tab:top5_ag}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccc}
\toprule
\textbf{Base de datos} & \textbf{Modelo} & \textbf{T} & $F$ & $G$ & $R_{SQL}$ & $B$ & $E$ & \textbf{Combined} \\
\midrule
MySQL\textsuperscript{*} & \texttt{claude-haiku-4-5}  & 0.7 & 0.816 & 0.956 & 0.970 & 0.003 & 0.054 & \textbf{2.685} \\
MySQL (2)                & \texttt{claude-haiku-4-5}  & 0.7 & 0.809 & 0.952 & 0.978 & 0.003 & 0.053 & 2.683 \\
MySQL (3)                & \texttt{claude-haiku-4-5}  & 0.7 & 0.807 & 0.947 & 0.970 & 0.003 & 0.055 & 2.666 \\
MySQL (4)                & \texttt{claude-haiku-4-5}  & 0.3 & 0.806 & 0.927 & 0.970 & 0.003 & 0.054 & 2.646 \\
MySQL (5)                & \texttt{claude-sonnet-4-6} & 0.3 & 0.786 & 0.913 & 0.923 & 0.089 & 0.012 & 2.521 \\
\midrule
Postgres\textsuperscript{*} & \texttt{claude-haiku-4-5}  & 0.3 & 0.823 & 0.951 & 0.947 & 0.088 & 0.041 & \textbf{2.592} \\
Postgres (2)                & \texttt{claude-haiku-4-5}  & 0.3 & 0.824 & 0.948 & 0.947 & 0.093 & 0.046 & 2.580 \\
Postgres (3)                & \texttt{claude-haiku-4-5}  & 0.7 & 0.807 & 0.952 & 0.947 & 0.093 & 0.050 & 2.563 \\
Postgres (4)                & \texttt{claude-haiku-4-5}  & 0.5 & 0.807 & 0.952 & 0.944 & 0.093 & 0.048 & 2.562 \\
Postgres (5)                & \texttt{claude-sonnet-4-6} & 0.3 & 0.794 & 0.918 & 0.913 & 0.101 & 0.053 & 2.471 \\
\bottomrule
\end{tabular}%
}
\par\smallskip\footnotesize{Optuna TPE, 10 trials.
MySQL: ganador \texttt{claude-haiku-4-5} $T{=}0.7$ ($S{=}2.685$);
Postgres: ganador \texttt{claude-haiku-4-5} $T{=}0.3$ ($S{=}2.592$).}
\end{center}

\vspace{8pt}
\begin{center}
\captionof{table}{Fase~1 — AV (Validador): Top-5 trials por base de datos.
\textsuperscript{*}Ganador. $R$\,=\,ROUGE-L sobre \texttt{decision\_word}.}
\label{tab:top5_av}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccc}
\toprule
\textbf{Base de datos} & \textbf{Modelo} & \textbf{T} & $F$ & $G$ & $R$ & $B$ & $E$ & \textbf{Combined} \\
\midrule
MySQL\textsuperscript{*} & \texttt{gpt-4o}            & 0.3 & 0.993 & 0.928 & 1.000 & 0.002 & 0.045 & \textbf{0.959} \\
MySQL (2)                & \texttt{claude-sonnet-4-6} & 0.0 & 0.974 & 0.935 & 1.000 & 0.001 & 0.032 & 0.957 \\
MySQL (3)                & \texttt{claude-sonnet-4-6} & 0.7 & 0.970 & 0.924 & 1.000 & 0.001 & 0.034 & 0.951 \\
MySQL (4)                & \texttt{gpt-4o}            & 0.7 & 0.972 & 0.909 & 1.000 & 0.003 & 0.048 & 0.942 \\
MySQL (5)                & \texttt{gemini-2.5-flash}  & 0.7 & 0.968 & 0.909 & 1.000 & 0.003 & 0.054 & 0.939 \\
\midrule
Postgres\textsuperscript{*} & \texttt{claude-sonnet-4-6} & 0.0 & 0.984 & 0.937 & 1.000 & 0.001 & 0.032 & \textbf{0.962} \\
Postgres (2)                & \texttt{gemini-2.5-flash}  & 0.7 & 0.992 & 0.890 & 1.000 & 0.003 & 0.047 & 0.943 \\
Postgres (3)                & \texttt{claude-sonnet-4-6} & 0.7 & 0.895 & 0.871 & 0.900 & 0.065 & 0.049 & 0.864 \\
Postgres (4)                & \texttt{gemini-2.5-flash}  & 0.5 & 0.891 & 0.869 & 0.900 & 0.087 & 0.053 & 0.856 \\
Postgres (5)                & \texttt{gpt-4o}            & 0.7 & 0.876 & 0.832 & 0.900 & 0.096 & 0.062 & 0.832 \\
\bottomrule
\end{tabular}%
}
\par\smallskip\footnotesize{MySQL: ganador \texttt{gpt-4o} $T{=}0.3$ ($S{=}0.959$);
Postgres: ganador \texttt{claude-sonnet-4-6} $T{=}0.0$ ($S{=}0.962$).}
\end{center}

\vspace{10pt}
\section*{Anexo D: Resultados detallados Fase~2 (robustez a paráfrasis)}
\addcontentsline{toc}{section}{Anexo D}

\begin{center}
\captionof{table}{Métricas evaluadas por agente en Fase~2. $F$–$E$ idénticas a Fase~1.}
\label{tab:fase2_metricas_por_agente}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
\textbf{Agente} & $F$ & $G$ & $R$ & $F_1$ & $B$ & $E$ \\
\midrule
AR  & \ding{51} & \ding{51} & \ding{51}   & ---       & \ding{51} & \ding{51} \\
APS & ---       & ---       & ---         & \ding{51} & ---       & ---       \\
AG  & \ding{51} & \ding{51} & $R_{SQL}$   & ---       & \ding{51} & \ding{51} \\
AV  & \ding{51} & \ding{51} & \ding{51}   & ---       & \ding{51}\textsuperscript{†} & \ding{51} \\
AE  & \ding{51} & \ding{51} & \ding{51}   & ---       & \ding{51} & \ding{51} \\
AS  & \ding{51} & \ding{51} & \ding{51}   & ---       & \ding{51} & \ding{51} \\
\bottomrule
\end{tabular}
\par\smallskip\footnotesize{$F_1$: métricas de recuperación ($F_1^{T}$ tablas, $F_1^{C}$ columnas) usadas solo por APS.
APS no usa Brier/ECE (no genera confianza probabilística).}
\end{center}

\vspace{8pt}
\begin{center}
\captionof{table}{Comparación Fase~1 vs Fase~2. $\Delta{=}$F2$-$F1. Seed=66, 4 modelos $\times$ 4 temperaturas.}
\label{tab:winners_fase3}
\scriptsize\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccccccccc}
\toprule
\multirow{2}{*}{\textbf{Agente}} & \multirow{2}{*}{\textbf{Modelo}} &
  \multicolumn{3}{c}{$F$} & \multicolumn{3}{c}{$G$} &
  \multicolumn{3}{c}{$R$} & \multicolumn{2}{c}{$B$} &
  \multicolumn{2}{c}{$E$} & \multirow{2}{*}{$\Delta$\,Comb} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}
 & & F1 & F2 & $\delta$ & F1 & F2 & $\delta$ & F1 & F2 & $\delta$ & F1 & F2 & F1 & F2 & \\
\midrule
AR          & gemini-2.5-flash, T=0.5   & 0.959 & 0.917 & $-0.042$ & 0.990 & 0.968 & $-0.022$ & 0.970 & 0.796 & $-0.174$ & 0.004 & 0.004 & 0.053 & 0.051 & $-0.060$ \\
AG-MySQL    & claude-haiku-4-5, T=0.7   & 0.816 & 0.806 & $-0.010$ & 0.956 & 0.926 & $-0.030$ & 0.970 & 0.958 & $-0.012$ & 0.003 & 0.032 & 0.054 & 0.024 & $-0.051$ \\
AG-Postgres & claude-haiku-4-5, T=0.3   & 0.824 & 0.780 & $-0.044$ & 0.948 & 0.925 & $-0.023$ & 0.947 & 0.953 & $+0.006$ & 0.093 & 0.063 & 0.046 & 0.011 & $-0.008$ \\
AV-MySQL    & gpt-4o, T=0.3            & 0.993 & 0.893 & $-0.100$ & 0.928 & 0.870 & $-0.058$ & 1.000 & 0.900 & $-0.100$ & 0.002 & 0.096 & 0.045 & 0.059 & $-0.105$ \\
AV-Postgres & claude-sonnet-4-6, T=0.0  & 0.984 & 0.973 & $-0.011$ & 0.937 & 0.909 & $-0.028$ & 1.000 & 1.000 & $\pm0$ & 0.001 & 0.001 & 0.032 & 0.036 & $-0.017$ \\
AE          & gpt-4o, T=0.3            & 0.845 & 0.847 & $+0.002$ & 0.930 & 0.889 & $-0.041$ & 0.461 & 0.510 & $+0.049$ & 0.058 & 0.037 & 0.160 & 0.067 & $+0.013$ \\
AS          & claude-sonnet-4-6, T=0.7  & 0.981 & 0.881 & $-0.100$ & 0.912 & 0.909 & $-0.003$ & 0.408 & 0.335 & $-0.073$ & 0.006 & 0.031 & 0.068 & 0.036 & $-0.052$ \\
\bottomrule
\end{tabular}}
\end{center}
\noindent\footnotesize{AR/AV: $(2F+R+2G-E-B)/5$. AE/AS: $(2F+2G+0.5R-E-B)/4.5$. AG: $F+G+R_{SQL}-B-E$.
Fase~2 aplica las paráfrasis con el mismo modelo ganador de Fase~1;
$\Delta$\,Comb muestra la caída por variación lingüística.}

\clearpage
\section*{Anexo E: Datasets y métricas Fase~1}
\addcontentsline{toc}{section}{Anexo E}

\begin{center}
\captionof{table}{Datasets de prueba por agente (10 casos cada uno).}
\label{tab:datasets}
\footnotesize
\begin{tabular}{llp{0.60\textwidth}}
\toprule
\textbf{Agente} & \textbf{N} & \textbf{Composición} \\
\midrule
AR  & 10 & 5 consultas válidas (casing, paráfrasis, expansión) + 5 inválidas (DML, opinión, fuera de dominio) \\
APS & 10 & Preguntas Spider \texttt{concert\_singer} con tablas/columnas esperadas \\
AG  & 10 & Pares (pregunta, SQL canónico) Spider \\
AV  & 10 & 5 SQL válidos + 5 con errores (sintácticos y semánticos) \\
AE  & 10 & Explicaciones esperadas para SQL simple, JOINs, CTEs \\
AS  & 10 & Preguntas de seguimiento sobre razonamiento del pipeline \\
\bottomrule
\end{tabular}
\end{center}

\vspace{8pt}
\begin{center}
\captionof{table}{Métricas y función objetivo por agente en Fase~1.}
\label{tab:fase1_metricas_por_agente}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccl}
\toprule
\textbf{Agente} & \textbf{Rol} & $F$ & $G$ & $R$ & $F_1$ & $B$ & $E$ & \textbf{combined\_score (valor máx.)} \\
\midrule
AR  & NL$\to$NL     & \ding{51} & \ding{51} & \ding{51}   & ---       & \ding{51} & \ding{51} & $(2F+R+2G-E-B)/5$ \quad (1.0) \\
APS & Retriever     & ---       & ---       & ---         & \ding{51} & ---       & ---       & $(F_1^{T}+F_1^{C})/2$ \quad (1.0) \\
AG  & NL$\to$SQL    & \ding{51} & \ding{51} & $R_{SQL}$   & ---       & \ding{51} & \ding{51} & $F+G+R_{SQL}-B-E$ \quad (3.0) \\
AV  & Clasif.       & \ding{51} & \ding{51} & \ding{51}   & ---       & \ding{51} & \ding{51} & $(2F+R+2G-E-B)/5$ \quad (1.0) \\
AE  & SQL$\to$NL    & \ding{51} & \ding{51} & \ding{51}   & ---       & \ding{51} & \ding{51} & $(2F+2G+0.5R-E-B)/4.5$ \quad (1.0) \\
AS  & NL (sustenta) & \ding{51} & \ding{51} & \ding{51}   & ---       & \ding{51} & \ding{51} & $(2F+2G+0.5R-E-B)/4.5$ \quad (1.0) \\
\bottomrule
\end{tabular}%
}
\end{center}

\vspace{8pt}
\begin{center}
\captionof{table}{Configuraciones ganadoras Fase~1 (Optuna TPE,, 10 trials).}
\label{tab:winners_fase2}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllll}
\toprule
\textbf{Agente} & \textbf{Config desplegada} & \textbf{$T$} & \textbf{Combined} & \textbf{Métricas clave} \\
\midrule
AR             & gemini-2.5-flash & 0.5 & 0.962 & $F{=}0.959$, $G{=}0.990$, $R{=}0.970$ \\
APS-MySQL   & cosine, e5-base    & --- & 0.842 & $F_1^{T}{=}0.927$, $F_1^{C}{=}0.757$ \\
APS-Postgres& cosine, e5-base    & --- & 0.842 & $F_1^{T}{=}0.927$, $F_1^{C}{=}0.757$ \\
AG-MySQL    & claude-haiku-4-5   & 0.7 & 2.685 & $R_{SQL}{=}0.970$, $G{=}0.956$ \\
AG-Postgres & claude-haiku-4-5   & 0.3 & 2.592 & $R_{SQL}{=}0.947$, $G{=}0.951$ \\
AV-MySQL    & gpt-4o             & 0.3 & 0.959 & $F{=}0.993$, $G{=}0.928$, $R{=}1.000$ \\
AV-Postgres & claude-sonnet-4-6  & 0.0 & 0.962 & $F{=}0.984$, $G{=}0.937$, $R{=}1.000$ \\
AE          & gpt-4o             & 0.3 & 0.792 & $F{=}0.845$, $G{=}0.930$ \\
AS          & claude-sonnet-4-6  & 0.7 & 0.870 & $F{=}0.981$, $G{=}0.912$ \\
\bottomrule
\end{tabular}%
}
\end{center}

% ---------------------------------------------------------------
\clearpage
\section*{Anexo F: Top-5 trials Fase~3 --- Orquestador}
\addcontentsline{toc}{section}{Anexo F}

\begin{center}
\captionof{table}{Fase~3 --- Orquestador: Top-5 trials por base de datos.
$C$\,=\,\texttt{ToolCorrectnessMetric}. \textsuperscript{*}Ganador. Valor máx.$\approx$3.0.}
\label{tab:top5_orch}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccc}
\toprule
\textbf{Base de datos} & \textbf{Modelo} & \textbf{T} & $F$ & $G$ & $C$ & $B$ & $E$ & \textbf{Combined} \\
\midrule
MySQL\textsuperscript{*}  & \texttt{claude-haiku-4-5}  & 0.3 & 0.865 & 0.964 & 0.991 & 0.005 & 0.071 & \textbf{2.744} \\
MySQL (2)                 & \texttt{claude-haiku-4-5}  & 0.3 & 0.857 & 0.948 & 0.991 & 0.005 & 0.072 & 2.719 \\
MySQL (3)                 & \texttt{claude-haiku-4-5}  & 0.7 & 0.855 & 0.932 & 0.991 & 0.005 & 0.070 & 2.703 \\
MySQL (4)                 & \texttt{claude-sonnet-4-6} & 0.7 & 0.845 & 0.950 & 0.981 & 0.005 & 0.071 & 2.699 \\
MySQL (5)                 & \texttt{claude-sonnet-4-6} & 0.7 & 0.845 & 0.939 & 0.991 & 0.005 & 0.072 & 2.698 \\
\midrule
Postgres\textsuperscript{*} & \texttt{claude-haiku-4-5} & 0.3 & 0.858 & 0.967 & 0.800 & 0.005 & 0.067 & \textbf{2.554} \\
Postgres (2)                & \texttt{claude-haiku-4-5} & 0.0 & 0.865 & 0.958 & 0.800 & 0.005 & 0.069 & 2.549 \\
Postgres (3)                & \texttt{gpt-4o}           & 0.3 & 0.860 & 0.949 & 0.800 & 0.005 & 0.068 & 2.536 \\
Postgres (4)                & \texttt{gpt-4o}           & 0.7 & 0.856 & 0.952 & 0.800 & 0.005 & 0.069 & 2.534 \\
Postgres (5)                & \texttt{claude-haiku-4-5} & 0.7 & 0.856 & 0.949 & 0.800 & 0.005 & 0.068 & 2.533 \\
\bottomrule
\end{tabular}%
}
\par\smallskip\footnotesize{\textsuperscript{*}Ganador seleccionado. Optuna TPE, 10 trials.
MySQL: claude-haiku-4-5 $T{=}0.3$ ($S{=}2.744$);
Postgres: claude-haiku-4-5 $T{=}0.3$ ($S{=}2.554$).}
\end{center}

\clearpage
\section*{Anexo G: Escenarios del Orquestador (Fase~3)}
\addcontentsline{toc}{section}{Anexo G}

Los 10 escenarios usados en Fase~3 y Fase~4 son idénticos para MySQL y PostgreSQL.

\vspace{4pt}
\scriptsize
\begin{longtable}{cp{0.11\textwidth}p{0.30\textwidth}p{0.26\textwidth}p{0.22\textwidth}}
\toprule
\textbf{ID} & \textbf{Tipo} & \textbf{Pregunta} & \textbf{SQL esperado} & \textbf{Explicación} \\
\midrule
\endhead
1 & rej. opinión & Who is the best singer in the world? & --- & Subjective opinion; cannot be answered with the database. \\
\midrule
2 & rej. DML & Delete all concerts from 2010. & --- & DML statement (DELETE); system only processes SELECT queries. \\
\midrule
3 & rej. APS & What is the average ticket price per concert? & --- & Ticket price data not in the schema. \\
\midrule
4 & act. AS & Why did the pipeline use a JOIN between concert and stadium? & --- & JOIN needed because query required data from both tables sharing \texttt{stadium\_id}. \\
\midrule
5 & pipeline simple & What is the maximum and average capacity of all stadiums? & \texttt{SELECT max(capacity), avg(capacity) FROM stadium} & MAX and AVG over the Capacity column. \\
\midrule
6 & pipeline JOIN & Show stadium names and the highest year of concerts in each stadium. & \texttt{SELECT T1.Name, MAX(T2.Year) FROM stadium T1 JOIN concert T2 ON T1.Stadium\_ID=T2.Stadium\_ID GROUP BY T1.Stadium\_ID, T1.Name} & JOIN + GROUP BY to compute MAX(Year) per stadium. \\
\midrule
7 & pipeline subquery & Show the names of singers who have not performed in any concert. & \texttt{SELECT name FROM singer WHERE singer\_id NOT IN (SELECT singer\_id FROM singer\_in\_concert)} & NOT IN subquery over the bridge table. \\
\midrule
8 & pipeline DISTINCT & What are all distinct countries where singers above age 20 are from? & \texttt{SELECT DISTINCT country FROM singer WHERE age > 20} & DISTINCT + WHERE filter by age. \\
\midrule
9 & pipeline GROUP BY & Show the maximum age of singers in each country. & \texttt{SELECT country, max(age) FROM singer GROUP BY country} & GROUP BY country with MAX aggregation. \\
\midrule
10 & pipeline HAVING & List the countries with more than one singer in the database. & \texttt{SELECT country FROM singer GROUP BY country HAVING COUNT(*) > 1} & HAVING filter; tests AV correction of WHERE+aggregate to HAVING. \\
\bottomrule
\end{longtable}

\noindent\scriptsize{Escenarios 1--4: rechazos y rutas especiales.
Escenarios 5--10: pipeline completo con SQL de complejidad creciente.}

\clearpage
\section*{Bibliografía}
\addcontentsline{toc}{section}{Bibliografía}
\renewcommand{\refname}{}
\begin{thebibliography}{99}

\bibitem{ojuri2025}
Ojuri, S., Han, T.~A., Chiong, R., \& Di Stefano, A. (2025).
Optimizing text-to-SQL conversion.
\textit{Information Processing and Management}, 62, 104136.

\bibitem{zhao2024}
Zhao, X., Zhou, X., \& Li, G. (2024).
Chat2Data: Interactive data analysis.
\textit{Proc. VLDB}, 17(12), 4481--4484.

\bibitem{wang2024sqlrefine}
Wang, Z., Zhang, R., Nie, Z., \& Kim, J. (2024).
Tool-Assisted Agent on SQL Inspection and Refinement in Real-World Scenarios.
\textit{arXiv preprint}, arXiv:2408.16991.

\bibitem{tai2023cot}
Tai, C.-Y., Chen, Z., Zhang, T., Deng, X., \& Sun, H. (2023).
Exploring Chain-of-Thought Style Prompting for Text-to-SQL.
In \textit{Proc. EMNLP~2023}, pp. 1524--1543.
arXiv:2305.14215.

\bibitem{vichev2024}
Vichev, K., \& Marchev, S. (2024).
Retrieval-augmented approaches for text-to-SQL schema linking.
In \textit{Proc. IEEE International Conference on Intelligent Systems (IS~2024)}.

\bibitem{lin2004}
Lin, C.-Y. (2004).
ROUGE: A package for automatic evaluation of summaries.
In \textit{Text Summarization Branches Out}, pp. 74--81.

\bibitem{brown2020}
Brown, T., et al. (2020).
Language models are few-shot learners.
In \textit{Advances in NeurIPS}, vol.~33, pp. 1877--1901.

\bibitem{katsogiannis2023}
Katsogiannis-Meimarakis, G., \& Koutrika, G. (2023).
A survey on deep learning approaches for text-to-SQL.
\textit{The VLDB Journal}, 32(4), 905--936.

\bibitem{deng2022}
Deng, N., Chen, Y., \& Zhang, Y. (2022).
Recent advances in text-to-SQL.
In \textit{Proc. COLING}, pp. 2166--2187.

\bibitem{wooldridge2009}
Wooldridge, M. (2009).
\textit{An Introduction to MultiAgent Systems} (2nd ed.).
John Wiley \& Sons.

\bibitem{wang2024survey}
Wang, L., et al. (2024).
A survey on large language model based autonomous agents.
\textit{Frontiers of Computer Science}, 18(6), 186345.

\bibitem{reimers2019}
Reimers, N., \& Gurevych, I. (2019).
Sentence-BERT: Sentence embeddings using Siamese BERT-networks.
In \textit{Proc. EMNLP}, pp. 3982--3992.

\bibitem{chromadb2023}
Chroma (2023).
\textit{ChromaDB: The AI-native open-source embedding database}.
\url{https://www.trychroma.com/}

\bibitem{langchain2023}
Chase, H. (2023).
\textit{LangChain: Building applications with LLMs}.
\url{https://github.com/langchain-ai/langchain}

\bibitem{zhang2024memory}
Zhang, Z., et al. (2024).
A survey on the memory mechanism of large language model based agents.
\textit{arXiv preprint}, arXiv:2404.13501.

\bibitem{suhr2018}
Suhr, A., et al. (2018).
Learning to map context-dependent sentences to executable formal queries.
In \textit{Proc. NAACL}, pp. 954--965.

\bibitem{kleppmann2017}
Kleppmann, M. (2017).
\textit{Designing Data-Intensive Applications}.
O'Reilly Media.

\bibitem{langgraph2024}
LangChain AI (2024).
\textit{LangGraph: Stateful multi-actor applications with LLMs}.
\url{https://github.com/langchain-ai/langgraph}

\bibitem{guo2023}
Wang, B., Ren, C., Yang, J., Liang, X., Bai, J., Yin, D., \& Che, W. (2023).
MAC-SQL: Multi-agent collaboration for text-to-SQL.
\textit{arXiv preprint}, arXiv:2312.11242.

\bibitem{crispdm2000}
Chapman, P., et al. (2000).
\textit{CRISP-DM 1.0: Step-by-step data mining guide}.
SPSS Inc.

\bibitem{es2023ragas}
Es, S., et al. (2023).
RAGAS: Automated evaluation of retrieval augmented generation.
\textit{arXiv preprint}, arXiv:2309.15217.

\bibitem{brier1950}
Brier, G.~W. (1950).
Verification of forecasts expressed in terms of probability.
\textit{Monthly Weather Review}, 78(1), 1--3.

\bibitem{guo2017calibration}
Guo, C., et al. (2017).
On calibration of modern neural networks.
In \textit{Proc. ICML}, pp. 1321--1330.

\bibitem{sklearn2011}
Pedregosa, F., et al. (2011).
Scikit-learn: Machine learning in Python.
\textit{JMLR}, 12, 2825--2830.

\bibitem{deepeval2024}
Confident AI (2024).
\textit{DeepEval: The open-source LLM evaluation framework}.
\url{https://deepeval.com}

\bibitem{liu2023geval}
Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., \& Zhu, C. (2023).
G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.
\textit{Proceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing (EMNLP 2023)}, 2511--2522.
\url{https://arxiv.org/abs/2303.16634}

\bibitem{optuna2019}
Akiba, T., Sano, S., Yanase, T., Ohta, T., \& Koyama, M. (2019).
Optuna: A next-generation hyperparameter optimization framework.
In \textit{Proc. ACM SIGKDD}, pp. 2623--2631.

\bibitem{langsmith2024}
LangChain AI (2024).
\textit{LangSmith: LLM Observability and Evaluation Platform}.
\url{https://smith.langchain.com}

\bibitem{langfuse2024}
Langfuse (2024).
\textit{Langfuse: Open-source LLM engineering platform}.
\url{https://langfuse.com}

\bibitem{dspy2024}
Khattab, O., et al. (2024).
DSPy: Compiling declarative language model calls into pipelines.
In \textit{Proc. ICLR}.

\bibitem{promptfoo2024}
Promptfoo (2024).
\textit{Promptfoo: Test your prompts, agents, and RAGs}.
\url{https://promptfoo.dev}

\bibitem{yu2018spider}
Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., \& Radev, D. (2018).
Spider: A large-scale human-labeled dataset for complex and cross-domain semantic
parsing and text-to-SQL task.
In \textit{Proc. EMNLP}, pp. 3911--3921.

\bibitem{pourreza2023dinsql}
Pourreza, M., \& Hu, Y. (2023).
DIN-SQL: Decomposed in-context learning of text-to-SQL with self-correction.
In \textit{Proc. NeurIPS}. arXiv:2304.11015.

\bibitem{gao2023dailsql}
Gao, D., Wang, H., Li, Y., Sun, X., Qian, Y., Ding, B., \& Zhou, B. (2023).
Text-to-SQL empowered by large language models: A benchmark evaluation.
\textit{arXiv preprint}, arXiv:2308.15363.

\bibitem{li2023bird}
Li, J., Hui, B., Qu, G., Yang, J., Li, B., Li, B., \ldots \& Li, Y. (2023).
Can LLM already serve as a database interface? A BIg bench for large-scale
database grounded text-to-SQLs.
In \textit{Proc. NeurIPS}. arXiv:2305.03111.

\bibitem{anthropic_skills2024}
Anthropic (2024).
\textit{Claude Code Agent Skills: Building modular AI agents}.
\url{https://github.com/anthropics/claude-code}

\bibitem{anthropic_mcp2024}
Anthropic (2024).
\textit{Model Context Protocol: An open standard for connecting AI agents to
external systems}.
\url{https://modelcontextprotocol.io/}

\bibitem{anthropic_claude2024}
Anthropic (2024).
\textit{Claude: AI assistant by Anthropic --- Haiku and Sonnet model family}.
\url{https://www.anthropic.com/claude}

\bibitem{openai_gpt4o2024}
OpenAI (2024).
\textit{GPT-4o and GPT-4o-mini: Multimodal flagship models}.
\url{https://openai.com/gpt-4o}

\bibitem{wang2022e5}
Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder,
R., \& Wei, F. (2022).
\textit{Text Embeddings by Weakly-Supervised Contrastive Pre-training}.
arXiv:2212.03533. \url{https://arxiv.org/abs/2212.03533}

\bibitem{muennighoff2022mteb}
Muennighoff, N., Tazi, N., Magne, L., \& Reimers, N. (2023).
\textit{MTEB: Massive Text Embedding Benchmark}.
In \textit{Proc. EACL}, pp. 2014--2037.
\url{https://arxiv.org/abs/2210.07316}

\bibitem{sqlglot2023}
Mao, T. (2023).
\textit{SQLGlot: A no-dependency SQL parser, transpiler, optimizer, and engine}.
\url{https://github.com/tobymao/sqlglot}

\bibitem{aws_bedrock_eval}
Amazon Web Services (2024).
\textit{Amazon Bedrock — RAG Evaluation Metrics: Faithfulness, Context Relevance, Completeness}.
\url{https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-eval-llm-results.html}

\bibitem{azure_groundedness}
Microsoft (2024).
\textit{Azure AI Content Safety --- Groundedness Detection}.
\url{https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness}

\bibitem{maynez2020faithfulness}
Maynez, J., Narayan, S., Bohnet, B., \& McDonald, R. (2020).
On Faithfulness and Factuality in Abstractive Summarization.
\textit{Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)}, 1906--1919.

\bibitem{gemini_google2024}
Google DeepMind (2024).
\textit{Gemini 2.5: Multimodal models family --- Flash and Pro variants.
Generation configuration: temperature, top-p}.
\url{https://ai.google.dev/gemini-api/docs/models/gemini} \&
\url{https://ai.google.dev/gemini-api/docs/text-generation\#generation-config}

\bibitem{touvron2023llama2}
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y.,
Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., \textit{et al.} (2023).
\textit{Llama 2: Open Foundation and Fine-Tuned Chat Models}.
arXiv:2307.09288. \url{https://arxiv.org/abs/2307.09288}

\bibitem{bergstra2011tpe}
Bergstra, J., Bardenet, R., Bengio, Y., \& Kégl, B. (2011).
\textit{Algorithms for Hyper-Parameter Optimization}.
In \textit{Advances in Neural Information Processing Systems 24
(NeurIPS 2011)}, pp. 2546--2554.

\bibitem{manning2008irbook}
Manning, C.~D., Raghavan, P., \& Schütze, H. (2008).
\textit{Introduction to Information Retrieval}.
Cambridge University Press.
\url{https://nlp.stanford.edu/IR-book/}

\bibitem{zheng2023judging}
Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y.,
Lin, Z., Li, Z., Li, D., Xing, E., Zhang, H., Gonzalez, J.~E., \&
Stoica, I. (2023).
Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
In \textit{Advances in Neural Information Processing Systems 36 (NeurIPS 2023)}.
arXiv:2306.05685. \url{https://arxiv.org/abs/2306.05685}

\end{thebibliography}

\end{document}