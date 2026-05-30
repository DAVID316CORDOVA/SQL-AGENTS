"""
agents/APS/mysql/prompt.py

Prompts de APS para MySQL 8.0.
SEMANTIC_PROMPT/SEMANTIC_PROMPT_EN: generacion de descripciones semanticas
ricas para las tablas y columnas durante la indexacion en ChromaDB.
"""

SEMANTIC_PROMPT = """
=== CONTEXTO ===
Eres un experto en bases de datos MySQL. Tu trabajo es generar descripciones
semanticas para tablas y columnas, optimizadas para busqueda por similitud
vectorial. La calidad de las descripciones determina si el sistema encuentra
o no la columna correcta para una pregunta del usuario.

=== ESTRUCTURA OBLIGATORIA PARA COLUMNAS ===
Cada descripcion de columna debe escribirse en UN SOLO PARRAFO, en este orden:

1. EJEMPLOS DE VALORES CONCRETOS — al menos 5 ejemplos del tipo de dato real
   que esa columna almacenaria. Esto es lo MAS IMPORTANTE para discriminar
   en busqueda vectorial. Si es geografica, indica el nivel jerarquico
   (ciudad / pais / region / continente).
   REGLA CRITICA: usa SOLO nombres reales y reconocibles (Lima, Bogota,
   Madrid, Matematicas, Fisica). PROHIBIDO inventar nombres aleatorios
   o ficticios tipo Faker (East Yolandabury, Matthewton, Sergioland,
   etc.) aunque la BD este poblada con datos sinteticos.

2. SINONIMOS — al menos 5 alternativas, mezclando español e ingles.

3. EXCLUSIONES — lista 2-3 tipos de valor que esta columna NO almacena,
   especialmente otros niveles jerarquicos similares o entidades parecidas
   pero distintas (ciudades vs paises, materias vs profesores, etc.).

=== EJEMPLO DE DESCRIPCION CORRECTA ===
columna 'course_name' en una BD academica:
"Ejemplos de valores: Matematicas, Fisica, Quimica, Historia, Programacion,
Biologia, Calculo, Algebra Lineal. Sinonimos: materia, asignatura, curso,
ramo, course, subject, class. NO almacena nombres de profesores ni codigos
numericos ni departamentos academicos."

columna 'city' en una BD academica:
"Ejemplos de valores: Lima, Bogota, Buenos Aires, Madrid, Santiago, Caracas,
Quito. Nivel geografico: ciudad. Sinonimos: ciudad, localidad, city, town,
urbe. NO almacena paises, continentes, regiones administrativas ni
departamentos politicos."

columna 'max_students' en una BD academica:
"Ejemplos de valores: 20, 30, 40, 50, 100. Sinonimos: capacidad, cupo,
limite de inscripciones, capacity, enrollment limit. NO almacena nombres
ni IDs ni listas de estudiantes, solo numero entero entre 1 y 500."

=== ESTRUCTURA PARA TABLAS ===
Mas corta. Tres elementos:
1. Que entidad almacena.
2. Sinonimos de la entidad (3-5).
3. Tipos de pregunta que un usuario haria sobre esta tabla.

=== REGLAS ===
- Las descripciones de columna deben empezar SIEMPRE con "Ejemplos de valores:"
  para que la busqueda vectorial las encuentre por los valores tipicos.
- NO incluyas SQL, tipos MySQL, ni terminos tecnicos de base de datos.
- Si la columna es PK o FK, eso va al final como nota corta.

Schema MySQL:
{schema_text}

Responde SOLO JSON:
{{
  "tables": {{
    "nombre_tabla": "descripcion corta de la entidad"
  }},
  "columns": {{
    "tabla.columna": "Ejemplos de valores: ... Sinonimos: ... NO almacena: ..."
  }}
}}"""


# Version en INGLES del SEMANTIC_PROMPT, para datasets en ingles
# (Spider, BIRD, WikiSQL). Activado automaticamente cuando
# ACTIVE_DATASET comienza con 'spider:' o 'bird:'.
SEMANTIC_PROMPT_EN = """
=== CONTEXT ===
You are an expert in MySQL databases. Your task is to generate semantic
descriptions for tables and columns, optimized for vector similarity search.
The quality of these descriptions determines whether the system finds the
correct column for a user's question.

=== REQUIRED STRUCTURE FOR COLUMNS ===
Each column description must be written in ONE SINGLE PARAGRAPH, in this order:

1. CONCRETE VALUE EXAMPLES — at least 5 examples of the real data type the
   column would store. This is the MOST IMPORTANT signal for vector search.
   If geographic, indicate the hierarchy level (city / country / region /
   continent).
   CRITICAL RULE: use ONLY real and recognizable names (London, Tokyo,
   New York, Physics, Chemistry). FORBIDDEN: invented or Faker-style
   names (East Yolandabury, Matthewton, etc.) even if the DB has
   synthetic data.

2. SYNONYMS — at least 5 alternatives, mixing common English phrasings.

3. EXCLUSIONS — list 2-3 value types this column does NOT store,
   especially other similar hierarchy levels or close-but-different
   entities (cities vs countries, subjects vs teachers, etc.).

=== EXAMPLES OF CORRECT DESCRIPTIONS ===
column 'course_name' in an academic DB:
"Examples of values: Mathematics, Physics, Chemistry, History, Programming,
Biology, Calculus, Linear Algebra. Synonyms: subject, course, class,
discipline, module. Does NOT store teacher names, numeric codes, or
academic departments."

column 'city' in an academic DB:
"Examples of values: London, Tokyo, New York, Madrid, Paris, Berlin,
Sydney. Geographic level: city. Synonyms: city, town, locality,
metropolitan area, urban area. Does NOT store countries, continents,
administrative regions or political departments."

column 'max_students' in an academic DB:
"Examples of values: 20, 30, 40, 50, 100. Synonyms: capacity, enrollment
limit, maximum seats, course capacity, cap. Does NOT store names, IDs,
or student lists — only an integer between 1 and 500."

=== STRUCTURE FOR TABLES ===
Shorter. Three elements:
1. What entity it stores.
2. Synonyms of the entity (3-5).
3. Types of question a user would ask about this table.

=== RULES ===
- Column descriptions must ALWAYS start with "Examples of values:" so that
  vector search finds them by their typical values.
- DO NOT include SQL, MySQL types, or technical database terms.
- If the column is PK or FK, note it briefly at the end.

MySQL Schema:
{schema_text}

Respond ONLY with JSON:
{{
  "tables": {{
    "table_name": "short description of the entity"
  }},
  "columns": {{
    "table.column": "Examples of values: ... Synonyms: ... Does NOT store: ..."
  }}
}}"""
