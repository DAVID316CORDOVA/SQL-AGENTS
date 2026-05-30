# bird_european_football_2

## ARCHIVO DE DESCRIPCION DE LA BASE DE DATOS

Esta base de datos contiene estadisticas detalladas del futbol profesional
europeo cubriendo los principales campeonatos nacionales (Inglaterra,
Espania, Alemania, Italia, Francia, Holanda, Portugal, Belgica, Polonia,
Escocia y Suiza) durante varias temporadas consecutivas. Es parte del
benchmark BIRD para evaluacion de sistemas Text-to-SQL sobre datos reales
de dominio deportivo.

La base almacena informacion de partidos, equipos, jugadores con sus
atributos fisicos y tecnicos, ligas, paises y atributos tacticos de los
equipos. Cada registro de jugador o equipo incluye decenas de metricas
(chance creation passing, defence aggression, build up play speed, entre
otras) que describen el estilo de juego y las capacidades.

Los analistas deportivos consultan esta base para responder preguntas como:
- Cual es la calificacion mas alta de un equipo en un atributo especifico.
- Que jugadores tienen mejor potencial en una posicion determinada.
- Como ha evolucionado el rendimiento de un equipo entre temporadas.
- Comparaciones entre clubes en estilo de juego ofensivo o defensivo.

Caracteristicas relevantes del dominio:
- Los nombres tecnicos de las metricas (chance creation passing, marking,
  pressure) son terminos del videojuego FIFA y reflejan atributos numericos
  del 1 al 100.
- Cada metrica numerica suele venir acompanada de una clasificacion textual
  que la categoriza como Slow, Balanced, Fast, etc.
- Los nombres de equipos y jugadores son los oficiales en su idioma
  original; por ejemplo Ajax, Bayern Munich, Real Madrid.
- Los atributos de equipo y jugador estan en tablas separadas y se
  relacionan por identificadores de API.
