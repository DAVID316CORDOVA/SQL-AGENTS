# demo_db

## ARCHIVO DE DESCRIPCION DE LA BASE DE DATOS

Esta base de datos representa el sistema de gestion academica de una
universidad. Almacena la informacion de las carreras ofrecidas, los
estudiantes inscritos, los profesores que dictan asignaturas, las materias
del plan de estudios, las matriculas por periodo academico y las
calificaciones obtenidas por cada estudiante.

El proposito principal del sistema es soportar las operaciones diarias del
area academica: registrar matriculas semestrales, asignar profesores a
materias, llevar control de notas finales y consultar el historial academico
de los estudiantes.

Los usuarios tipicos son:
- Coordinadores de carrera, que consultan rendimiento, asistencia y
  promedios por programa.
- Profesores, que revisan las notas de sus estudiantes y los listados de
  cursos asignados.
- Administracion academica, que produce reportes globales por carrera,
  estadisticas de matricula y proyecciones por periodo.

Preguntas frecuentes que los usuarios formulan en lenguaje natural:
- Cuantos estudiantes hay matriculados en cada carrera.
- Que profesores dictan una materia especifica.
- Cual es el promedio de notas de un curso o de un estudiante.
- Listar los estudiantes inscritos en una materia determinada.
- Que materias requieren prerrequisitos.

Caracteristicas relevantes del dominio:
- Sistema OLTP transaccional, datos actualizados en tiempo real.
- Modelo relacional con entidades principales: estudiantes, profesores,
  materias, carreras, matriculas y calificaciones.
- Las calificaciones siguen una escala numerica de 0 a 5.
- Cada estudiante tiene un identificador unico y puede estar matriculado en
  varias materias por periodo.
- Las relaciones principales son: estudiante - matricula - materia -
  profesor; carrera - materia.
