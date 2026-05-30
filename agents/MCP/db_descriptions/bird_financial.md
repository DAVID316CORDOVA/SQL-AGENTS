# bird_financial

## ARCHIVO DE DESCRIPCION DE LA BASE DE DATOS

Esta base de datos contiene los registros operativos de un banco real de la
Republica Checa correspondiente al periodo 1993-1998. Es uno de los datasets
incluidos en el benchmark BIRD para evaluacion de sistemas Text-to-SQL sobre
datos del mundo real.

La base almacena cuentas de clientes, prestamos otorgados, tarjetas de
credito emitidas, transacciones bancarias, ordenes de pago permanentes y
datos demograficos agregados por distrito. Permite responder preguntas
analiticas sobre comportamiento crediticio, distribucion geografica de
clientes, patrones de gasto y caracteristicas demograficas.

Los analistas del banco utilizan esta base para responder preguntas como:
- Que porcentaje de cuentas tiene un prestamo vigente.
- Cual es el monto promedio de los prestamos por distrito.
- Que clientes tienen tarjeta de credito y prestamo simultaneamente.
- Cual es el comportamiento de pago segun el estado del prestamo.

Caracteristicas relevantes del dominio:
- Sistema OLTP historico con datos congelados en el periodo 1993-1998.
- Las cuentas pueden tener multiples tarjetas y prestamos asociados.
- Los prestamos manejan estados codificados que indican si el contrato esta
  vigente o cerrado y si tiene deudas pendientes; la interpretacion exacta
  de cada codigo se documenta en el diccionario de datos.
- Los datos demograficos por distrito provienen de censos oficiales.
- Las fechas se almacenan en formato compactado segun la convencion de la
  epoca; el diccionario de datos especifica el formato exacto.
