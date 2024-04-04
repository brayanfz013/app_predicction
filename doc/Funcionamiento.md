---
link: https://www.notion.so/Funcionamiento-aabd852f735b4fd09cb81f513c4efb4c
notionID: aabd852f-735b-4fd0-9cb8-1f513c4efb4c
---
# Funcionamiento

## ENTRADAS:

El sistema se comportar de diferentes maneras en función de la ejecución del los códigos teniendo diferentes salidas todo funciona en base a un archivo YAML el cual contiene los campos que se encargan de configurar la ejecución del codigo:

### CONFIG.YAML:

```yaml
logs_file: Ruta del archivo donde se almacenan los logs

data_source: Segmento de seleecion de la fuentes de datos para el proceso
  input: Escritura de fuente de entrada redis/postgres/sqlserver/...
  cache: Escritura de fuente encargada de hacer el cache /redis/...
  output: Escritura de fuente de salida postgres/sqlserver/....

connection_data_source: Parametros de conexion para la fuente de data_source
	que son nombrados
  postgresql:
    host: Seleccion de host /localhost
    database: Nombre de la base de datos /predicciones
    user:  Usuario de la base de datos /bdebian
    password: Contrasena del usuario de la base de datos0913..
    port: Puerto a usar5432
  # engine: psycopg2

  redis: parametros de configuracion de redis
    host: 127.0.0.1
    db: 1
    port: 6379
    # password: 0913

  sqlserver: parametros de configuracion de sqlserver
    server: 127.0.0.1
    database: master
    username: sa
    password: 0913PASSword!

query_template: Formato de las peticiones de la query a la base de datos
  table: custom_invoices Nombre de la tabla a usar 
  columns: Nombre de las columnas que se quieren extraer
    "0": CREATDDT
    "1": ITEMNMBR
    "2": QUANTITY
    "3": CUSTNMBR
		.
		.
		.
  order: #index seleccion que tipo de orden asc o des
  where: #posicion > 1 Condicional para filtrar los datos al momento de la query

type_data: Tipo datos usados en la base de datos para exportar al sistema
	se tiene que enumerar los demas campos con column#: tipo de dato
  column0: date
  column1: string
  column2: integer
  column3: string
		.
		.
		.
	columnN: tipo de dato

query_template_write: Formato para hacer una query de salida que va al ouput del data_soucer
  table: modelopredicciones
  columns: 
    "0": fecha
    "1": predicion
    "2": real
    "3": CUSTNMBR
		.
		.
		.
  order: #index seleccion que tipo de orden asc o des
  where: #posicion > 1 Condicional para filtrar los datos al momento de la query

type_data_out: Tipo de datos de salida para los datos de output del data_soucer
	se tiene  que colocar el nombre de las columnas del query_template_write
  fecha: date
  predicion: float
  real: float
  CUSTNMBR: string

filter_data: Sistema de filtrado de los datos importados por nombre de las columanas
	usadas en query_template
  date_column: CREATDDT Nombre de la columna que contiene el campo de la fechas
  predict_column: QUANTITY Nombre de la columna la cual se quiere predecir
  filter_1_column: ITEMNMBR Nombre de la columna por la cual se va a filtrar para hacer
	las predicciones
  filter_1_feature: SME4-90 Caracteristica o elemento de la columna sobre la cual se 
	hacer el filtrado de los datos 
  # filter_2_column: CUSTNMBR
  # filter_2_feature: F00575
	.
	.
	.
  # filter_n_column: columna n
  # filter_n_feature: feature n
  group_frequecy: W # Frecuencia 

scale: True #Escalar los datos para las predicciones
first_train: True # Muestra para saber si es primer entrenamiento del modelo
optimize: False # Parametro para usar optimizacion de modelo 
forecast_val: 4 #Tiempo de 
exp_time_cache: 8600000
```

Los tipo de datos usados predeterminado en el código son :

```python
{
	fechas: 'date'
    enteros: 'integer'
    reales: 'float'
    caracteres/strings: 'string'
}
```

### NOTA:

- Se debe de tener cuidado con el parametro optimize en el momento del entrenamieto dado que si son muchos modelos esto puede tomar hasta mas 8 horas por modelo

Se tiene codigo para genera una o mulitples predicciones a lo cual se puede ejecutar de la siguente  manera 

**run_train.py:** Entrenamiento para una columan completa de una base de datos seleccionanda en el archivo YAML en la parte de **********************************************************filter_data: filter_1_column********************************************************** Esto ignora la parte completa de ********************************filter_1_feature********************************  y toma todos los elemento de la columna y hace un modelo para cada uno de lo elementos

**run_predict.py :** Predicciones para una columan completa de una base de datos seleccionanda en el archivo YAML en la parte de **********************************************************filter_data: filter_1_column********************************************************** Esto ignora la parte completa de ********************************filter_1_feature********************************  y toma todos los elemento de la columna y hace un modelo para cada uno de lo elementos

**train.py:** Codigo para entrenar un modelo unico para una o varias series de caracteristicas de **********************filter_data**********************

**predict.py:** Codigo para ejecutar un modelo unico para una o varias series de caracteristicas de **********************filter_data**********************

## SALIDAS:

El codigo genera una serie de tablas para los datos de predicciones como para los datos con los que hacen las predicciones generando metricas para evaluar la condicion de los resultados

Los resultados se generan en donde se escojan los parametros de **data_source** en ***output*** en el archivo de configuraciones del YAML, 

La elección de la métrica para medir la variación en tus datos depende en gran medida de tus necesidades y del contexto de tus datos. Aquí hay algunas métricas comunes y cuándo podrían ser útiles:

1. **Rango**: Esta es la diferencia entre el valor máximo y el mínimo de tus datos. Es útil si solo necesitas una idea rápida de la propagación total de tus datos, pero puede ser engañosa si tus datos tienen valores atípicos.
2. **Varianza**: Esta es la media de los cuadrados de las diferencias entre cada dato y la media. La varianza puede ser útil si necesitas una medida que tenga en cuenta todos los datos y da más peso a las grandes desviaciones de la media. Sin embargo, está en unidades cuadradas, lo que puede ser difícil de interpretar.
3. **Desviación estándar**: Esta es la raíz cuadrada de la varianza. Tiene la ventaja de estar en las mismas unidades que tus datos, lo que a menudo la hace más fácil de interpretar que la varianza.
4. **Coeficiente de variación**: Este es la desviación estándar dividida por la media, a menudo expresada como porcentaje. Es útil si estás comparando la variación entre diferentes conjuntos de datos que tienen diferentes unidades o rangos.
5. **Cuartiles y rango intercuartílico (IQR)**: Los cuartiles dividen tus datos en cuatro partes iguales, y el IQR es la diferencia entre el tercer cuartil (Q3) y el primer cuartil (Q1). Estas medidas son útiles si deseas una imagen más detallada de la distribución de tus datos y son resistentes a los valores atípicos.
6. **Desviaciones absolutas medias (MAD)**: Es la media de las diferencias absolutas entre cada dato y la media. Al igual que la desviación estándar y la varianza, mide la dispersión en los datos. Pero a diferencia de esas medidas, no da un peso adicional a las desviaciones grandes.

## Alertas 

Descripción de estados de alertas:

```yaml
{ 
     Nombre del modelo :
		{
			Nombre de alerta: indicador de aleta
		}
}
```
 **indicadores de alerta**:

	0: Alerta desactivada
	1: Alerta activa
	2: Indicativo creciente
	3: Indicativo Decreciente
	4: Datos insuficientes
