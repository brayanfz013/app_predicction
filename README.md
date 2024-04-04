

Descripción general

Modulo para syncbox encargado de hacer predicciones de series de tiempo en funcion de las configuración de un archivo yaml para toda una base de datos completa, se tienen parámetros generalizados para todos los modelos con la capacidad de afinar los parámetros del modelo

Listado de modelos pendiente para prediccion de datos

- Multi-horiazond Quantile Recurrente Forest MQRNN
- Deep space state model DSSM
- DeepAR

![Pasted image 20230621163227.png](doc/Pasted_image_20230621163227.png)

## Codigo

- [Data](doc/Data.md)
- [Features](doc/Features.md)
- [Lib](doc/Lib.md)
- [Models](doc/Models.md)

## Guias y Proceso

- [Descripcion General del código](doc/Descripcion%20General%20del%20codigo.md)
- [Funcionamiento](doc/Funcionamiento.md)
- [[Alertas]]

## Instalacion

- [Configuracion de python](doc/Configuracion%20de%20python.md)
- [Configuración de CUDA](doc/Configuracion%20de%20CUDA.md)
- [Configurar base de datos](doc/Configurar%20base%20de%20datos.md)

## To Do

- [ ]  Agregar un selecto de base de datos a los parámetros de configuración del archivo Yaml
- [x] Agregarle el sistema automatizado de los parámetros para lo modelos diferente a los Nbeats ✅ 2024-04-04
- [x]  Hacer un cache de los datos para que se van a procesar
- [x]  Hacer tabla de Varianza real contra estimada del modelo
- [x]  Métricas descriptivas, Gráficas estadísticas
- [x] Mejorar el sistema de log para todo el flujo de ejecución indicando mejor los errores ✅ 2023-08-14
- [x]  Hacer que las metricas como un método aparte interactivo que pueda se escalable , es decir aplicarle un patron de diseño
- [ ] Agregar sistemas de alertas para cada modelo
	- [x] Implementar los patrones de diseño de observer, que evalué la data para generar alertas en el sistema, tanto para productos que tengan o no modelo ✅ 2023-08-17
	- [ ] hacer el observador guarde los registro en Redis o en PostgreSQL
	- [x] implementar el patron de diseño para strategy ✅ 2023-08-17
	- [x] Aplicar los patrones a los datos ✅ 2023-08-23
	- [ ] Registrar las salidas en la base de datos
	- [x] Verificar que se cumpla según los criterios de un archivo de configuración esta para lo criterios de filtrado ✅ 2023-08-23
	- [x] agregar un método que evalúa la estacionariedad y la tendencia de los datos almacenados ✅ 2023-08-17
	- [ ] Agregar un metodo para verificar la fecha anterior del filtro
- [x] Preguntar a sergio por los tiempos de venta de alto y bajo, cuales son las cantidades bajas de venta y que criterio se deben usar para disparar las alertas ✅ 2023-08-23

