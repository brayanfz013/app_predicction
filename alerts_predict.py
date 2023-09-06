# -*- coding: utf-8 -*-
# =============================================================================
__author__ = "Brayan Felipe Zapata "
__copyright__ = "Copyright 2007, The Cogent Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Brayan Felipe Zapata"
__email__ = "bzapata@smgsoftware.com"
__status__ = "Production"
# =============================================================================
'''
Codigo para aplicar criterios de filtrado a la base de datos, generando
alerta para lo diferentes modelos en un base de datos escribiendolos en
en postgres o redis
'''
# =============================================================================
import os
import yaml
import logging
import numpy as np
from scipy import stats
from pathlib import Path
from src.features.features_fix_data import ColumnsNameHandler
from src.lib.class_load import LoadFiles
from src.features.features_postgres import HandleDBpsql
from src.lib.factory_data import HandleRedis, SQLDataSourceFactory, get_data
from src.models.args_data_model import Parameters, AlertsData
from src.data.logs import LOGS_DIR
from src.features.filter_data import (AlertaPorBajaCantidad,
                                      AlertaPorTiempoDeVentaBajo,
                                      AlertaPorCambiosBruscosEnLasVentas,
                                      AlertaPorInventarioInactivo,
                                      #   AlertasPorVariacionPreciosProveedores,
                                      AlertaPorSeguimientoTendencias,
                                      AlertaPorDemandaEstacional,
                                      AlertaObserver,
                                      Inventario
                                      )


path_folder = os.path.dirname(__file__)
folder_model = Path(path_folder).joinpath('scr/data/save_models')

handler_load = LoadFiles()
ruta_actual = os.path.dirname(__file__)
handler_redis = HandleRedis()
data_source = HandleDBpsql()

# =================================================================
#             Configuracion Logger
# =================================================================
# Configura un logger personalizado en lugar de usar el logger raíz
logfile = ruta_actual+'/src/data/config/logging.conf'
logging.config.fileConfig(os.path.join(LOGS_DIR, logfile))
logger = logging.getLogger('filter')
logger.debug('Inciando secuencia de entrenamiento')

# =================================================================
#             Cargar los parametros
# =================================================================
CONFIG_FILE = ruta_actual+'/src/data/config/config.yaml'
with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    parameters = yaml.safe_load(file)

logger.debug("Probando el estacion de datos de sql")
parametros = Parameters(**parameters)
# data = get_data(SQLDataSourceFactory(**parameters))

CONFIG_FILE = ruta_actual+'/src/data/config/alerts.yaml'
with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    alerts_parameters_source = yaml.safe_load(file)

# =================================================================
#     Cargar los resultados de la base de datos de predicciones
# =================================================================

# Traer los datos de las predicciones
predictec_data = {'fecha': 'date',
                  'predicion': 'float',
                  'code': 'string'
                  }

parameters['query_template']['table'] = 'modelopredicciones'
parameters['query_template']['columns'] = {
    str(index): key for index, key in enumerate(predictec_data.keys())}
parameters['type_data'] = {
    'columns'+str(index): key for index, key in enumerate(predictec_data.values())}
parameters['filter_data']['date_column'] = 'fecha'
parameters['filter_data']['predict_column'] = 'predicion'
parameters['filter_data']['filter_1_column'] = 'code'

data = get_data(SQLDataSourceFactory(**parameters))

handler_data = ColumnsNameHandler(data, **parameters['query_template'])

# Traer metricas de las predicciones en la base de datos
metrics_data = {'rango': 'float',
                'varianza': 'float',
                'desviacion_estandar': 'float',
                'coeficiente_varianza': 'float',
                'quantile_q1': 'float',
                'quantile_q3': 'float',
                'interquantile': 'float',
                'desviacion_media_absoluta': 'float',
                'init_date': 'date',
                'end_date': 'date',
                'product': 'string'
                }

parameters['query_template']['table'] = 'metric_data'
parameters['query_template']['columns'] = {
    str(index): key for index, key in enumerate(metrics_data.keys())}
parameters['type_data'] = {
    'columns'+str(index): key for index, key in enumerate(metrics_data.values())}

data_metricas = get_data(SQLDataSourceFactory(**parameters))

# Clase para manipular y transformar los datos
handler_metricas = ColumnsNameHandler(
    data_metricas, **parameters['query_template'])


MODEL_FOLDERS = Path(ruta_actual).joinpath('src/data/save_models').as_posix()
items = []
for folder in Path(MODEL_FOLDERS).iterdir():
    if folder.is_dir() and folder.name != '__pycache__':
        items.append(folder.name)
items = items[0:5]


# =================================================================
#           Secuencia de alarmas
# =================================================================
for item in items:

    # Verificar si el archivo de configuracion de alarmas ya existe
    PARAM_ALARM = Path(ruta_actual).joinpath(
        'src/data/alarm_files').joinpath(item).with_suffix('.yaml').as_posix()
    print(PARAM_ALARM)
    if not os.path.isfile(PARAM_ALARM):
        handler_load.save_yaml(
            datafile=alerts_parameters_source, savepath=PARAM_ALARM)
        alerts_parameters = alerts_parameters_source
    else:
        with open(PARAM_ALARM, 'r', encoding='utf-8') as file:
            alerts_parameters = yaml.safe_load(file)

    # Traer metricas de las predicciones en la base de datos
    metrics_data = {'rango': 'float',
                    'varianza': 'float',
                    'desviacion_estandar': 'float',
                    'coeficiente_varianza': 'float',
                    'quantile_q1': 'float',
                    'quantile_q3': 'float',
                    'interquantile': 'float',
                    'desviacion_media_absoluta': 'float',
                    'init_date': 'date',
                    'end_date': 'date',
                    'product': 'string'
                    }

    parameters['query_template']['table'] = 'metric_data'
    parameters['query_template']['columns'] = {
        str(index): key for index, key in enumerate(metrics_data.keys())}
    parameters['type_data'] = {
        'columns'+str(index): key for index, key in enumerate(metrics_data.values())}
    parameters['filter_data']['filter_1_column'] = 'product'

    # Criterio de filtrado de informacion por item existente
    filter_item = handler_metricas.dataframe[parameters['filter_data']
                                             ['filter_1_column']] == item
    data_metricas = handler_metricas.dataframe[filter_item]

    # display(data_metricas)

    # Traer los datos de las predicciones
    predictec_data = {'fecha': 'date',
                      'predicion': 'float',
                      'code': 'string'
                      }
    parameters['query_template']['table'] = 'modelopredicciones'
    parameters['query_template']['columns'] = {
        str(index): key for index, key in enumerate(predictec_data.keys())}
    parameters['type_data'] = {
        'columns'+str(index): key for index, key in enumerate(predictec_data.values())}
    parameters['filter_data']['date_column'] = 'fecha'
    parameters['filter_data']['predict_column'] = 'predicion'
    parameters['filter_data']['filter_1_column'] = 'code'

    # Criterio de filtrado de informacion por item existente
    filter_item = handler_data.dataframe[parameters['filter_data']
                                         ['filter_1_column']] == item
    data = handler_data.dataframe[filter_item]

    # Creacion de alertas para gene
    alerta_baja_cantidad = AlertaPorBajaCantidad(
        # alerts_parameters['alerta_bajacantidad']['cantidadbaja'],
        cantidad=250,
        item=item,
        column='predicion',
        config=parametros.connection_data_source)
    observer_baja_cantidad = AlertaObserver(alerta_baja_cantidad)

    alerta_tiempo_de_venta_bajo = AlertaPorTiempoDeVentaBajo(
        # alerts_parameters['alerta_tiempodeventabajo']['min_dias'],
        min_dias_venta=10,
        item=item,
        config=parametros.connection_data_source)
    observer_tiempo_de_venta_bajo = AlertaObserver(alerta_tiempo_de_venta_bajo)

    alerta_cambio_ventas = AlertaPorCambiosBruscosEnLasVentas(
        umbral_varianza=alerts_parameters['alerta_cambiosbruscos']['umbral_varianza'],
        umbral_desviacion_estandar=alerts_parameters['alerta_cambiosbruscos']['umbral_desviacion'],
        item=item,
        config=parametros.connection_data_source)
    observer_cambio_ventas = AlertaObserver(alerta_cambio_ventas)

    alerta_inventario_inactivo = AlertaPorInventarioInactivo(
        max_dias_inactivo=alerts_parameters['alerta_inventarioinactivo']['max_dias_inactivo'],
        item=item,
        previus_val=alerts_parameters['alerta_inventarioinactivo']['valor_anterior_inventario'],
        config=parametros.connection_data_source)
    observer_inventario_inactivo = AlertaObserver(alerta_inventario_inactivo)

    alerta_seguimiento_tendencias = AlertaPorSeguimientoTendencias(
        threshold=alerts_parameters['alerta_seguimientotendencias']['threshold'],
        item=item,
        config=parametros.connection_data_source)
    observer_seguimiento_tendencias = AlertaObserver(
        alerta_seguimiento_tendencias)

    alerta_demanda_estacional = AlertaPorDemandaEstacional(
        item=item,
        threshold=alerts_parameters['alerta_demandaestacional']['threshold'],
        column_time='fecha',
        column_value='predicion',
        config=parametros.connection_data_source)
    observer_demanda_estacional = AlertaObserver(alerta_demanda_estacional)

    # Crear una instancia de Inventario y adjuntar el observador
    data.set_index('fecha', inplace=True)

    # Linea temporal para eliminar repetidos en la serie de tiempo
    data = data.reset_index().drop_duplicates(
        subset='fecha', keep='first').set_index('fecha')

    # Metodo para administrar alertas y observadores
    inventario_historic = Inventario(data)
    inventario_metricas = Inventario(data_metricas)

    inventario_historic.attach(observer_baja_cantidad)
    inventario_historic.attach(observer_inventario_inactivo)
    inventario_historic.attach(observer_demanda_estacional)
    inventario_historic.attach(observer_tiempo_de_venta_bajo)
    inventario_metricas.attach(observer_cambio_ventas)
    inventario_metricas.attach(observer_seguimiento_tendencias)

    # Evaluar el inventario
    inventario_historic.evaluar_historico(
        init_data=data.index[0], end_date=data.index[-1])
    inventario_metricas.evaluar_metricas()

    # Actualiza los parametros en caso de ser necesario
    alerts_parameters['alerta_inventarioinactivo']['valor_anterior_inventario'] = float(
        alerta_inventario_inactivo.previus_stock)
    handler_load.save_yaml(datafile=alerts_parameters, savepath=PARAM_ALARM)
