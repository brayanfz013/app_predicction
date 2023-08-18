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
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from src.lib.class_load import LoadFiles
from src.features.features_postgres import HandleDBpsql
from src.lib.factory_data import HandleRedis, SQLDataSourceFactory, get_data
from src.lib.factory_models import ModelContext
from src.lib.factory_prepare_data import (DataCleaner, DataModel,
                                          MeanImputation, OutliersToIQRMean)
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
    alerts_parameters = yaml.safe_load(file)

# parameter_alerts = AlertsData(**alerts_parameters)

# =================================================================
#             Cargar datos de la fuente de datos
# =================================================================
try:
    # Interacion para hacer un cache de los datos en redis
    logger.debug("No existen datos en cache")
    data = handler_redis.get_cache_data(
        hash_name=parametros.query_template['table'],
        config=parametros.connection_data_source
    )

    # Verificar que existieran datos en cache
    if data is None:
        #     #Peticion de la API
        #     url  = 'http://192.168.115.99:3333/getinvoices'
        #     response = requests.get(url)

        #     if response.status_code == 200:
        #         invoices  = response.json()
        #     else:
        #         print(response.status_code)

        #     data = pd.DataFrame(invoices)
        #     filter_cols = list(parameters['query_template']['columns'].values())AttributeError
        #     data = data[filter_cols]
        logger.debug('Extrayendo datos de la fuente de datos')
        data = get_data(SQLDataSourceFactory(**parameters))

        data = handler_redis.set_cache_data(
            hash_name=parametros.query_template['table'],
            old_dataframe=data,
            new_dataframe=None,
            exp_time=parametros.exp_time_cache,
            config=parametros.connection_data_source
        )
except Exception as error:
    logger.debug("No se puede hacer un cache de la fuente de datos")
    logger.error('Error')

# =================================================================
#             Limpieza de datos
# =================================================================
# Nuevos datos para reemplazar en las columnas
new_types = []
base = {
    'date': np.datetime64,
    'integer': int,
    'float': float,
    'string': object,
}

for dtypo in parameters['type_data'].values():
    # print(dtypo)
    new_types.append(base[dtypo])

# metodo para transformar los tipo de datos
strategy = {
    int: np.mean,
    float: np.mean,
    object: stats.mode
}

# Estrategias para imputar los datos faltantes de NA
replace = {
    int: lambda x: int(float(x.replace(',', ''))),
    float: lambda x: float(x.replace(',', '')),
    object: lambda x: x.strip()
}
# =================================================================
# Imputacion de los datos
imputation = MeanImputation(
    replace_dtypes=new_types,
    strategy_imputation=strategy,
    preprocess_function=replace,
    **parameters)

logger.debug('Realizando imputacion de datos')

# Patron de diseno de seleecio=n de estrategia
cleaner = DataCleaner(imputation)
data_imputation = cleaner.clean(data)
data = data_imputation
MIN_DATA_VOLUME = 365

# Segmento de filtrado de datos para simplificar la cantidad de modelos
# Filtrado por fecha reciente del ultimo ano
filter = data.dataframe[parameters['filter_data']
                        ['date_column']] >= pd.Timestamp('2023-01-01')
# Seleccion de los datos mas recientes en el dataframe
last_values_filter = data.dataframe[filter][parameters['filter_data']
                                            ['filter_1_column']].value_counts().index.to_list()
# Verificacion si las etiquetas estan en el dataframe general
newest_labels = data.dataframe[parameters['filter_data']['filter_1_column']].isin(
    last_values_filter)
# Subseleccionar los datos
data = data.dataframe[newest_labels]
# criterial = data.dataframe[filter][parameters['filter_data']['filter_1_column']].value_counts()>MIN_DATA_VOLUME
# Filtrar por datos que tenga mas de 365 registros
criterial = data[parameters['filter_data']
                 ['filter_1_column']].value_counts() > MIN_DATA_VOLUME
items = data[parameters['filter_data']['filter_1_column']].value_counts()[
    criterial].index.to_list()

for item in items:

    # =================================================================
    #             Cargar los parametros de alertas de modelos
    # =================================================================
    
    CONFIG_FILE = ruta_actual+'/src/data/config/config.yaml'
    with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
        parameters = yaml.safe_load(file)
    
    PARAM_ALARM = Path(ruta_actual).joinpath( 'src/data/alarm_files').joinpath(item).with_suffix('.yaml').as_posix()
    if not os.path.isfile(PARAM_ALARM):
        # PARAM_ALARM = Path(ruta_actual).joinpath( 'src/data/alarm_files').joinpath(item).with_suffix('.yaml').as_posix()
        handler_load.save_yaml(datafile=alerts_parameters,savepath=PARAM_ALARM)
    else:
        with open(PARAM_ALARM, 'r', encoding='utf-8') as file:
            alerts_parameters = yaml.safe_load(file)
    
    try:
        parameters['filter_data']['filter_1_feature'] = item
        date = parameters['filter_data']['date_column']
        filter_col = parameters['filter_data']['filter_1_column']
        quantity = parameters['filter_data']['predict_column']

        # data[date] = pd.to_datetime(data[date])
        
        # Filtrar el DataFrame para contener sólo el artículo que quieres evaluar
        item_data = data[data[filter_col] == item].copy()

        # Asegúrate de que los datos estén ordenados por fecha
        item_data.sort_values(date, inplace=True)

        # Define la serie temporal
        time_series = item_data.set_index(date)[quantity]

        #Dataframe para evaluar las alertas que son de todos los datos
        group_item = time_series.groupby([pd.Grouper(freq='W')]).sum()
        group_item = pd.DataFrame(group_item)

        #Dataframe con las metricas descriptivas para las alertas
        original = data_imputation.metrics_column(group_item[quantity])
        original['init_date'] = str(group_item.index.min())
        original['end_date'] = str(group_item.index.max())
        original['product'] = item
        df_metrics = pd.DataFrame([original])


    except (ValueError,Exception) as error_predict:
        print(error_predict)


# =================================================================
#           Secuencia de alarmas
# =================================================================
# Creacion de alertas para gene
alerta_baja_cantidad = AlertaPorBajaCantidad(
    alerts_parameters['alerta_bajacantidad']['cantidadbaja'])
observer_baja_cantidad = AlertaObserver(alerta_baja_cantidad)

alerta_tiempo_de_venta_bajo = AlertaPorTiempoDeVentaBajo(
    alerts_parameters['alerta_tiempodeventabajo']['min_dias'])
observer_tiempo_de_venta_bajo = AlertaObserver(alerta_tiempo_de_venta_bajo)

alerta_cambio_ventas = AlertaPorCambiosBruscosEnLasVentas(
    alerts_parameters['alerta_cambiosbruscos']['umbral_varianza'],
    alerts_parameters['alerta_cambiosbruscos']['umbral_desviacion'])
observer_cambio_ventas = AlertaObserver(alerta_cambio_ventas)

alerta_inventario_inactivo = AlertaPorInventarioInactivo(
    alerts_parameters['alerta_inventarioinactivo']['max_dias_inactivo'])
observer_inventario_inactivo = AlertaObserver(alerta_inventario_inactivo)

alerta_seguimiento_tendencias = AlertaPorSeguimientoTendencias(
    alerts_parameters['alerta_seguimientotendencias']['threshold'])
observer_seguimiento_tendencias = AlertaObserver(alerta_seguimiento_tendencias)

# Crear el inventario inicial
inventario_inicial = [
    {'modelo': 'manzana', 'cantidad': 100, 'init_date': '2022-01-01', 'end_date': '2022-01-15',
        'varianza': 12, 'desviacion_estandar': 12, 'coeficiente_varianza': 42},
    {'modelo': 'banana', 'cantidad': 10, 'init_date': '2022-01-01', 'end_date': '2022-01-05',
        'varianza': 43, 'desviacion_estandar': 13, 'coeficiente_varianza': 42}
]

# Crear una instancia de Inventario y adjuntar el observador
inventario = Inventario(inventario_inicial)
inventario.attach(observer_baja_cantidad)
inventario.attach(observer_tiempo_de_venta_bajo)
inventario.attach(observer_cambio_ventas)
inventario.attach(observer_inventario_inactivo)
inventario.attach(observer_seguimiento_tendencias)


# Agregar la alerta al inventario
inventario.agregar_alerta(alerta_baja_cantidad)
inventario.agregar_alerta(alerta_tiempo_de_venta_bajo)
inventario.agregar_alerta(alerta_cambio_ventas)
inventario.agregar_alerta(alerta_inventario_inactivo)
inventario.agregar_alerta(alerta_seguimiento_tendencias)

# Evaluar el inventario
inventario.evaluar_inventario()
