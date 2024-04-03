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
"""Codigo para ejecutar un entrenamiento completo de una base de datos """
# =============================================================================

import json
import os

import numpy as np
import pandas as pd
import yaml
import requests
from pathlib import Path
from scipy import stats
from src.lib.class_load import LoadFiles
from src.lib.factory_data import SQLDataSourceFactory, get_data
from src.features.features_redis import HandleRedis
from src.lib.factory_models import ModelContext
from src.lib.factory_prepare_data import (
    DataCleaner,
    DataModel,
    MeanImputation,
    OutliersToIQRMean,
)
from src.models.args_data_model import Parameters
from src.data.logs import LOGS_DIR
import logging

handler_load = LoadFiles()
handler_redis = HandleRedis()
ruta_actual = os.path.dirname(__file__)

# =================================================================
#             Configuracion Logger
# =================================================================
# Configura un logger personalizado en lugar de usar el logger raíz
logfile = ruta_actual + "/src/data/config/logging.conf"
logging.config.fileConfig(os.path.join(LOGS_DIR, logfile))
logger = logging.getLogger("train")
logger.debug("Inciando secuencia de entrenamiento")

# =================================================================
#             Cargar los parametros
# =================================================================
CONFIG_FILE = ruta_actual + "/src/data/config/config.yaml"
with open(CONFIG_FILE, "r", encoding="utf-8") as file:
    parameters = yaml.safe_load(file)

logger.debug("Archivo de configuraciones cargado")
parametros = Parameters(**parameters)

# =================================================================
#             Cargar datos de la fuente de datos
# =================================================================
# Interacion para hacer un cache de los datos en redis
try:
    logger.debug("verficando si existe data cache")
    data = handler_redis.get_cache_data(
        hash_name=parametros.query_template["table"],
        config=parametros.connection_data_source,
    )
    # Condicional para actualizar datos en caso de existan datos en redis
    if data is not None:
        logger.debug("Existe data en cache")

        # Secuencia de codigo para perdir nuevos datos a la base de datos
        date_col_query = parameters["query_template"]["columns"]["0"]
        LAST_DAY = str(data.iloc[-1][0])
        parameters["query_template"]["where"] = f" \"{date_col_query}\" > '{LAST_DAY}'"
        parameters["query_template"]["order"] = "".join(
            ['"' + columna + '"' for columna in [date_col_query]]
        )

        logger.debug("Realizando peticion a la fuente de datos")
        # #Peticion de la API
        # url  = 'http://192.168.115.99:3333/getinvoices'
        # response = requests.get(url)

        # if response.status_code == 200:
        #     invoices  = response.json()
        # else:
        #     logger.debug(response.status_code)
        # logger.debug('Generando Union de datos')
        # new = pd.DataFrame(invoices)
        # filter_cols = list(parameters['query_template']['columns'].values())
        # new = new[filter_cols]

        # Extraccion de la nueva data para actualizar
        new = get_data(SQLDataSourceFactory(**parameters))
        logger.debug("Actualizando cache en redis")
        data = handler_redis.set_cache_data(
            hash_name=parametros.query_template["table"],
            old_dataframe=data,
            new_dataframe=new,
            exp_time=parametros.exp_time_cache,
            config=parametros.connection_data_source,
        )
        logger.debug("Actualizacion completa de datos en redis")

    # Verificar que existieran datos en cache
    if data is None:
        logger.debug("No existe cache de datos")
        # #Peticion de la API
        # url  = 'http://192.168.115.99:3333/getinvoices'
        # response = requests.get(url)

        # if response.status_code == 200:
        #     invoices  = response.json()
        # else:
        #     logger.debug(response.status_code)
        # logger.debug('Generando Dataframe de datos')

        # data = pd.DataFrame(invoices)
        # filter_cols = list(parameters['query_template']['columns'].values())
        # data = data[filter_cols]
        data = get_data(SQLDataSourceFactory(**parameters))
        logger.debug("Insertando datos de cache en redis")
        data = handler_redis.set_cache_data(
            hash_name=parametros.query_template["table"],
            old_dataframe=data,
            new_dataframe=None,
            exp_time=parametros.exp_time_cache,
            config=parametros.connection_data_source,
        )

except ValueError as error:
    logger.debug("[ERROR] No se puede hacer un cache de la fuente de datos")
    logger.debug(error)
    exit()


# =================================================================
#             Limpieza de datos
# =================================================================
# Nuevos datos para reemplazar en las columnas
new_types = []
base = {
    "date": np.datetime64,
    "integer": int,
    "float": float,
    "string": object,
}

for dtypo in parameters["type_data"].values():
    # print(dtypo)
    new_types.append(base[dtypo])

# metodo para transformar los tipo de datos
strategy = {int: np.mean, float: np.mean, object: stats.mode}

# Estrategias para imputar los datos faltantes de NA
replace = {
    int: lambda x: int(float(x.replace(",", ""))),
    float: lambda x: float(x.replace(",", "")),
    object: lambda x: x.strip(),
}
# =================================================================
# Imputacion de los datos
imputation = MeanImputation(
    replace_dtypes=new_types,
    strategy_imputation=strategy,
    preprocess_function=replace,
    **parameters,
)

logger.debug("Realizando imputacion de datos")

# Patron de diseno de seleecion de estrategia
cleaner = DataCleaner(imputation)
data_imputation = cleaner.clean(data)

MIN_DATA_VOLUME = 365
logger.debug("Filtrando informacion")
# Segmento de filtrado de datos para simplificar la cantidad de modelos
# Filtrado por fecha reciente del ultimo ano
filter = data_imputation.dataframe[parameters["filter_data"]["date_column"]] >= pd.Timestamp(
    "2000-01-01"
)
# Seleccion de los datos mas recientes en el dataframe
last_values_filter = (
    data_imputation.dataframe[filter][parameters["filter_data"]["filter_1_column"]]
    .value_counts()
    .index.to_list()
)
# Verificacion si las etiquetas estan en el dataframe general
newest_labels = data_imputation.dataframe[parameters["filter_data"]["filter_1_column"]].isin(
    last_values_filter
)
# Subseleccionar los datos
selected_data = data_imputation.dataframe[newest_labels]

# Filtrar por datos que tenga mas de 365 registros
criterial = (
    selected_data[parameters["filter_data"]["filter_1_column"]].value_counts() > MIN_DATA_VOLUME
)
items = (
    selected_data[parameters["filter_data"]["filter_1_column"]]
    .value_counts()[criterial]
    .index.to_list()
)

items = items[0:5]
logger.debug("Filtrado terminado")

for item in items:
    logger.debug("Iniciando entrenamiento de : %s", item)
    try:
        parameters["filter_data"]["filter_1_feature"] = item
        logger.debug("removiendo outliners : %s", item)

        # =================================================================
        # Remocion de outliners y seleccion de columnas
        # Cambio de estrategia para remover outliners
        cleaner.strategy = OutliersToIQRMean(**parameters)
        data_filled = cleaner.clean(data_imputation.dataframe)
        logger.debug("Preparando datos para el modelo : %s", item)

        # =================================================================
        # Preparacion de los dato para el modelos escalizado y filtrado
        data_for_model = DataModel(**parameters)

        # Cambio de estrategia para preparar los datos para modexlo
        cleaner.strategy = data_for_model
        data_ready, scaler_data = cleaner.clean(data_filled)

        # =================================================================
        if not parameters["scale"]:
            data_ready = scaler_data.inverse_transform(data_ready)

        # =================================================================
        #            Preparacion de modelo
        # =================================================================
        MODE_USED = "NBeatsModel"
        modelo = ModelContext(model_name=MODE_USED, data=data_ready, split=83, **parameters)
        logger.debug("Entrenando modelo : %s", item)
        # Entrenar el modelo
        model_trained = modelo.train()

        # Optimizar los parametros del modelo
        if parameters["optimize"]:
            logger.debug("Optimizando entrenamiento de : %s", item)
            model_trained = modelo.optimize()

        # Guargar los modelos entrenados
        modelo.save(model_trained, scaler=scaler_data)

        logger.debug("Finalizando entrenamiento de : %s", item)

    except (Exception, ValueError) as error_train:
        logger.error(error_train)
