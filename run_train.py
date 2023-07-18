''' 
Codigo para ejecutar un entrenamiento completo de una base de datos
'''


import json
import os

import numpy as np
import pandas as pd
import yaml

from pathlib import Path
from scipy import stats
from src.lib.class_load import LoadFiles
from src.lib.factory_data import HandleRedis, SQLDataSourceFactory, get_data
from src.lib.factory_models import ModelContext
from src.lib.factory_prepare_data import (DataCleaner, DataModel,
                                          MeanImputation, OutliersToIQRMean)
from src.models.args_data_model import Parameters

handler_load = LoadFiles()
handler_redis = HandleRedis()
ruta_actual = os.path.dirname(__file__)
# =================================================================
#             Cargar datos de la fuente de datos
# =================================================================

CONFIG_FILE = ruta_actual+'/src/data/config/config.yaml'
with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    parameters = yaml.safe_load(file)

parametros = Parameters(**parameters)

# Interacion para hacer un cache de los datos en redis
try:
    data = handler_redis.set_cache_data(
        hash_name=parametros.query_template['table'],
        old_dataframe=None,
        new_dataframe=None,
        exp_time=parametros.exp_time_cache,
        config=parametros.connection_data_source
    )
    # Verificar que existieran datos en cache
    if data is None:
        # #Peticion de la API
        # url  = 'http://192.168.115.99:3333/getinvoices'
        # response = requests.get(url)

        # if response.status_code == 200:
        #     invoices  = response.json()
        # else:
        #     print(response.status_code)

        # data = pd.DataFrame(invoices)
        # filter_cols = list(parameters['query_template']['columns'].values())
        # data = data[filter_cols]

        data = get_data(SQLDataSourceFactory(**parameters))

        data = handler_redis.set_cache_data(
            hash_name=parametros.query_template['table'],
            old_dataframe=data,
            new_dataframe=None,
            exp_time=parametros.exp_time_cache,
            config=parametros.connection_data_source
        )
except ValueError as error:
    print("[ERROR] No se puede hacer un cache de la fuente de datos")
# 910051 rows x 4 columns

print(data.shape)

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

# Patron de diseno de seleecio=n de estrategia
cleaner = DataCleaner(imputation)
data_imputation = cleaner.clean(data)
print("IMPUTACION DE DATOS")
print(data_imputation.dataframe)


MIN_DATA_VOLUME = 365
criterial = data_imputation.dataframe[parameters['filter_data']
                                      ['filter_1_column']].value_counts() > MIN_DATA_VOLUME
items = data_imputation.dataframe[parameters['filter_data']['filter_1_column']].value_counts()[
    criterial].index.to_list()

for item in items:
    parameters['filter_data']['filter_1_feature'] = item

    # =================================================================
    # Remocion de outliners y seleccion de columnas
    outliners = OutliersToIQRMean(**parameters)

    # Cambio de estrategia para remover outliners
    cleaner.strategy = outliners
    data_filled = cleaner.clean(data_imputation.dataframe)

    # =================================================================
    # Preparacion de los dato para el modelos escalizado y filtrado
    data_for_model = DataModel(**parameters)

    # Cambio de estrategia para preparar los datos para modelo
    cleaner.strategy = data_for_model
    data_ready, scaler_data = cleaner.clean(data_filled)
    #=================================================================
    if not parameters['scale']:
        data_ready = scaler_data.inverse_transform(data_ready)

    # =================================================================
    #            Preparacion de modelo
    # =================================================================

    MODE_USED = 'NBeatsModel'
    modelo = ModelContext(model_name = MODE_USED,
                          data=data_ready,
                          split=83,
                          **parameters)

    #Entrenar el modelo
    model_trained = modelo.train()

    #Optimizar los parametros del modelo
    if parameters['optimize']:
        model_trained = modelo.optimize()

    #Guargar los modelos entrenados
    modelo.save(model_trained,scaler=scaler_data)

    print('metodo finalizado')

