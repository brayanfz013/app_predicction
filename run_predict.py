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
'''Codigo base para hacer predicciones sobre toda la base de datos  '''
# =============================================================================


import os
import re
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from src.lib.class_load import LoadFiles
# from src.features.features_fix_data import PrepareData
from src.lib.factory_data import SQLDataSourceFactory, get_data, create_table, set_data,HandleRedis
from src.lib.factory_models import ModelContext
from src.lib.factory_prepare_data import (DataCleaner, DataModel,
                                          MeanImputation, OutliersToIQRMean)
from src.models.DP_model import Modelos
from src.features.features_postgres import HandleDBpsql
from src.models.args_data_model import Parameters

path_folder = os.path.dirname(__file__)
folder_model = Path(path_folder).joinpath('scr/data/save_models')


handler_load = LoadFiles()
ruta_actual = os.path.dirname(__file__)
handler_redis = HandleRedis()
data_source = HandleDBpsql()
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
    #     #Peticion de la API
    #     url  = 'http://192.168.115.99:3333/getinvoices'
    #     response = requests.get(url)

    #     if response.status_code == 200:
    #         invoices  = response.json()
    #     else:
    #         print(response.status_code)

    #     data = pd.DataFrame(invoices)
    #     filter_cols = list(parameters['query_template']['columns'].values())
    #     data = data[filter_cols]

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

# =================================================================
#             Limpieza de datos
# =================================================================

new_types = []
base = {
    'date': np.datetime64,
    'integer': int,
    'float': float,
    'string': 'object',
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


# Imputacion de los datos
imputation = MeanImputation(
    replace_dtypes=new_types,
    strategy_imputation=strategy,
    preprocess_function=replace,
    **parameters
)

# Patron de diseno de seleecion de estrategia
cleaner = DataCleaner(imputation)
data_imputation = cleaner.clean(data)

MIN_DATA_VOLUME = 365
criterial = data_imputation.dataframe[parameters['filter_data']
                                      ['filter_1_column']].value_counts() > MIN_DATA_VOLUME
items = data_imputation.dataframe[parameters['filter_data']['filter_1_column']].value_counts()[
    criterial].index.to_list()

for item in items:

    try:
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
        # =================================================================
        if not parameters['scale']:
            data_ready = scaler_data.inverse_transform(data_ready)

        # =================================================================
        #            Preparacion de modelo
        # =================================================================

        MODE_USED = 'NBeatsModel'
        modelo = ModelContext(model_name=MODE_USED,
                            data=data_ready,
                            split=83,
                            **parameters)

        # Rutas de los parametros para predicciones
        model_train = modelo.save_path.joinpath(
            'model').with_suffix('.pt').as_posix()
        scaler = modelo.save_path.joinpath('scaler').with_suffix('.pkl').as_posix()
        last_pred = modelo.save_path.joinpath(
            'previus').with_suffix('.json').as_posix()
        parameters_model = modelo.save_path.joinpath(
            'parametros').with_suffix('.json').as_posix()

        # Cargar escaler
        scaler = handler_load.load_scaler(scaler)

        # Cargar modelo para hacer las predicciones
        MODE_USED = 'NBeatsModel'
        IntModel = Modelos[MODE_USED]
        model_trained = IntModel.load(model_train)

        pred_series = modelo.predict(
            model=model_trained,
            data=modelo.tunne_parameter.data,
            horizont=parameters['forecast_val']
        )

        # Invertir predicciones escaler de entrenamietno
        pred_scale = scaler.inverse_transform(pred_series)

        # Invertir Predicciones escaler de transformacion de los datos
        # pred_scale = scaler_data.inverse_transform(pred_series)

        data_frame_predicciones = pred_scale.pd_dataframe()
        column_field = list(data_frame_predicciones.columns)
        data_frame_predicciones.reset_index(inplace=True)
        data_frame_predicciones[parameters['filter_data']
                                ['predict_column']].clip(lower=0, inplace=True)
        # ===============================================================================================
        #                                         Metricas
        # ===============================================================================================
        # Esta parte tiene un ToDo importante: Tiene que ordenarse y optimizarce para se escalable
        # De momento funciona de manera estatica para ciertas cosas sobre todo el tema de la escritura 
        # en postgres, ademas de tener codigo copiado de funciones internas ya ordenadas
        

        # Cuantificar metricas de la columan de predicciones
        metric_columns_pred = data_imputation.metrics_column(
            data_frame_predicciones[parameters['filter_data']['predict_column']]
        )
        # Seleccion de columans para generar el dataframe de salida para la base de datos
        filter_temp = []
        for filter_list in parameters['filter_data']:
            if 'feature' in filter_list:
                filter_temp.append(parameters['filter_data'][filter_list])

        for adding_data in filter_temp:
            data_frame_predicciones[str(adding_data)] = adding_data

        new_names = list(parameters['query_template_write']['columns'].values())
        rename = {x: y for x, y in zip(
            list(data_frame_predicciones.columns), new_names)}
        data_frame_predicciones.rename(columns=rename, inplace=True)

        # Crear tabla para guardas la informacion
        create_table(SQLDataSourceFactory(**parameters))

        # Ingresar los datos a la base de datos
        set_data(SQLDataSourceFactory(**parameters), data_frame_predicciones)

        print(data_frame_predicciones)
        # ===============================================================================================
        #                            METRICAS
        # ===============================================================================================
        filter_columns = [column for column in parameters['filter_data'] if re.match(
            r'filter_\d+_column', column)]
        filter_feature = [column for column in parameters['filter_data'] if re.match(
            r'filter_\d+_feature', column)]

        value_product = []
        for i in filter_feature:
            value_product.append(parameters['filter_data'][i])
        fecha = parameters['query_template_write']['columns']['0']
        min_date = data_frame_predicciones[fecha].min()
        max_data = data_frame_predicciones[fecha].max()
        metric_columns_pred['init_date'] = data_frame_predicciones[fecha].min()
        metric_columns_pred['end_date'] = data_frame_predicciones[fecha].max()
        metric_columns_pred['product'] = '/'.join(value_product)

        type_data_out = {'Rango': 'float',
                        'Varianza': 'float',
                        'Desviacion_estandar': 'float',
                        'Coeficiente_varianza': 'float',
                        'Quantile Q1': 'float',
                        'Quantile Q3': 'float',
                        'InterQuantile': 'float',
                        'Desviacion_media_absoluta': 'float',
                        'init_date': 'date',
                        'end_date': 'date',
                        'product': 'string'
                        }

        fix_data_dict = {
            'table': 'metric_predict',
            'columns': {str(index): key for index, key in enumerate(type_data_out.keys())},
            'order': 'index',
            'where': 'posicion > 1'
        }

        parameters['query_template_write'] = fix_data_dict
        parameters['type_data_out'] = type_data_out

        create_table(SQLDataSourceFactory(**parameters))

        send_metrics = pd.DataFrame([metric_columns_pred])

        set_data(SQLDataSourceFactory(**parameters), send_metrics)

        # ===============================================================================================
        #                            ORIGINAL DATA
        # ===============================================================================================

        data_filled.reset_index(inplace=True)
        date_col = parameters['filter_data']['date_column']
        data_col = parameters['filter_data']['predict_column']

        # Filtrar data por tiempo
        filter_date = data_filled[(data_filled[date_col] >= metric_columns_pred['init_date']) & (
            data_filled[date_col] <= metric_columns_pred['end_date'])]
        original = data_imputation.metrics_column(filter_date[data_col])
        original['init_date'] = metric_columns_pred['init_date']
        original['end_date'] = metric_columns_pred['end_date']
        original['product'] = '/'.join(value_product)

        fix_data_dict = {
            'table': 'metric_data',
            'columns': {str(index): key for index, key in enumerate(type_data_out.keys())},
            'order': 'index',
            'where': 'posicion > 1'
        }

        parameters['query_template_write'] = fix_data_dict
        parameters['type_data_out'] = type_data_out

        create_table(SQLDataSourceFactory(**parameters))

        send_metrics = pd.DataFrame([original])

        set_data(SQLDataSourceFactory(**parameters), send_metrics)
        break
    except (ValueError,Exception) as error_predict:
        print("Error en las prediccioens")
        print(error_predict)
# import os
# import yaml
# import requests
# import numpy as np
# import pandas as pd
# from scipy import stats
# from pathlib import Path
# from src.lib.class_load import LoadFiles
# from src.features.features_fix_data import PrepareData
# from src.lib.factory_data import SQLDataSourceFactory, get_data,create_table, set_data
# from src.lib.factory_models import ModelContext
# from src.lib.factory_prepare_data import (DataCleaner, DataModel,
#                                           MeanImputation, OutliersToIQRMean)
# from src.models.DP_model import Modelos

# path_folder = os.path.dirname(__file__)
# folder_model = Path(path_folder).joinpath('scr/data/save_models')

# handler_load = LoadFiles()
# ruta_actual = os.path.dirname(__file__)

# # =================================================================
# #             Cargar datos de la fuente de datos
# # =================================================================
# CONFIG_FILE = ruta_actual+'/src/data/config/config.yaml'
# with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
#     parameters = yaml.safe_load(file)

# # #Peticion de la API
# # url  = 'http://192.168.115.99:3333/getinvoices'
# # response = requests.get(url)

# # if response.status_code == 200:
# #     invoices  = response.json()
# # else:
# #     print(response.status_code)

# # data = pd.DataFrame(invoices)
# # filter_cols = list(parameters['query_template']['columns'].values())
# # data = data[filter_cols]

# print("Probando el estacion de datos de sql")
# data = get_data(SQLDataSourceFactory(**parameters))

# # =================================================================
# #             Limpieza de datos
# # =================================================================

# new_types = []
# base = {
#     'date':np.datetime64,
#     'integer': int,
#     'float': float,
#     'string': object,
# }

# for dtypo in parameters['type_data'].values():
#     # print(dtypo)
#     new_types.append(base[dtypo])

# #metodo para transformar los tipo de datos
# strategy = {
#     int:np.mean,
#     float:np.mean,
#     object:stats.mode
# }

# #Estrategias para imputar los datos faltantes de NA
# replace = {
#     int:lambda x: int(float(x.replace(',',''))),
#     float:lambda x: float(x.replace(',','')),
#     object:lambda x: x.strip()
# }

# # Imputacion de los datos
# imputation = MeanImputation(
#     replace_dtypes=new_types,
#     strategy_imputation=strategy,
#     preprocess_function=replace,
#     **parameters
# )
# # Remocion de outliners y seleccion de columnas
# outliners = OutliersToIQRMean(**parameters)

# # Preparacion de los dato para el modelos escalizado y filtrado
# data_for_model = DataModel(**parameters)

# # Patron de diseno de seleecion de estrategia
# cleaner = DataCleaner(imputation)
# data_imputation = cleaner.clean(data)

# # Cambio de estrategia para remover outliners
# cleaner.strategy = outliners
# data_filled = cleaner.clean(data_imputation.dataframe)

# # Cambio de estrategia para preparar los datos para modelo
# cleaner.strategy = data_for_model
# data_ready, scaler_data = cleaner.clean(data_filled)

# # =================================================================
# #            Cargar modelo
# # =================================================================
# MODE_USED = 'NBeatsModel'
# modelo = ModelContext(model_name=MODE_USED,
#                       data=data_ready,
#                       split=83,
#                       **parameters)

# # Rutas de los parametros para predicciones
# model_train = modelo.save_path.joinpath('model').with_suffix('.pt').as_posix()
# scaler = modelo.save_path.joinpath('scaler').with_suffix('.pkl').as_posix()
# last_pred = modelo.save_path.joinpath('previus').with_suffix('.json').as_posix()
# parameters_model = modelo.save_path.joinpath('parametros').with_suffix('.json').as_posix()


# # Cargar escaler
# scaler = handler_load.load_scaler(scaler)

# # Cargar modelo para hacer las predicciones
# MODE_USED = 'NBeatsModel'
# IntModel = Modelos[MODE_USED]
# model_trained = IntModel.load(model_train)

# pred_series = modelo.predict(
#     model=model_trained,
#     data=modelo.tunne_parameter.data,
#     horizont=parameters['forecast_val']
# )

# # Invertir predicciones escaler de entrenamietno
# pred_scale = scaler.inverse_transform(pred_series)

# ## Invertir Predicciones escaler de transformacion de los datos
# # pred_scale = scaler_data.inverse_transform(pred_series)

# data_frame_predicciones = pred_scale.pd_dataframe()
# column_field = list(data_frame_predicciones.columns)
# # data_frame_predicciones['Varianza'] = data_frame_predicciones[column_field].pct_change() * 100
# data_frame_predicciones.reset_index(inplace=True)

# filter = []
# for filter_list in parameters['filter_data']:
#     if 'feature' in filter_list:
#         filter.append(parameters['filter_data'][filter_list])

# for adding_data  in filter:
#     data_frame_predicciones[str(adding_data)] = adding_data

# new_names = list(parameters['query_template_write']['columns'].values())
# rename= {x:y for x,y in zip(list(data_frame_predicciones.columns),new_names)}
# data_frame_predicciones.rename(columns=rename,inplace=True)

# # Crear tabla para guardas la informacion
# create_table(SQLDataSourceFactory(**parameters))

# set_data(SQLDataSourceFactory(**parameters),data_frame_predicciones)
