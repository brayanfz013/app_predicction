'''Codigo para la ejecucion  y llamando para hacer predicciones en los modelos'''


import json
import os

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.lib.class_load import LoadFiles
from src.lib.factory_data import SQLDataSourceFactory, get_data
from src.lib.factory_models import ModelContext
from src.lib.factory_prepare_data import (DataCleaner, DataModel,
                                          MeanImputation, OutliersToIQRMean)

# from src.models.DP_model import Modelos

handler_load = LoadFiles()

ruta_actual = os.path.dirname(__file__)
print(ruta_actual)
#=================================================================
#             Cargar datos de la fuente de datos 
#=================================================================

CONFIG_FILE = ruta_actual+'/src/data/config/config.yaml'
with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    parameters = yaml.safe_load(file)

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


print("Probando el estacion de datos de sql")
data = get_data(SQLDataSourceFactory(**parameters))

#=================================================================
#             Limpieza de datos
#=================================================================
# Nuevos datos para reemplazar en las columnas
new_types = []
base = {
    'date':np.datetime64,
    'integer': int,
    'float': float,
    'string': object,
}

for dtypo in parameters['type_data'].values():
    # print(dtypo)
    new_types.append(base[dtypo])

#metodo para transformar los tipo de datos
strategy = {
    int:np.mean,
    float:np.mean,
    object:stats.mode
}

#Estrategias para imputar los datos faltantes de NA
replace = {
    int:lambda x: int(float(x.replace(',',''))),
    float:lambda x: float(x.replace(',','')),
    object:lambda x: x.strip()
}
print(data)

#Imputacion de los datos
imputation = MeanImputation(
                            replace_dtypes=new_types,
                            strategy_imputation=strategy,
                            preprocess_function=replace,
                            **parameters
                            )
#Remocion de outliners y seleccion de columnas
outliners = OutliersToIQRMean(**parameters)

#Preparacion de los dato para el modelos escalizado y filtrado
data_for_model = DataModel(**parameters)

#Patron de diseno de seleecion de estrategia
cleaner = DataCleaner(imputation)
data_imputation = cleaner.clean(data)

#Cambio de estrategia para remover outliners
cleaner.strategy = outliners
data_filled = cleaner.clean(data_imputation.dataframe)

# Cambio de estrategia para preparar los datos para modelo
cleaner.strategy = data_for_model
data_ready,scaler_data = cleaner.clean(data_filled)

if not parameters['scale']:
    data_ready = scaler_data.inverse_transform(data_ready)


# =================================================================
#            Preparacion de modelo
# =================================================================
# model_names = list(Modelos.keys())

# for name in model_names:
#     print(name)

#MODE_USED = 'RNNModel'
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

# =================================================================
#             Guardado de informacion
# =================================================================
# import requests
# import pprint
# import pandas as pd

# url  = 'http://192.168.115.99:3333/getinvoices'

# response = requests.get(url)

# if response.status_code == 200:
#     invoices  = response.json()
#     # pprint.pprint(invoices)
# else: 
#     print(response.status_code)


# data = pd.DataFrame(invoices).head()
# print(data)

# """
# ITEMNMBR : codigo item
# QUANTITY : valor a predecir
# """
