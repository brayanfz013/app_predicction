'''Codigo para la ejecucion  y llamando para hacer predicciones en los modelos'''


import json
# from app_prediction.src.lib.factory_data import client_code, SQLDataSourceFactory, NoSQLDataSourceFactory,PlainTextFileDataSourceFactory
import os

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.lib.class_load import LoadFiles
from src.lib.factory_data import SQLDataSourceFactory, get_data
from src.lib.factory_models import ModelContext  # , Modelos, Parameters_model
from src.lib.factory_prepare_data import (DataCleaner, DataModel,
                                          MeanImputation, OutliersToIQRMean)
# from src.models.args_data_model import (ModelBlockRNN, ModelDLinearModel,
#                                         ModelExponentialSmoothing, ModelFFT,
#                                         ModelNBEATSModel, ModelNlinearModel,
#                                         ModelRNN, ModelTCNModel, ModelTFTModel,
#                                         ModelTransformerModel)
from src.models.DP_model import Modelos

handler_load = LoadFiles()

ruta_actual = os.path.dirname(__file__)
print(ruta_actual)
#=================================================================
#             Cargar datos de la fuente de datos 
#=================================================================

CONFIG_FILE = ruta_actual+'/src/data/config/config.yaml'
with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    parameters = yaml.safe_load(file)

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
    'string': 'object',
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
    float:lambda x: float(x.replace(',',''))
}


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

#Cambio de estrategia para preparar los datos para modelo
cleaner.strategy = data_for_model
data_ready,scaler_data = cleaner.clean(data_filled)

if not parameters['scale']:
    data_ready = scaler_data.inverse_transform(data_ready)

#=================================================================
#            Preparacion de modelo
#=================================================================
model_names = list(Modelos.keys())

# for name in model_names:
#     print(name)

# MODE_USED = 'RNNModel'
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


# if __name__ == '__main__':
#     import os
#     from pathlib import Path
#     path_folder = os.path.dirname(__file__)
#     print(path_folder)
#     print(Path(path_folder).parents[0])

#     # path_folder = str(Path(path_folder).parents)
#     # print(path_folder)