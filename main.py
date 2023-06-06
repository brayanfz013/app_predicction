'''Codigo para la ejecucion  y llamando para hacer predicciones en los modelos'''

import json 

# from app_prediction.src.lib.factory_data import client_code, SQLDataSourceFactory, NoSQLDataSourceFactory,PlainTextFileDataSourceFactory
from src.lib.factory_data import get_data,SQLDataSourceFactory

CONFIG_FILE = "/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/parameter/data_params_run.json"

with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    parameters = json.load(file)


print("Probando el estacion de datos de sql")
get_data(SQLDataSourceFactory(**parameters))