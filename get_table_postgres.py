'''
Codigo para traer datos de una tabla en postgres usando python junto con las funciones 
propiamente creadas
'''

from pathlib import Path
from src.features.features_postgres import HandleDBpsql

QUERY_DATA = '/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/querys/get_table.sql'
CONNECTION_PARAMETERS = '/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/config/database_postgrest.ini'
LOGGER_CONFIG = '/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/config/logging.conf'

pg_handler = HandleDBpsql(logfile=LOGGER_CONFIG)

query_data_ready = pg_handler.prepare_query(QUERY_DATA)

data_postgres = pg_handler.get_table(
    connection_parameters=CONNECTION_PARAMETERS,
    query=query_data_ready, 
    )