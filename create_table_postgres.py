'''
Codigo para crear una tabla en postgres usando python junto con las funciones 
propiamente creadas
'''

import os
from pathlib import Path
from src.features.features_postgres import HandleDBpsql



QUERY_DATA = '/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/querys/create_table.sql'
CONNECTION_PARAMETERS = '/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/config/database_postgrest.ini'
LOGGER_CONFIG = '/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/config/logging.conf'


pg_handler = HandleDBpsql(logfile=LOGGER_CONFIG)

parameter_connection = pg_handler.get_config_file(CONNECTION_PARAMETERS,section='postgresql')

#Check connection
# pg_handler.check_connection(CONNECTION_PARAMETERS)

query_data_ready = pg_handler.prepare_query(QUERY_DATA)

pg_handler.send_query(connection_parameters=CONNECTION_PARAMETERS, query=query_data_ready)

