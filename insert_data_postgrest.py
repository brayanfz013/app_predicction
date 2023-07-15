'''
Codigo para insertar datos en  una tabla en postgres usando python junto con las funciones 
propiamente creadas
'''

from pathlib import Path
from src.features.features_postgres import HandleDBpsql

QUERY_DATA = '/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/querys/create_table.sql'
CONNECTION_PARAMETERS = '/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/config/database_postgrest.ini'
# LOGGER_CONFIG = '/home/bdebian/Documents/Projects/Stoke_prediccition/app_prediction/src/data/config/logging.conf'
DATA_TO_INSERT = '/mnt/data extra/DATASET/PREDICTION_DATA/Customer_Invoices.csv'

pg_handler = HandleDBpsql()

query_data_ready = pg_handler.prepare_query(QUERY_DATA)

pg_handler.insert_data_from_csv(
    connection_parameters=CONNECTION_PARAMETERS,
    query=query_data_ready, 
    data_path=DATA_TO_INSERT)