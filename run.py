''' 
Codigo para ejecutar un entrenamiento completo de una base de datos
'''


import os
import yaml
from src.lib.class_load import LoadFiles
from src.models.args_data_model import ParamsPostgres, Parameters
from src.lib.factory_data import SQLDataSourceFactory, get_data,HandleRedis
import redis 


handler_load = LoadFiles()
ruta_actual = os.path.dirname(__file__)
#=================================================================
#             Cargar datos de la fuente de datos 
#=================================================================

CONFIG_FILE = ruta_actual+'/src/data/config/config.yaml'
with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    parameters = yaml.safe_load(file)


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

# data = get_data(SQLDataSourceFactory(**parameters))

h_redis = HandleRedis()

parametros = Parameters(**parameters)

connection  = h_redis.get_config_file(parametros.connection_data_source)
print(connection)

pong = h_redis.check_connection(parametros.connection_data_source)
 
print(pong)