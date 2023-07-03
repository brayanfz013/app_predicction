import os
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from src.lib.class_load import LoadFiles
from src.features.features_fix_data import PrepareData
from src.lib.factory_data import SQLDataSourceFactory, get_data
from src.lib.factory_models import ModelContext  # , Modelos, Parameters_model
from src.lib.factory_prepare_data import (DataCleaner, DataModel,
                                          MeanImputation, OutliersToIQRMean)


path_folder = os.path.dirname(__file__)
folder_model = Path(path_folder).joinpath('scr/data/save_models')

handler_load = LoadFiles()
ruta_actual = os.path.dirname(__file__)

# #=================================================================
# #             Cargar datos de la fuente de datos
# #=================================================================

CONFIG_FILE = ruta_actual+'/src/data/config/config.yaml'
with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    parameters = yaml.safe_load(file)

print("Probando el estacion de datos de sql")
data = get_data(SQLDataSourceFactory(**parameters))


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

#Patron de diseno de seleecion de estrategia
cleaner = DataCleaner(imputation)
data_imputation = cleaner.clean(data)

#Tener la informacion lista para las predicciones
parameters_filter = parameters['filter_data']
handle_data = PrepareData(data_imputation.dataframe,**parameters['query_template'])
handle_data.filter_column(
    parameters_filter['filter_1_column'],
    parameters_filter['filter_1_feature'],
    string_filter=False
)
handle_data.filter_column(
    parameters_filter['filter_2_column'],
    parameters_filter['filter_2_feature'],
    string_filter=True
)
handle_data.get_expand_date(parameters_filter['date_column'])
handle_data.set_index_col(parameters_filter['date_column'])
handle_data.group_by_time(parameters_filter['predict_column'],frequency_group='D')

MODE_USED = 'NBeatsModel'
modelo = ModelContext(model_name = MODE_USED,
                      data=handle_data.dataframe,
                      split=83,
                      **parameters)

#Rutas de los parametros para predicciones
model_train = modelo.save_path.joinpath('model').with_suffix('.pkl').as_posix()
scaler = modelo.save_path.joinpath('scaler').with_suffix('.pkl').as_posix()
last_pred =  modelo.save_path.joinpath('previus').with_suffix('.json').as_posix()

#Cargar escaler
scaler = handler_load.load_scaler(scaler)

#Cargar el ultimo punto de predicciones
last_prediction = handler_load.json_to_dict(last_pred)[0]
past, future = modelo.tunne_parameter.data.split_after(
    pd.Timestamp(last_prediction['last_date_pred']))

#Cargar modelo para hacer las predicciones