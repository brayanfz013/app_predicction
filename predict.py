import os
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from src.lib.class_load import LoadFiles
from src.features.features_fix_data import PrepareData
from src.lib.factory_data import SQLDataSourceFactory, get_data,create_table, set_data
from src.lib.factory_models import ModelContext
from src.lib.factory_prepare_data import (DataCleaner, DataModel,
                                          MeanImputation, OutliersToIQRMean)
from src.models.DP_model import Modelos

path_folder = os.path.dirname(__file__)
folder_model = Path(path_folder).joinpath('scr/data/save_models')

handler_load = LoadFiles()
ruta_actual = os.path.dirname(__file__)

# =================================================================
#             Cargar datos de la fuente de datos
# =================================================================
CONFIG_FILE = ruta_actual+'/src/data/config/config.yaml'
with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
    parameters = yaml.safe_load(file)

print("Probando el estacion de datos de sql")
data = get_data(SQLDataSourceFactory(**parameters))

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
    object:lambda x: x.strip()
    
}


# Imputacion de los datos
imputation = MeanImputation(
    replace_dtypes=new_types,
    strategy_imputation=strategy,
    preprocess_function=replace,
    **parameters
)
# Remocion de outliners y seleccion de columnas
outliners = OutliersToIQRMean(**parameters)

# Preparacion de los dato para el modelos escalizado y filtrado
data_for_model = DataModel(**parameters)

# Patron de diseno de seleecion de estrategia
cleaner = DataCleaner(imputation)
data_imputation = cleaner.clean(data)

# Cambio de estrategia para remover outliners
cleaner.strategy = outliners
data_filled = cleaner.clean(data_imputation.dataframe)

# Cambio de estrategia para preparar los datos para modelo
cleaner.strategy = data_for_model
data_ready, scaler_data = cleaner.clean(data_filled)

# =================================================================
#            Cargar modelo
# =================================================================
MODE_USED = 'NBeatsModel'
modelo = ModelContext(model_name=MODE_USED,
                      data=data_ready,
                      split=83,
                      **parameters)

# Rutas de los parametros para predicciones
model_train = modelo.save_path.joinpath('model').with_suffix('.pt').as_posix()
scaler = modelo.save_path.joinpath('scaler').with_suffix('.pkl').as_posix()
last_pred = modelo.save_path.joinpath('previus').with_suffix('.json').as_posix()
parameters_model = modelo.save_path.joinpath('parametros').with_suffix('.json').as_posix()


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

## Invertir Predicciones escaler de transformacion de los datos
# pred_scale = scaler_data.inverse_transform(pred_series)


data_frame_predicciones = pred_scale.pd_dataframe()
column_field = list(data_frame_predicciones.columns)
data_frame_predicciones['Varianza'] = data_frame_predicciones[column_field].pct_change() * 100
data_frame_predicciones.reset_index(inplace=True)

filter = []
for filter_list in parameters['filter_data']:
    if 'feature' in filter_list:
        filter.append(parameters['filter_data'][filter_list])

for adding_data  in filter:
    data_frame_predicciones[str(adding_data)] = adding_data

new_names = list(parameters['query_template_write']['columns'].values())
rename= {x:y for x,y in zip(list(data_frame_predicciones.columns),new_names)}
data_frame_predicciones.rename(columns=rename,inplace=True)

# Crear tabla para guardas la informacion 
create_table(SQLDataSourceFactory(**parameters))

set_data(SQLDataSourceFactory(**parameters),data_frame_predicciones)
